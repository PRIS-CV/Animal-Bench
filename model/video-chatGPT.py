import os
import io
import json

import sys
sys.path.append('/mnt/sdb/data/jingyinuo/code/Video-QA')

from video_chat2.utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from video_chat2.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset

from torchvision import transforms

import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import argparse
from VideoChatGPT.video_chatgpt.eval.model_utils import initialize_model, load_video
from VideoChatGPT.video_chatgpt.inference import video_chatgpt_infer
from VideoChatGPT.video_chatgpt.video_conversation import conv_templates, SeparatorStyle
from VideoChatGPT.video_chatgpt.model.utils import KeywordsStoppingCriteria

# %%
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, num_segments=8, resolution=224):
        self.data_list = []
        for k, v in data_list.items():
            for i in v:
                with open(os.path.join(data_dir, i[0]), 'r') as f:
                    json_data = json.load(f)
                for data in json_data:
                    self.data_list.append({
                        'task_type': k,
                        'prefix': i[1],
                        'data_type': i[2],
                        'bound': i[3],
                        'data': data
                    })
        
        self.decord_method = {
            'video': self.read_video,
        }
        
        self.num_segments = num_segments
        
        # transform
        crop_size = resolution
        scale_size = resolution
        self.crop_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])
    
    def __str__(self):
        len_list = {}
        option_list = {}
        for data in self.data_list:
            if data['task_type'] not in len_list:
                len_list[data['task_type']] = 0
            len_list[data['task_type']] += 1
            if data['task_type'] not in option_list:
                option_list[data['task_type']] = 0
            option_list[data['task_type']] += len(data['data']['candidates'])
        
        correct = 0
        total = 0
        res = f"There are {len(self.data_list)} videos as follow:\n"
        for k, v in len_list.items():
            correct += len_list[k]
            total += option_list[k]
            res += f"{v} for {k} ({option_list[k]} options => {len_list[k]/option_list[k]*100:.2f}%)\n"
            correct = correct + 1 / option_list[k]
        res += f"Total random accuracy: {correct/total*100:.2f}%"
        return res.rstrip()
        
    def __len__(self):
        return len(self.data_list)
    
    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices
    
    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 
        img_array = vr.get_batch(frame_indices).numpy()
        # Set target image height and width
        target_h, target_w = self.crop_size, self.crop_size
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(img_array, size=(target_h, target_w))
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        # Reshape array to match number of clips and frames
        n_clips = 1
        img_array = img_array.reshape(
            (n_clips, self.num_segments, img_array.shape[-3], img_array.shape[-2], img_array.shape[-1]))
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(self.num_segments)]
        return clip_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        '''
        question格式:
        Question: ~~
        Option:
        '''
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(data['candidates']):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def __getitem__(self, idx):
        decord_method = self.decord_method[self.data_list[idx]['data_type']]
        bound = None
        if self.data_list[idx]['bound']:
            bound = (
                self.data_list[idx]['data']['start'],
                self.data_list[idx]['data']['end'],
            )
        video_path = os.path.join(self.data_list[idx]['prefix'], self.data_list[idx]['data']['video'])
        
        print(video_path)
        
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens

# %%
def infer_mvbench(
        conv_mode,
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False
    ):
    video_frames = data_sample["video"]
    
    video_list = []
    with torch.no_grad():
        # Preprocess video frames and get image tensor
        image_tensor = image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

        # Move image tensor to GPU and reduce precision to half
        image_tensor = image_tensor.half().cuda()

        # Generate video spatio-temporal features
        with torch.no_grad():
            image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
            frame_features = image_forward_outs.hidden_states[-2][:, 1:] # Use second to last layer as in LLaVA
        video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    if system_llm:
        question = system + data_sample['question'] + question_prompt
    else:
        question = data_sample['question'] + question_prompt
    if model.get_model().vision_config.use_vid_start_end:
        qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len + DEFAULT_VID_END_TOKEN
    else:
        qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.system = system
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer_prompt)
    prompt = conv.get_prompt()
    prompt = prompt.replace('</s>', '')

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=True,
            temperature=1.0,
            max_new_tokens=200,
            stopping_criteria=[stopping_criteria])

    # Check if output is the same as input
    n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

    # Decode output tokens
    outputs = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()
    outputs = return_prompt + outputs
    print(outputs)
    print(f"GT: {data_sample['answer']}")
    return outputs

def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True
        
    return flag
# %%
# model
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

model_name = '/mnt/sdb/data/jingyinuo/code/model/LLaVA-Lightening-7B-v1-1'
projection_path = '/mnt/sdb/data/jingyinuo/code/model/Video-ChatGPT-7B-v1-1/video_chatgpt-7B.bin'
conv_mode = 'video-chatgpt_v1'

model, vision_tower, tokenizer, image_processor, video_token_len = initialize_model(model_name, projection_path)
model = model.eval()
# %%
# dataset
data_list = {
    "Object Existence": [("object_existence.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Object Recognition": [("AK_object_recognition.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_object_recognition.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Action Recognition": [("AK_action_recognition.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_action_recognition.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Action Sequence": [("action_sequence.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)], # has start & end
    "Action Prediction": [("action_prediction.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)], # has start & end
    "Action Localization": [("action_localization_new.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)],
    "Reasoning": [("reasoning.json", "/mnt/sdb/data/jingyinuo/dataset/NExT-QA/NExTVideo", "video", False)],
    "Object Count": [("object_count.json", "/mnt/sdb/data/jingyinuo/dataset/MSRVTT-QA/data/train-video", "video", False)],
    "Action Count": [("action_count.json", "/mnt/sdb/data/jingyinuo/dataset/TGIF-QA/gifs", "video", False)],
    "PA": [("AK_pa.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_pa.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "BB": [("AK_bb.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_bb.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False), ("LoTE_bb.json", "/mnt/sdb/data/jingyinuo/LoTE-Animal/data", "video", False)],
    "SI": [("AK_si.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_si.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False), ("LoTE_si.json", "/mnt/sdb/data/jingyinuo/LoTE-Animal/data", "video", False)],
    "DS": [("AK_ds.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_ds.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
}

data_dir = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/data"

num_frame = 16
resolution = 224

dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)

# %%
# test
save_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/results/videochatgpt"

correct = 0
total = 0
res_list = []
acc_dict = {}

for example in tqdm(dataset):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    
    pred = infer_mvbench(
        conv_mode,
        example, 
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of animals. Based on your observations, select the best option that accurately addresses the question.\n",
        question_prompt="\nOnly give the best option.",
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=True,
        system_llm=True
    )
    gt = example['answer']
    res_list.append({
        'pred': pred,
        'gt': gt
    })
    if check_ans(pred=pred, gt=gt):
        acc_dict[task_type][0] += 1
        correct += 1
    print(f"Part  Acc: {acc_dict[task_type][0] / acc_dict[task_type][1] * 100 :.2f}%")
    print(f"Total Acc: {correct / total * 100 :.2f}%")
    print('-' * 30, task_type, '-' * 30)

with open(f"{save_path}.json", "w") as f:
    json.dump({
        "acc_dict": acc_dict,
        "res_list": res_list
    }, f)