import os
import io
import json
import sys
sys.path.append('/mnt/sdb/data/jingyinuo/code/Video-QA/')

from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
from decord import VideoReader, cpu
import numpy as np
from video_chat2.utils.easydict import EasyDict
import torch
from transformers import StoppingCriteria, StoppingCriteriaList

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
# %%
def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + ": " + message + conv.sep
        else:
            ret += role + ":"
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + ": " + message
        else:
            if message:
                ret += role + ": " + message + conv.sep
            else:
                ret += role + ":"
    return ret

class Animal_Bench(Dataset):
    def __init__(self, data_dir, data_list, e=None, num_segments=8, resolution=224):
        self.data_list = []
        self.e = e
        self.num_segments = num_segments
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
        
        # transform
        crop_size = resolution
        scale_size = resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            # Stack(),
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
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].numpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs, torch_imgs.shape[0]

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
        if self.e:
            video_path = video_path.replace('/mnt/sdb/data/jingyinuo/','/mnt/sdb/data/jingyinuo/dataset/edit/' + self.e + '/final/')
        print(video_path)
        
        video_frame, slice_len = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': video_frame,
            'slice_len': slice_len, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }


# %%
def infer_mvbench(
        data_sample, tokenizer, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False
    ):
    
    video_frames = data_sample['video']
    # print(video)
#     video_list.append(torch.zeros_like(video_emb))

    if system_llm:
        qs = system + data_sample['question'] + question_prompt
    else:
        qs = data_sample['question'] + question_prompt
        
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN * data_sample['slice_len'] + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN * data_sample['slice_len'] + '\n' + qs

    chat = EasyDict({
        "system": system,
        "roles": ("Human", "Assistant"),
        "messages": [],
        "sep": "###"
    })

    conv = chat
    conv.messages.append([conv.roles[0], qs])
    conv.messages.append([conv.roles[1], answer_prompt])
    
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    
    temperature = 1.0
    top_p = 0.9
    num_beams = 1
    
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
        0).cuda()

    stop_str = conv["sep"]
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
                    input_ids,
                    images=video_frames.half().cuda(),
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    num_beams=num_beams,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=200,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])
    output_ids = output_ids.sequences
    input_token_len = input_ids.shape[1]
    
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = return_prompt + outputs.strip()
    print(outputs)
    print(f"GT: {data_sample['answer']}")
    return outputs

# %%
def check_ans(pred, gt):
    flag = False
    
    pred_list = pred.lower().split(' ')
    # print(pred_list)
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
# load model
model_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/ChatUniVi/Chat-Univi-7B"  # or "Chat-UniVi/Chat-UniVi-13B"

disable_torch_init()
model_name = "ChatUniVi"
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
if mm_use_im_patch_token:
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
if mm_use_im_start_end:
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
model.resize_token_embeddings(len(tokenizer))

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
image_processor = vision_tower.image_processor

if model.config.config["use_cluster"]:
    for n, m in model.named_modules():
        m = m.to(dtype=torch.bfloat16)
# %%
# dataset
data_list = {
    "Object Existence": [("object_existence.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Object Recognition": [("AK_object_recognition.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_object_recognition.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Action Recognition": [("AK_action_recognition.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_action_recognition.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "Action Sequence": [("action_sequence.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)], # has start & end
    "Action Prediction": [("action_prediction.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)], # has start & end
    "Action Localization": [("action_localization.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video_grounding/dataset", "video", True)],
    "Reasoning": [("reasoning.json", "/mnt/sdb/data/jingyinuo/dataset/NExT-QA/NExTVideo", "video", False)],
    "Object Count": [("object_count.json", "/mnt/sdb/data/jingyinuo/dataset/MSRVTT-QA/data/train-video", "video", False)],
    "Action Count": [("action_count.json", "/mnt/sdb/data/jingyinuo/dataset/TGIF-QA/gifs", "video", False)],
    "PM": [("AK_pm.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_pm.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
    "BM": [("AK_bm.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_bm.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False), ("LoTE_bm.json", "/mnt/sdb/data/jingyinuo/LoTE-Animal/data", "video", False)],
    "SA": [("AK_sa.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_sa.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False), ("LoTE_sa.json", "/mnt/sdb/data/jingyinuo/LoTE-Animal/data", "video", False)],
    "PD": [("AK_pd.json", "/mnt/sdb/data/jingyinuo/animal_kingdom/video", "video", False), ("mmnet_pd.json", "/mnt/sdb/data/jingyinuo/mmnet/trimmed_video", "video", False)],
}

data_dir = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/data"

num_frame = 16
resolution = 224
e = None
dataset = Animal_Bench(data_dir, data_list, e, num_segments=num_frame, resolution=resolution)

# %%
# test
save_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/results/chatunivi"

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
        example, 
        tokenizer,
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of animals. Based on your observations, select the best option that accurately addresses the question.\n",
        question_prompt="\nOnly give the best option.",
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=True,
        system_llm=True,
        # image_processor=None,
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