# %%
import torch
import sys
import os
sys.path.append('/mnt/sdb/data/jingyinuo/code/Video-QA')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
from transformers import AutoTokenizer
import torch
from enum import Enum

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
import cv2
import numpy as np
import random
import io
import json

from Valley.valley.model.valley_model import ValleyLlamaForCausalLM
from Valley.valley.util.config import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_VIDEO_FRAME_TOKEN,
    DEFAULT_VI_START_TOKEN,
    DEFAULT_VI_END_TOKEN,
    DEFAULT_VIDEO_TOKEN,
)

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
        return torch_imgs

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
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
        
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video_file': video_path,
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }
        
def infer_mvbench(
        data_sample, tokenizer, device, model, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
    ):
    video_file = data_sample['video_file']
    print(video_file)
    images = data_sample['video']

    message = [
    {"role": "system", "content": system},
]

    gen_kwargs = dict(
        do_sample=True,
        temperature=1.0,
        max_new_tokens=200,
    )
    
    if system_llm:
        prompt = f"{DEFAULT_VIDEO_TOKEN} " + system + data_sample['question'] + question_prompt
    else:
        prompt = f"{DEFAULT_VIDEO_TOKEN} " + data_sample['question'] + question_prompt
    
    message.append({"role": "user", "content": prompt + '\n'})
    message.append({"role": "assistent", "content": answer_prompt})
    llm_message = model.completion(tokenizer, images, message, gen_kwargs, device)
    # remove potential explanation
    llm_message = return_prompt + llm_message[0]
    # llm_message = llm_message[0]
    print(llm_message)
    print(f"GT: {data_sample['answer']}")
    return llm_message

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
def init_vision_token(model, tokenizer):
    vision_config = model.get_model().vision_tower.config
    (
        vision_config.im_start_token,
        vision_config.im_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
    (
        vision_config.vi_start_token,
        vision_config.vi_end_token,
    ) = tokenizer.convert_tokens_to_ids([DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN])
    vision_config.vi_frame_token = tokenizer.convert_tokens_to_ids(
        DEFAULT_VIDEO_FRAME_TOKEN
    )
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids(
        [DEFAULT_IMAGE_PATCH_TOKEN]
    )[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'luoruipu1/Valley2-7b'
model = ValleyLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
init_vision_token(model, tokenizer)

model = model.bfloat16().to(device)
model.eval()

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

dataset = MVBench_dataset(data_dir, data_list, num_segments=num_frame, resolution=resolution)

# %%
# test
save_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/results/valley"

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
        device,
        model,
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