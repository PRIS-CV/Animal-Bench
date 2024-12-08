import os
import torch
import sys
sys.path.append('/mnt/sdb/data/jingyinuo/code/Video-QA')
from VideoLLaVA.llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from VideoLLaVA.llava.conversation import conv_templates, SeparatorStyle
from VideoLLaVA.llava.model.builder import load_pretrained_model
from VideoLLaVA.llava.utils import disable_torch_init
from VideoLLaVA.llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria

import subprocess

import io
import json

from video_chat2.utils.easydict import EasyDict

import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
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

import imageio

import cv2

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


def ask(text, conv):
    conv.messages.append([conv.roles[0], DEFAULT_X_TOKEN['VIDEO'] + '\n' + text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False, tokenizer=None):
    conv.append_message(conv.roles[1], answer_prompt)
    # if answer_prompt:
    #     prompt = get_prompt2(conv)
    # else:
    #     prompt = get_prompt(conv)
    prompt = conv.get_prompt()
    prompt = prompt.replace('</s>', '')
    # print(prompt)
    key = ['video']
    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(0).to('cuda')
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=[img_list, key],
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])
    
    output_token = output_ids[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
            
    # output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    # output_text = output_text.split('###')[0]  # remove the stop sign '###'
    # output_text = output_text.split('Assistant:')[-1].strip()
    
    output_text = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    output_text = output_text.split(conv.sep)[0]  # remove the stop sign '###'
    output_text = output_text.split('ASSISTANT:')[-1].strip()
    # conv.messages[-1][1] = output_text
    
    # input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    # outputs = outputs.strip()
    # if outputs.endswith(stop_str):
    #     outputs = outputs[:-len(stop_str)]
    # outputs = outputs.strip()
    
    return output_text, output_token.cpu().numpy()

class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, e=None, num_segments=8, resolution=224):
        self.e = e
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
        if self.e:
            video_path = video_path.replace('/mnt/sdb/data/jingyinuo/','/mnt/sdb/data/jingyinuo/dataset/edit/' + self.e + '/final/')
        torch_imgs = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video_path': video_path,
            'video': torch_imgs, 
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type'],
            'bound': bound
        }
        
def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=True,
        system_llm=False,
        tokenizer=None,
        processor=None,
    ):
    video = data_sample["video_path"]
    bound = data_sample['bound']
    print(video)
    # tensor = data_sample['video']
    # tensor = tensor.permute(1,0,2,3).unsqueeze(0).float().half()
    # print(tensor.dtype)
    video_processor = processor['video']
    if bound:
        video_tensor = video_processor(video, clip_start_sec=bound[0], clip_end_sec=bound[1], return_tensors='pt')['pixel_values']
    else:
        video_tensor = video_processor(video, return_tensors='pt')['pixel_values']        
    if type(video_tensor) is list:
        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
    else:
        tensor = video_tensor.to(model.device, dtype=torch.float16)
    # print(tensor.dtype)
    conv_mode = "llava_v1"
    conv = []
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    # chat = EasyDict({
    #     "system": system,
    #     "roles": ("USER", "ASSISTANT"),
    #     "messages": [],
    #     "sep": "</s>"
    # })

    # chat.messages.append([chat.roles[0], f"<Video>\n"])
    conv.system = system
    if system_llm:
        prompt = DEFAULT_X_TOKEN['VIDEO'] + '\n' + system + data_sample['question'] + question_prompt # DEFAULT_X_TOKEN['VIDEO'] + '\n' + 
    else:
        prompt = DEFAULT_X_TOKEN['VIDEO'] + '\n' + data_sample['question'] + question_prompt # DEFAULT_X_TOKEN['VIDEO'] + '\n' + 
        
    conv.append_message(conv.roles[0], prompt)
    
    llm_message = answer(
        conv=conv, model=model, do_sample=False, 
        img_list=tensor, max_new_tokens=200, 
        answer_prompt=answer_prompt, print_res=print_res, tokenizer=tokenizer,
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0].replace('</s>', '')
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
    # if pred_option.replace('.', '') == gt_option:
    #     flag = True
    # if pred_content in gt_content:
    #     flag = True
    # elif gt_content in pred_content:
    #     flag = True
        
    return flag

# %%
# model
disable_torch_init()
model_path = '/mnt/sdb/data/jingyinuo/code/LanguageBind/Video-LLaVA-7B'
device = 'cuda'
load_4bit, load_8bit = True, False
model_name = get_model_name_from_path(model_path)
tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device) # , load_8bit, load_4bit, device=device
save_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/results/videollava2"

correct = 0
total = 0
res_list = []
acc_dict = {}

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
dataset = MVBench_dataset(data_dir, data_list, e, num_segments=num_frame, resolution=resolution)

for example in tqdm(dataset):
    task_type = example['task_type']
    if task_type not in acc_dict:
        acc_dict[task_type] = [0, 0] # correct, total
    acc_dict[task_type][1] += 1
    total += 1
    pred = infer_mvbench(
        example, 
        system="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of animals. Based on your observations, select the best option that accurately addresses the question.\n",
        question_prompt="\nOnly give the best option.",
        answer_prompt="Best option:(",
        return_prompt='(',
        system_q=False,
        print_res=True,
        system_llm=True,
        tokenizer=tokenizer,
        processor=processor,
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