import os
import sys
sys.path.append('/mnt/sdb/data/jingyinuo/code/Video-QA')
import torch
import io
import json
from video_chat.models.videochat import VideoChat
from video_chat.utils.easydict import EasyDict
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from video_chat.models.video_transformers import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms
# from dataset.video_transforms import (
#     GroupNormalize, GroupScale, GroupCenterCrop, 
#     Stack, ToTorchFormatTensor
# )

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Video, HTML
from peft import get_peft_model, LoraConfig, TaskType
import copy
import imageio
import ffmpeg

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

def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    # if print_res:
    #     # print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to("cuda:0").input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs

def ask(text, conv):
    conv.messages.append([conv.roles[0], text + '\n'])

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
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([835]).to("cuda"),
        torch.tensor([2277, 29937]).to("cuda")]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text, output_token.cpu().numpy()

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
class MVBench_dataset(Dataset):
    def __init__(self, data_dir, data_list, e=None, num_segments=8, resolution=224):
        self.data_list = []
        self.e = e
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


    def read_video(self, video_path, bound):
        vr = VideoReader(video_path, ctx=cpu(0))
        num_frames = len(vr)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0) 

        # transform
        crop_size = 224
        scale_size = 224
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]

        transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std) 
        ])

        images_group = list()
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = transform(images_group)
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg

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
        print(video_path)
        torch_imgs, msg = decord_method(video_path, bound)
        question, answer = self.qa_template(self.data_list[idx]['data'])
            
        return {
            'video': torch_imgs, 
            'msg': msg,
            'question': question, 
            'answer': answer,
            'task_type': self.data_list[idx]['task_type']
        }

# %%
def infer_mvbench(
        data_sample, system="", 
        question_prompt='', # add in the end of question
        answer_prompt=None, # add in the begining of answer
        return_prompt='',  # add in the begining of return message
        system_q=False, # whether add question in the system prompt for QFormer
        print_res=False,
        system_llm=False
    ):
    video = data_sample["video"]
    msg = data_sample["msg"]
    TC, H, W = video.shape
    video = video.reshape(1, TC//3, 3, H, W).to("cuda")
    
    video_list = []
    with torch.no_grad():
        video_emb, _ = model.encode_img(video)
    video_list.append(video_emb)
#     video_list.append(torch.zeros_like(video_emb))
    
    chat = EasyDict({
    "system": system,
    "roles": ("Human", "Assistant"),
    "messages": [],
    "sep": "###"
})

    chat.messages.append([chat.roles[0], f"<Video><VideoHere></Video> {msg}\n"])
    
    if system_llm:
        prompt = system + data_sample['question'] + question_prompt
    else:
        prompt = data_sample['question'] + question_prompt
    
    ask(prompt, chat)
    
    llm_message = answer(
        conv=chat, model=model, do_sample=False, 
        img_list=video_list, max_new_tokens=100, 
        answer_prompt=answer_prompt, print_res=print_res
    )[0]
    # remove potential explanation
    llm_message = return_prompt + llm_message.strip().split('\n')[0]
    print(llm_message)
    print(f"GT: {data_sample['answer']}")
    return llm_message

# %%
# model
from video_chat.utils.config import Config
config_file = "/mnt/sdb/data/jingyinuo/code/Video-QA/video_chat/configs/config_7b.json"
cfg = Config.from_file(config_file)
model = VideoChat(config=cfg.model)
model = model.to(torch.device(cfg.device))
model = model.eval()

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

# %%
num_frame = 16
resolution = 224
e = None
dataset = MVBench_dataset(data_dir, data_list, e, num_segments=num_frame, resolution=resolution)

# %%
# test
save_path = "/mnt/sdb/data/jingyinuo/code/Video-QA/Animal-Bench/results/videochat"
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


