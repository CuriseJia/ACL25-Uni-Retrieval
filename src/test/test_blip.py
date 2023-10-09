import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import time
from fvcore.nn import FlopCountAnalysis, parameter_count_table

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchstat import stat

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from models.blip_retrieval import blip_retrieval
import utils
from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader


def getI2TR1Accuary(prob):
    temp = prob.detach().cpu().numpy()
    temp = np.argsort(temp, axis=1)
    count = 0
    for i in range(prob.shape[0]):
        if temp[i][prob.shape[1]-1] == i:
            count+=1
    acc = count/prob.shape[0]
    return acc

def load_image(image, image_size, device, batch=16):
    # print('start load image')
    images = []
    for i in range(batch):
        raw_image = Image.open(image[i]).convert('RGB')

        w, h = raw_image.size

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        # img = transform(raw_image).unsqueeze(0).to(device)
        img = transform(raw_image).to(device)
        images.append(img)
    
    imgs = torch.stack(images)

    return imgs


text_list=[]
ori_image=[]
sketch_image=[]

device = "cuda:5" if torch.cuda.is_available() else "cpu"

model = blip_retrieval(pretrained='BLIP/model_large_retrieval_coco.pth', image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
model.eval()
model.to(device)


pair = json.load(open('imagenet/200-original-long-val.json', 'r'))
acc = []
for i in tqdm(range(0, 10)):
    ori_image.clear()
    sketch_image.clear()
    
    for j in range(64):
        index = random.randint(0, len(pair))
        # text_list.append(pair[index]['caption'])
        ori_image.append(os.path.join('imagenet/val/', pair[index]['image_path']))
        sketch_image.append(os.path.join('imagenet/imagenet-p/val/', pair[index]['image_path']))

    ori_images = load_image(ori_image, 224, device, 16)
    sketch_images = load_image(sketch_image, 224, device, 16)

    ori_feat = model.visual_encoder(ori_images)   
    ori_embed = model.vision_proj(ori_feat)            
    ori_embed = F.normalize(ori_embed,dim=-1)

    ske_feat =  model.visual_encoder(sketch_images)
    ske_embed = model.vision_proj(ske_feat)  
    ske_embed = F.normalize(ske_embed,dim=-1)


    

    prob = torch.softmax(ori_embed @ ske_embed.T, dim=-1)

    acc.append(getI2TR1Accuary(prob))


r1 = sum(acc)/len(acc)
print(r1)