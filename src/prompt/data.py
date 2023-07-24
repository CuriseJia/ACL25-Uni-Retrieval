import json
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
import torch

def get_classname():
    convert={}
    file = open('/public/home/jiayanhao/imagenet/convert.txt', 'r')
    lines = file.readlines()
    for line in lines:
        folder = line.split(' ')[0]
        classname = line.split(' ')[1].strip('\n')
        convert[folder]=classname
    return convert




class SixModalDataset(Dataset):
    def __init__(self, json_path, modal):
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()

    
    def __len__(self):
        len = 0
        for item in self.convert.values():
            length += len(self.dataset['original'][item])
        return len
    
    def __getitem__(self, index):
        original_image_path = '/public/home/jiayanhao/imagenet/train' + self.dataset['original'][index]['image']
        image = Image.open(original_image_path).convert('RGB')

        return image


class OriginalLongDataset(Dataset):
    def __init__(self, json_path, image_transform):
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        original_image_path = '/public/home/jiayanhao/imagenet/train/' + self.dataset[index]['image_path']
        image = self.image_transform(Image.open(original_image_path).convert('RGB'))
        long_caption = self.dataset[index]['caption']
        negative_feature = torch.zeros(512, dtype=torch.float32)

        return [image, long_caption, negative_feature]
