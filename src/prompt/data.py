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
    def __init__(self, json_path, image_transform, mode='train'):
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        long_caption = self.dataset[index]['caption']

        if self.mode == 'train':
            original_image_path = '/public/home/jiayanhao/imagenet/train/' + self.dataset[index]['image_path']
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            negative_image_path = '/public/home/jiayanhao/imagenet/train/' + self.dataset[np.random.randint(1, len(self.dataset))]['image_path']
            negative_image = self.image_transform(Image.open(negative_image_path).convert('RGB'))
            return [original_image, long_caption, negative_image]
        else:
            original_image_path = '/public/home/jiayanhao/imagenet/val/' + self.dataset[index]['image_path']
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            return [original_image, long_caption]
