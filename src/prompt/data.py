import os
import json
import numpy as np
import cv2
from PIL import Image

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from prefetch_generator import BackgroundGenerator
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


class Image2TextDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform, mode='train'):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        long_caption = self.dataset[index]['caption']

        if self.mode == 'train':
            original_image_path = os.path.join(self.root_path, self.dataset[index]['image_path'])
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            negative_image_path = os.path.join(self.root_path, self.dataset[np.random.randint(1, len(self.dataset))]['image_path'])
            negative_image = self.image_transform(Image.open(negative_image_path).convert('RGB'))
            return [original_image, long_caption, negative_image]
        else:
            original_image_path = os.path.join(self.root_path, self.dataset[index]['image_path'])
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            return [original_image, long_caption]


class Image2ImageDataset(Dataset):
    def __init__(self, root_path, other_path, json_path, image_transform, mode='train'):
        self.root_path = root_path
        self.other_path = other_path
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
        self.mode = mode
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):

        if self.mode == 'train':
            original_image_path = os.path.join(self.root_path, self.dataset[index]['origin_image'])
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            classname = os.path.join(self.root_path, self.dataset[index]['classname'])
            
            other_image_path = os.path.join(self.other_path, self.dataset[index]['sketch_image'])
            other_image = self.image_transform(Image.open(other_image_path).convert('RGB'))
            
            num = np.random.randint(1, len(self.dataset))
            negative_class = os.path.join(self.other_path, self.dataset[num]['classname'])
            while negative_class == classname:
                num = np.random.randint(1, len(self.dataset))
                negative_class = os.path.join(self.other_path, self.dataset[num]['classname'])
            
            negative_image_path = os.path.join(self.other_path, self.dataset[num]['sketch_image'])
            negative_image = self.image_transform(Image.open(negative_image_path).convert('RGB'))

            return [original_image, other_image, negative_image]
        else:
            original_image_path = os.path.join(self.root_path, self.dataset[index]['sketch_image'])
            original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
            other_image_path = os.path.join(self.other_path, self.dataset[index]['image_path'])
            other_image = self.image_transform(Image.open(other_image_path).convert('RGB'))
            return [original_image, other_image]
        

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class data_prefetcher():
    def __init__(self, loader):
        #loader 1：real
        #loader 2：fake
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()
 
 
    def preload(self):
        try:
            self.oimage = next(self.loader)
        except StopIteration:
            self.oimage = None
            return
        with torch.cuda.stream(self.stream):
            self.oimage = [self.oimage[0].cuda(non_blocking=True).float(),
                        self.oimage[1].cuda(non_blocking=True).float(),
                        self.oimage[2].cuda(non_blocking=True).float()]
 

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        oimage = self.oimage
        self.preload()
        return oimage