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
    file = open('imagenet/convert.txt', 'r')
    lines = file.readlines()
    for line in lines:
        folder = line.split(' ')[0]
        classname = line.split(' ')[1].strip('\n')
        convert[folder]=classname
    return convert


class I2TTrainDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform, tokenizer):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        # self.convert = get_classname()
        self.image_transform = image_transform
        # self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        long_caption = self.dataset[index]['caption']
        original_image_path = os.path.join(self.root_path, self.dataset[index]['image_path'])
        original_image = self.image_transform(Image.open(original_image_path))
        negative_image_path = os.path.join(self.root_path, self.dataset[np.random.randint(1, len(self.dataset))]['image_path'])
        negative_image = self.image_transform(Image.open(negative_image_path))
        return [original_image, long_caption, negative_image]


class I2TTestDataset(Dataset):
    def __init__(self, root_path, json_path, image_transform):
        self.root_path = root_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        long_caption = self.dataset[index]['caption']
        original_image_path = os.path.join(self.root_path, self.dataset[index]['image_path'])
        original_image = self.image_transform(Image.open(original_image_path).convert('RGB'))
        return [original_image, long_caption]


class I2ITrainDataset(Dataset):
    def __init__(self, root_path, other_path, json_path, image_transform):
        self.root_path = root_path
        self.other_path = other_path
        self.dataset = json.load(open(json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        classname = os.path.join(self.root_path, self.dataset[index]['classname'])

        original_image_path = os.path.join(self.root_path, self.dataset[index]['origin_image'])
        other_image_path = os.path.join(self.other_path, self.dataset[index]['art_image'])
        
        num = np.random.randint(1, len(self.dataset))
        negative_class = os.path.join(self.other_path, self.dataset[num]['classname'])
        while negative_class == classname:
            num = np.random.randint(1, len(self.dataset))
            negative_class = os.path.join(self.other_path, self.dataset[num]['classname'])
        negative_image_path = os.path.join(self.other_path, self.dataset[num]['art_image'])

        original_image = self.image_transform(Image.open(original_image_path))
        other_image = self.image_transform(Image.open(other_image_path))
        negative_image = self.image_transform(Image.open(negative_image_path))

        return [original_image, other_image, negative_image]


class I2ITestDataset(Dataset):
    def __init__(self, root_path, other_path, root_json_path, other_json_path, image_transform):
        self.root_path = root_path
        self.other_path = other_path
        self.root_dataset = json.load(open(root_json_path,'r'))
        self.other_dataset = json.load(open(other_json_path,'r'))
        self.convert = get_classname()
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.other_dataset)
    
    def __getitem__(self, index):
        original_image_path = os.path.join(self.root_path, self.root_dataset[index]['image'])
        original_image = self.image_transform(Image.open(original_image_path))
        original_classname = self.root_dataset[index]['classname']
        other_image_path = os.path.join(self.other_path, self.other_dataset[index]['image'])
        other_image = self.image_transform(Image.open(other_image_path))
        other_image_classname = self.other_dataset[index]['classname']
        return [original_image, other_image, original_classname, other_image_classname]
        

class I2MTrainDataset(Dataset):
    def __init__(self, root_path, other_path, json_path, image_transform):
        self.root_path = root_path
        self.other_path = other_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        original_image_path = os.path.join(self.root_path, self.dataset[index]['image'])
        other_image_path = os.path.join(self.other_path, self.dataset[index]['image'])
        negative_image_path = os.path.join(self.root_path, self.dataset[np.random.randint(1, len(self.dataset))]['image'])

        original_image = self.image_transform(Image.open(original_image_path))
        other_image = self.image_transform(Image.open(other_image_path))
        negative_image = self.image_transform(Image.open(negative_image_path))

        return [original_image, other_image, negative_image]


class I2MTestDataset(Dataset):
    def __init__(self, root_path, other_path, json_path, image_transform):
        self.root_path = root_path
        self.other_path = other_path
        self.dataset = json.load(open(json_path,'r'))
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        original_image_path = os.path.join(self.root_path, self.dataset[index]['image_path'])
        other_image_path = os.path.join(self.other_path, self.dataset[index]['image_path'])
        original_image = self.image_transform(Image.open(original_image_path))
        other_image = self.image_transform(Image.open(other_image_path))
        return [original_image, other_image]


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
    

class DataPrefetcher():
    def __init__(self, loader):
        self.stream = torch.cuda.Stream()
        self.loader = iter(loader)
        self.preload()
 

    def preload(self):
        try:
            self.data = next(self.loader)
        except StopIteration:
            self.data = None
            return
        with torch.cuda.stream(self.stream):
            temp = []
            for item in range(len(self.data)):
                temp.append(self.data[item].cuda(non_blocking=True))
            self.data = temp
 

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        self.preload()
        return data