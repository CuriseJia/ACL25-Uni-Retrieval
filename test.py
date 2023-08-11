import argparse
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import open_clip
from PIL import Image

from src.prompt.model import ShallowPromptTransformer
from main_1gpu import parse_args
# from src.prompt.data import OriginalLongDataset
# from src.prompt.utils import init_distributed_mode, setup_seed, get_rank, get_world_size, is_main_process, save_loss

args = parse_args()

model = ShallowPromptTransformer(args)
model = model.to('cuda:1')

model.load_state_dict(torch.load('output/epoch_i2s_0_10_1.pth'))

ori_image_1 = model.pre_process_val(Image.open('../imagenet/val/n01440764/ILSVRC2012_val_00002138.JPEG')).to('cuda:1', non_blocking=True)
ori_image_2 = model.pre_process_val(Image.open('../imagenet/val/n01514859/ILSVRC2012_val_00014879.JPEG')).to('cuda:1', non_blocking=True)
ori_image_3 = model.pre_process_val(Image.open('../imagenet/val/n01631663/ILSVRC2012_val_00007998.JPEG')).to('cuda:1', non_blocking=True)
ori_image_4 = model.pre_process_val(Image.open('../imagenet/val/n01742172/ILSVRC2012_val_00018907.JPEG')).to('cuda:1', non_blocking=True)
ori_image_5 = model.pre_process_val(Image.open('../imagenet/val/n01855672/ILSVRC2012_val_00018602.JPEG')).to('cuda:1', non_blocking=True)
re_image_1 = model.pre_process_val(Image.open('../imagenet/imagenet-sketch/n01440764/sketch_13.JPEG')).to('cuda:1', non_blocking=True)
re_image_2 = model.pre_process_val(Image.open('../imagenet/imagenet-sketch/n01514859/sketch_7.JPEG')).to('cuda:1', non_blocking=True)
re_image_3 = model.pre_process_val(Image.open('/public/home/jiayanhao/imagenet/imagenet-sketch/n01631663/sketch_6.JPEG')).to('cuda:1', non_blocking=True)
re_image_4 = model.pre_process_val(Image.open('/public/home/jiayanhao/imagenet/imagenet-sketch/n01742172/sketch_3.JPEG')).to('cuda:1', non_blocking=True)
# re_image_5 = model.pre_process_val(Image.open('/public/home/jiayanhao/imagenet/imagenet-sketch/n01855672/sketch_19.JPEG')).to('cuda:1', non_blocking=True)

ori_image = torch.stack([ori_image_1, ori_image_2, ori_image_3, ori_image_4, ori_image_5],dim=0)
re_image = torch.stack([re_image_1, re_image_2, re_image_3, re_image_4],dim=0)

ori_fea = model(ori_image, dtype='image')
re_fea = model(re_image, dtype='image')

ori_fea = F.normalize(ori_fea, dim=-1)
re_fea = F.normalize(re_fea, dim=-1)

prob = torch.softmax((100.0 * ori_fea @ re_fea.T), dim=-1)

print(prob.detach().cpu().numpy())