import argparse
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.prompt.model import OpenCLIP
from src.prompt.data import I2TTestDataset, I2ITestDataset, I2MTestDataset
from src.prompt.utils import init_distributed_mode, setup_seed, save_loss, getI2TR1Accuary, getI2IR1Accuary

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # project settings
    parser.add_argument('--output_dir', default='SMR/output/')
    parser.add_argument('--out_path', default='origin-mosaic-loss.jpg')
    parser.add_argument('--resume', default='SMR/output/0-4sketch_epoch2.pth', type=str, help='load checkpoints from given path')
    parser.add_argument('--device', default='cuda:3')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument("--local_rank", type=int)

    # data settings
    parser.add_argument("--type", type=str, default='image2text', help='choose train image2text or image2image.')
    parser.add_argument("--test_ori_dataset_path", type=str, default='imagenet/imagenet-p/val/')
    parser.add_argument("--test_art_dataset_path", type=str, default='imagenet/imagenet-sketch/val/')
    parser.add_argument("--test_json_path", type=str, default='imagenet/200-original-long-val.json')
    parser.add_argument("--test_other_json_path", type=str, default='imagenet/sketch.json')
    parser.add_argument("--test_batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def eval(args, model, dataloader):
    model.eval()

    if args.type == 'image2text':
        for data in tqdm(enumerate(dataloader)):
            image = data[1][0].to(device, non_blocking=True)
            long_caption = model.tokenizer(data[1][1]).to(device, non_blocking=True)

            image_feature = model(image, dtype='image')
            text_feature = model(long_caption, dtype='text')

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * image_feature @ text_feature.T), dim=-1)

            acc = getI2TR1Accuary(prob)

            print(acc)
    else:
        for data in enumerate(dataloader):
            origin_image = data[1][0].to(device, non_blocking=True)
            retrival_image = data[1][1].to(device, non_blocking=True)
            ori_classname = data[1][2]
            retri_classname = data[1][3]

            original_feature = model(origin_image, dtype='image')
            retrival_feature = model(retrival_image, dtype='image')

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * original_feature @ retrival_feature.T), dim=-1)

            acc = getI2IR1Accuary(prob, ori_classname, retri_classname)
            # acc = getI2TR1Accuary(prob)

            print(acc)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    model = OpenCLIP(args)
    model = model.to(device)

    test_dataset = I2TTestDataset(args.test_ori_dataset_path, args.test_json_path, model.pre_process_val)

    # test_dataset = I2ITestDataset(args.test_ori_dataset_path, args.test_art_dataset_path, args.test_json_path, args.test_other_json_path, model.pre_process_val)
    
    # test_dataset = I2MTestDataset(args.test_ori_dataset_path, args.test_art_dataset_path, args.test_json_path, model.pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.test_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=True,
                            drop_last=True
                            )
    
    eval(args, model, test_loader)