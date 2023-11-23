import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models import Uni_Retrieval
from src.models import I2ITestDataset, I2ITrainDataset, I2MTestDataset, I2MTrainDataset, I2TTestDataset, I2TTrainDataset
from src.models import setup_seed, getI2TR1Accuary, getI2IR1Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for FreeStyleRet-BLIP test on ImageNet-X Dataset.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load model checkpoint from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--gram_encoder_path', default='pretrained/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--style_prompt_path', default='pretrained/style_cluster.npy', type=str, help='load vgg from given path')

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train test2image or style2image.')
    parser.add_argument("--root_json_path", type=str, default='imagenet/test.json')
    parser.add_argument("--other_json_path", type=str, default='imagenet/test.json')
    parser.add_argument("--root_file_path", type=str, default='imagenet/')
    parser.add_argument("--other_file_path", type=str, default='imagenet-s/')
    parser.add_argument("--batch_size", type=int, default=16)

    # model settings
    parser.add_argument('--n_banks', type=int, default=4)
    parser.add_argument('--bank_dim', type=int, default=1024)
    parser.add_argument('--n_prompts', type=int, default=4)
    parser.add_argument('--prompt_dim', type=int, default=1024)

    args = parser.parse_args()
    return args


def S2IRetrieval(args, model, ori_images, pair_images):

    ori_feat = model(ori_images, dtype='image')
    ske_feat = model(pair_images, dtype='image')  

    prob = torch.softmax(ske_feat.view(args.batch_size, -1) @ ori_feat.view(args.batch_size, -1).permute(1, 0), dim=-1)

    return prob


def T2IRetrieval(args, model, ori_images, text_caption):
    ori_feat = model(ori_images, dtype='image')
    ske_feat = model(text_caption, dtype='text')

    prob = torch.softmax(ske_feat @ ori_feat.T, dim=-1)

    return prob


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    
    model = Uni_Retrieval(args)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    model.to(args.device)

    test_dataset = I2TTestDataset(args.root_file_path, args.root_json_path, model.pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)

    r1 = []
    
    if args.type == 'text2image':
        for data in enumerate(tqdm(test_loader)):
            caption = model.tokenizer(data[1][1]).to(args.device, non_blocking=True)
            image = data[1][0].to(args.device, non_blocking=True)

            prob = T2IRetrieval(args, model, image, caption)

            r1.append(getI2TR1Accuary(prob))

    else:
        for data in enumerate(tqdm(test_loader)):
            origin_image = data[1][0].to(args.device, non_blocking=True)
            retrival_image = data[1][1].to(args.device, non_blocking=True)

            prob = S2IRetrieval(args, model, origin_image, retrival_image)

            r1.append(getI2TR1Accuary(prob))

    resr1 = sum(r1)/len(r1)
    print(resr1)