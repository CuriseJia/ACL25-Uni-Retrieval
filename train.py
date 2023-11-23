import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models import Uni_Retrieval
from src.models import I2ITestDataset, I2ITrainDataset, I2MTestDataset, I2MTrainDataset, I2TTestDataset, I2TTrainDataset
from src.utils import setup_seed, save_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Uni-Retrieval Train.')

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

    # optimizer settings
    parser.add_argument('--clip_ln_lr', type=float, default=1e-5)
    parser.add_argument('--prompt_lr', type=float, default=1e-5)

    args = parser.parse_args()
    return args


def train(args, model, dataloader, optimizer):
    model.train()

    best_loss = 10000000

    losses = []
    epoches = []
    count = 0

    if args.type == 'text2image':
        for epoch in range(args.epochs):
            temp_loss = []

            for data in enumerate(tqdm(dataloader)): 
                caption = model.tokenizer(data[1][0]).to(args.device, non_blocking=True)
                image = data[1][1].to(args.device, non_blocking=True)
                negative_image = data[1][2].to(args.device, non_blocking=True)

                text_feature = model(caption, dtype='text')
                image_feature = model(image, dtype='image')
                negative_feature = model(negative_image, dtype='image')

                loss = model.get_loss(image_feature, text_feature, negative_feature, optimizer)

                temp_loss.append(loss)

                print("loss: {:.6f}".format(loss))

            if len(temp_loss)!=0:
                res = round(sum(temp_loss)/len(temp_loss), 6)
                print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if res<best_loss:
                best_loss = res
                save_obj = model.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, '{}_epoch{}.pth'.format(args.type, epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break
    
    else:   # style2image retrival
        for epoch in range(args.epochs):
            temp_loss = []
            
            for data in enumerate(tqdm(dataloader)):
                original_image = data[1][0].to(args.device, non_blocking=True)
                retrival_image = data[1][1].to(args.device, non_blocking=True)
                negative_image = data[1][2].to(args.device, non_blocking=True)

                original_feature = model(original_image, dtype='image')
                retrival_feature = model(retrival_image, dtype='image')
                negative_feature = model(negative_image, dtype='image')

                loss = model.get_loss(original_feature, retrival_feature, negative_feature, optimizer)

                temp_loss.append(loss)

                print("loss: {:.6f}".format(loss))
            
            if len(temp_loss)!=0:
                res = round(sum(temp_loss)/len(temp_loss), 6)
                print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if res<best_loss:
                best_loss = res
                save_obj = model.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, '{}_epoch{}.pth'.format(args.type, epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break
    
    return losses, epoches


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)

    model = Uni_Retrieval(args)
        
    model = model.to(args.device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    optimizer = torch.optim.Adam([
            {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
            {'params': [model.prompt_lr], 'lr': args.prompt_lr}])

    train_dataset = I2TTrainDataset(args.root_file_path, args.root_json_path, model.pre_process_train, model.tokenizer)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=4, shuffle=False, drop_last=True)

    loss, epochs = train(args, model, train_loader, optimizer)

    save_loss(loss, epochs, args.out_path)