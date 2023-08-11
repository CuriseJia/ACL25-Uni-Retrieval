import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.prompt.model import ShallowPromptTransformer
from src.prompt.data import Image2ImageDataset, Image2TextDataset, DataPrefetcher
from src.prompt.utils import init_distributed_mode, setup_seed, get_rank, get_world_size, is_main_process, save_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # base args
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--out_path', default='origin-sketch-loss.jpg')
    parser.add_argument('--resume', default='/public/home/jiayanhao/SMR/output/epoch_i2t_29.pth', type=str, help='load checkpoints from given path')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument("--local_rank", type=int)

    # data settings
    parser.add_argument("--type", type=str, default='image2image', help='choose train image2text or image2image.')
    parser.add_argument("--train_ori_dataset_path", type=str, default='/public/home/jiayanhao/imagenet/train/')
    parser.add_argument("--train_art_dataset_path", type=str, default='/public/home/jiayanhao/imagenet/imagenet-sketch/')
    parser.add_argument("--test_ori_dataset_path", type=str, default='/public/home/jiayanhao/imagenet/val/')
    parser.add_argument("--test_art_dataset_path", type=str, default='/public/home/jiayanhao/imagenet/imagenet-sketch/')
    parser.add_argument("--train_json_path", type=str, default='/public/home/jiayanhao/imagenet/200-original-sketch-0.json')
    parser.add_argument("--test_json_path", type=str, default='/public/home/jiayanhao/imagenet/200-original-long-val.json')
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--test_batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)

    # model args
    parser.add_argument('--prompt', type=str, default='ShallowPrompt')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    # optimizer args
    parser.add_argument('--clip_ln_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)

    args = parser.parse_args()
    return args


def train(args, model, dataloader, datafetcher, optimizer):
    model.train()
    model_without_ddp = model.module

    best_loss = 10000000
    losses = []
    epoches = []
    count = 0

    if args.type == 'image2text':
        for epoch in tqdm(range(args.epochs)):
            if args.distributed:
                dataprefetcer.sampler.set_epoch(epoch)

            temp_loss = []
            data = dataprefetcer.next()
            # for data in enumerate(tqdm(dataprefetcer)):
            while data is not None:
                data = dataprefetcer.next()

                

                image_feature = model(data[0], dtype='image')
                text_feature = model(model.module.tokenizer(data[1]), dtype='text')
                negative_feature = model(data[2], dtype='image')

                loss = model.module.get_loss(image_feature, text_feature, negative_feature)
                temp_loss.append(loss.detach().cpu().numpy())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            res = round(sum(temp_loss)/len(temp_loss), 6)
            print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if is_main_process():
                if res<best_loss:
                    best_loss = res
                    save_obj = model.state_dict()
                    torch.save(save_obj, os.path.join(args.output_dir, 'epoch_i2s_{}.pth'.format(epoch)))
                    count = 0
                else:
                    count +=1
                
                if best_loss < 0.0001 or count >= 5:
                    break

    else:
        for epoch in tqdm(range(args.epochs)):
            if args.distributed:
                dataloader.sampler.set_epoch(epoch)
            temp_loss = []
            
            data = datafetcher.next()
            # for data in enumerate(tqdm(dataloader)):
            while data is not None:
                original_feature = model(data[0], dtype='image')
                retrival_feature = model(data[1], dtype='image')
                negative_feature = model(data[2], dtype='image')

                # original_image = data[1][0].to(device, non_blocking=True)
                # retrival_image = data[1][1].to(device, non_blocking=True)
                # negative_image = data[1][2].to(device, non_blocking=True)

                # original_feature = model(original_image, dtype='image')
                # retrival_feature = model(retrival_image, dtype='image')
                # negative_feature = model(negative_image, dtype='image')

                loss = model_without_ddp.get_loss(original_feature, retrival_feature, negative_feature, optimizer)

                temp_loss.append(loss)

                print("loss: {:.6f}".format(loss))

                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                data = datafetcher.next()
            
            res = round(sum(temp_loss)/len(temp_loss), 6)
            print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if res<best_loss:
                best_loss = res
                save_obj = model.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, 'epoch_i2s_{}.pth'.format(epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break




if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    setup_seed(args.seed)
    device = torch.device(args.device)

    model = ShallowPromptTransformer(args)
    model = model.to(device)
    # model.load_state_dict(torch.load(args.resume))
    # model_without_ddp = model

    train_dataset = Image2ImageDataset(args.train_ori_dataset_path, args.train_art_dataset_path, args.train_json_path, model.pre_process_train, 'train')


    optimizer = torch.optim.Adam([
            {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
            {'params': [model.img_prompt], 'lr': args.prompt_lr}])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        # model_without_ddp = model.module 
        num_tasks = get_world_size()
        global_rank = get_rank()          
        sampler = DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.train_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            sampler=sampler,
                            shuffle=False,
                            drop_last=True
                            )
    train_prefetcher = DataPrefetcher(train_loader)

    loss, epochs = train(args, model, train_loader, train_prefetcher, optimizer)

    save_loss(loss, epochs, args.out_path)