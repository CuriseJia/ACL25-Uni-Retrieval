import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.prompt.model import ShallowPromptTransformer
from src.prompt.data import OriginalLongDataset
from src.prompt.utils import init_distributed_mode, setup_seed, get_rank, get_world_size, is_main_process

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # base args
    parser.add_argument('--config', default='configs/airproduct.yaml')
    parser.add_argument('--output_dir', default='output/')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--local_rank", type=int)

    # data args
    parser.add_argument("--dataset_path", type=str, default='/public/home/jiayanhao/imagenet/train/')
    parser.add_argument("--json_path", type=str, default='/public/home/jiayanhao/imagenet/200-original-long.json')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)

    # model args
    parser.add_argument('--prompt', type=str, default='ShallowPrompt')
    parser.add_argument('--n_prompts', type=int, default=4)
    parser.add_argument('--prompt_dim', type=int, default=768)

    # optimizer args
    parser.add_argument('--clip_ln_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)

    args = parser.parse_args()
    return args


def train(args, model, device, dataloader, optimizer):
    model.train()

    best_loss = 10000000

    for epoch in range(0, args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        for (image, long_caption, negative_feature) in enumerate(tqdm(dataloader)):
            image = model.pre_process_train(image).unsqueeze(0).to(device, non_blocking=True)
            long_caption = model.tokenizer(long_caption).to(device, non_blocking=True)
            negative_feature = negative_feature.to(device, non_blocking=True)

            image_feature = model(image, dtype='image')
            text_feature = model(long_caption, dtype='text')

            loss = model.triplet_loss(image_feature, text_feature, negative_feature)

            print("loss: {:.6f}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if is_main_process():
            if loss<best_loss:
                save_obj = model_without_ddp.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, 'epoch_{}.path'.format(epoch)))








if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    setup_seed(args.seed)
    device = torch.device(args.device)

    model = ShallowPromptTransformer(args)
    model = model.to(device)
    model_without_ddp = model

    train_dataset = OriginalLongDataset(args.json_path, model.pre_process_train)

    optimizer = torch.optim.Adam([
            {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
            {'params': [model.img_prompt], 'lr': args.prompt_lr}])

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module 
        num_tasks = get_world_size()
        global_rank = get_rank()          
        sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            sampler=sampler,
                            shuffle=False,
                            drop_last=True
                            )

    

    train(args, model, device, train_loader, optimizer)