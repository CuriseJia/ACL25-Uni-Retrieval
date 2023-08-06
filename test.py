import argparse
import os
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.prompt.model import ShallowPromptTransformer
from src.prompt.data import OriginalLongDataset
from src.prompt.utils import init_distributed_mode, setup_seed, get_rank, get_world_size, is_main_process, save_loss

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # base args
    parser.add_argument('--config', default='configs/airproduct.yaml')
    parser.add_argument('--output_dir', default='output/')        
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--local_rank", type=int)

    # data args
    parser.add_argument("--dataset_path", type=str, default='/public/home/jiayanhao/imagenet/train/')
    parser.add_argument("--train_json_path", type=str, default='/public/home/jiayanhao/imagenet/200-original-long.json')
    parser.add_argument("--test_json_path", type=str, default='/public/home/jiayanhao/imagenet/200-original-long-val.json')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)

    # model args
    parser.add_argument('--prompt', type=str, default='ShallowPrompt')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    # optimizer args
    parser.add_argument('--clip_ln_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)

    args = parser.parse_args()
    return args



def train(args, model, device, dataloader, optimizer):
    model.train()

    best_loss = 10000000

    losses = []
    epoches = []

    for epoch in tqdm(range(args.epochs)):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)
        
        temp_loss = []

        for data in enumerate(tqdm(dataloader)):
            image = data[1][0].to(device, non_blocking=True)
            long_caption = model.tokenizer(data[1][1]).to(device, non_blocking=True)
            negative_image = data[1][2].to(device, non_blocking=True)

            image_feature = model(image, dtype='image')
            negative_feature = model(negative_image, dtype='image')
            text_feature = model(long_caption, dtype='text')

            loss = model.triplet_loss(image_feature, text_feature, negative_feature)

            temp_loss.append(loss.detach().cpu().numpy())

            print("loss: {:.6f}".format(loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        res = round(sum(temp_loss)/len(temp_loss), 6)
        print("epoch_{} loss is {}.".format(epoch, res))
        losses.append(res)
        epoches.append(epoch)

        if res<best_loss:
            best_loss = res
            save_obj = model.state_dict()
            torch.save(save_obj, os.path.join(args.output_dir, 'epoch_mar1_{}.pth'.format(epoch)))
        
        if best_loss < 0.0001:
            break
    
    return losses, epoches


def eval(args, model, dataloader):
    model.eval()

    for data in tqdm(enumerate(dataloader)):
        image = data[1][0].to(device, non_blocking=True)
        long_caption = model.tokenizer(data[1][1]).to(device, non_blocking=True)

        image_feature = model(image, dtype='image')
        text_feature = model(long_caption, dtype='text')

        image_feature /= F.normalize(image_feature, dim=-1)
        text_feature /= F.normalize(text_feature, dim=-1)

        prob = torch.softmax((100.0 * image_feature @ text_feature.T), dim=-1)

        print(prob)

     # def testing_step(self):
    #     test_batch = self.get_fake_batch()
    #     img_tensor, txt_tensor = batch[:2]
    #     image_features = self.forward(img_tensor, dtype='image')
    #     text_features = self.forward(txt_tensor, dtype='text')
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #     print(text_probs)


if __name__ == "__main__":
    # args = parse_args()
    # # init_distributed_mode(args)
    # setup_seed(args.seed)
    # device = torch.device(args.device)

    # model = ShallowPromptTransformer(args)
    # model = model.to(device)
    # model_without_ddp = model

    # train_dataset = OriginalLongDataset(args.train_json_path, model.pre_process_train, 'train')
    # test_dataset = OriginalLongDataset(args.val_json_path, model.pre_process_val, 'test')

    # optimizer = torch.optim.Adam([
    #         {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
    #         {'params': [model.img_prompt], 'lr': args.prompt_lr}])


    # train_loader = DataLoader(dataset=train_dataset, 
    #                         batch_size=args.batch_size,
    #                         num_workers=args.num_workers,
    #                         pin_memory=True,
    #                         prefetch_factor=16,
    #                         shuffle=False,
    #                         drop_last=True
    #                         )
    # test_loader = DataLoader(dataset=test_dataset, 
    #                         batch_size=args.batch_size,
    #                         num_workers=args.num_workers,
    #                         pin_memory=True,
    #                         prefetch_factor=16,
    #                         shuffle=False,
    #                         drop_last=True
    #                         )


    # loss, epochs = train(args, model, device, train_loader, optimizer)
    loss= []
    epochs=[]
    save_loss(loss, epochs)
    # eval(args, model, test_loader)
    