from PIL import Image
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class ShallowPromptTransformer(nn.Module):
    def __init__(self, model_args, tgt_device='cpu'):
        super(ShallowPromptTransformer, self).__init__()
        self.model_args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device=tgt_device,
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.openclip.apply(freeze_all_but_bn)
        # Prompt Token
        self.img_prompt = nn.Parameter(torch.randn(
            self.model_args.n_prompts, self.model_args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=0.2)


    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            feat = self.openclip.encode_image(
                # torch.cat((data[:, :1, :], self.img_prompt.expand(data.shape[0], -1, -1), data[:, 1:,]), dim=1)
                # torch.cat((self.img_prompt.expand(data.shape[0], -1, -1).view(
                #     data.shape[0],data.shape[1],data.shape[2],data.shape[3]), data), dim=1)
                data + self.img_prompt.expand(data.shape[0], -1, -1).view(
                    data.shape[0],data.shape[1],data.shape[2],data.shape[3])
            )
        elif dtype == 'text':
            feat = self.openclip.encode_text(data)
        return feat
