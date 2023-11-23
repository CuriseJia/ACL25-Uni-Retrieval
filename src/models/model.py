import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import VGG


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


def prompt_bank_select(input, key, value):
    input = input.view(input.shape[0], -1)
    input_temp = input / torch.norm(input, dim=-1, keepdim=True)
    key_temp = key / torch.norm(key, dim=-1, keepdim=True)
    sim = torch.mm(input_temp, key_temp.T)
    sim_prob = F.softmax(sim, dim=1)
    feature = torch.mm(sim_prob, value)

    return feature


class Uni_Retrieval(nn.Module):
    def __init__(self, model_args):
        super(Uni_Retrieval, self).__init__()
        self.args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion2b_s32b_b82k')
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        self.visual = self.openclip.visual
        # Prompt Token
        self.key = nn.Parameter(torch.randn(
            self.args.n_banks, self.args.bank_dim))
        self.img_prompt = nn.Parameter(torch.randn(
            self.args.n_prompts, self.args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.encoder = VGG
        self.encoder.apply(freeze_model)
        self.bank_patch = nn.Conv2d(128, 256, 16, 16)
        self.bank_pool = nn.Linear(256, 4)
        self.bank_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, self.args.prompt_dim))
        self.style_patch = nn.Conv2d(256, 256, 16, 16)
        self.style_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, self.args.gram_prompt_dim))


    def _get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',  
                    '5': 'conv2_1',  
                    '10': 'conv3_1', 
                    '19': 'conv4_1', 
                    '21': 'conv4_2', 
                    '28': 'conv5_1',
                    '31': 'conv5_2'}  
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)   
            if name in layers:
                features[layers[name]] = x
    
        return features


    def _get_prompt(self, input):
        feature = self.key.expand(self.args.batch_size, -1, -1)
        style_feature = self.bank_patch(feature)
        n, c, h, w = style_feature.shape    # (b, 256, 7, 7)
        style_feature = style_feature.view(n, c, -1)  # (b*256, 49)
        style_feature = torch.bmm(style_feature, style_feature.transpose(1, 2))
        
        x = self._get_features(input, self.encoder)
        embed = self.bank_patch(x['conv3_1'])
        n, c, h, w = embed.shape
        x = embed.view(n, c, -1)  # (b*256, 49)
        x = torch.bmm(x, x.transpose(1, 2))
        feature = prompt_bank_select(x, style_feature.view(self.args.img_prompt, -1))       # (b, 65536)
        feature = self.style_patch(feature.view(self.args.batch_size, 256, 16, 16)).view(self.args.batch_size, 256)
        feature = self.style_linear(feature).unsqueeze(1).repeat(1, self.args.img_prompt, 1)

        return feature


    def _visual_forward(self, x):
        prompt = self._get_prompt(x)

        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        x = torch.cat([x[:, 0, :].unsqueeze(1), prompt, x[:, 1:, :]], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled, tokens = self.visual._global_pool(x)
        pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        
        return pooled
    

    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            feat = self._visual_forward(data)

        elif dtype == 'text':
            feat = self.openclip.encode_text(data)

        return feat
    

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()
    