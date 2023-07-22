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
                data, self.img_prompt.expand(data.shape[0], -1, -1)
            )
        elif dtype == 'text':
            feat = self.openclip.encode_text(data)
        return feat

    # def training_step(self):
    #     train_batch = self.get_fake_batch()
    #     img_tensor, txt_tensor, neg_feat = train_batch[:3]
    #     img_feat = self.forward(img_tensor, dtype='image')
    #     txt_feat = self.forward(txt_tensor, dtype='text')
    #     loss = self.triplet_loss(img_feat, txt_feat, neg_feat)
    #     print(loss)

    # def testing_step(self):
    #     test_batch = self.get_fake_batch()
    #     img_tensor, txt_tensor = batch[:2]
    #     image_features = self.forward(img_tensor, dtype='image')
    #     text_features = self.forward(txt_tensor, dtype='text')
    #     image_features /= image_features.norm(dim=-1, keepdim=True)
    #     text_features /= text_features.norm(dim=-1, keepdim=True)
    #     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    #     print(text_probs)
    
    # def get_fake_batch(self):
    #     image_name = "dog.png"
    #     text_list = ["a diagram", "a dog", "a cat"]
    #     image = self.preprocess(Image.open(image_name)).unsqueeze(0)
    #     text = self.tokenizer(text_list)
    #     neg_feat = torch.zeros(1, 512, dtype=torch.float32)
    #     return [image, text, neg_feat]