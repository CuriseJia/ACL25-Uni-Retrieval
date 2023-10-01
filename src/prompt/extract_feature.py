import open_clip
import torch.nn as nn
import torch
import numpy as np
from PIL import Image


model, _, pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device='cuda:5',
        )

model.eval()
model.to('cuda:5')

# for name, parm in model.named_parameters():
#     if not name.startswith('transformer') or name.startswith('visual'):
#         print(name)

emb = nn.Sequential(*list(model.named_parameters())[:5])

print(emb)