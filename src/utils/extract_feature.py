import open_clip
import torch.nn as nn
import numpy as np
from PIL import Image
import os


path = ''
temp = []

model, _, valp = open_clip.create_model_and_transforms(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device='cuda:0')

model.eval()
model.to('cuda:0')

filelist = os.listdir(path)

for i in range(1000):
    index = np.random.randint(0, len(filelist))
    img = valp(Image.open(os.path.join(path, filelist[index]))).unsqueeze(0).to('cuda:0')
    embedding = model.encode_image(img)
    temp.append(embedding)

result = sum(temp)/len(temp)