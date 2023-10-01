from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import numpy.random as random
import numpy as np
import json
import os.path as osp
from tqdm import tqdm
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def getI2TR1Accuary(prob):
    temp = prob.detach().cpu().numpy()
    temp = np.argsort(temp, axis=1)
    count = 0
    for i in range(prob.shape[0]):
        if temp[i][prob.shape[1]-1] == i:
            count+=1
    acc = count/prob.shape[0]
    return acc

text_list=[]
ori_image=[]
sketch_image=[]

device = "cuda:5" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained='imagebind_huge.pth')
model.eval()
model.to(device)
tensor = []
tensor.append(torch.rand(1, 3, 224, 224))
input2 = {
        # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(tensor, device),
    }


# 分析FLOPs
flops = FlopCountAnalysis(model, input2)
print("FLOPs: ", flops.total())

def params_count(model):
    """
    Compute the number of   parameters.
    Args:
        model (model): model to count   the number of parameters.
    """
    return np.sum([p.numel() for p in   model.parameters()]).item()

# print(params_count(model))

# pair = json.load(open('/public/home/jiayanhao/imagenet/200-original-long.json', 'r'))
# other = json.load(open('/public/home/jiayanhao/imagenet/200-original-sketch-0-10.json', 'r'))
# acc = []
# for i in tqdm(range(0, 10)):
#     for j in range(64):
#         index = random.randint(0, len(pair))
#         # text_list.append(pair[index]['caption'])
#         ori_image.append(osp.join('/public/home/jiayanhao/imagenet/imagenet-p/train/', pair[index]['image_path']))
#         sketch_image.append(osp.join('/public/home/jiayanhao/imagenet/imagenet-sketch/', other[index]['sketch_image']))
#     input1 = {
#         # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#         ModalityType.VISION: data.load_and_transform_vision_data(ori_image, device),
#     }
#     input2 = {
#         # ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#         ModalityType.VISION: data.load_and_transform_vision_data(sketch_image, device),
#     }

#     with torch.no_grad():
#         embeddings1 = model(input1)
#         embeddings2 = model(input2)

#     # prob = torch.softmax(embeddings1[ModalityType.VISION] @ embeddings1[ModalityType.TEXT].T, dim=-1)
#     prob = torch.softmax(embeddings1[ModalityType.VISION] @ embeddings2[ModalityType.VISION].T, dim=-1)

#     acc.append(getI2TR1Accuary(prob))

# r1 = sum(acc)/len(acc)
# print(r1)

# # print(
# #     "Vision x Text: ",
# #     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
# # )
# # print(
# #     "Audio x Text: ",
# #     torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
# # )
# # print(
# #     "Vision x Audio: ",
# #     torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
# # )