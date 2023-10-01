import torch
import argparse
from src.prompt.model import OpenCLIP, ShallowPromptTransformer
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    parser.add_argument('--prompt', type=str, default='ShallowPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args

args = parse_args()
# 创建resnet50网络
# model = OpenCLIP()
model = ShallowPromptTransformer(args)
model = model.to('cuda:5')

# 创建输入网络的tensor
tensor = torch.rand(1, 3, 224, 224).to('cuda:5')

# 分析FLOPs
flops = FlopCountAnalysis(model, tensor)
print("FLOPs: ", flops.total())