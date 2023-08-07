import argparse
import os.path as osp
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from src.prompt.model import ShallowPromptTransformer
# from src.prompt.data import OriginalLongDataset
# from src.prompt.utils import init_distributed_mode, setup_seed, get_rank, get_world_size, is_main_process, save_loss

path = osp.join('/Users/jiayanhao/SMR/src/prompt/', 'data/')
print(path)