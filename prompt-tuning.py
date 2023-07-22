from PIL import Image
import argparse
import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prompt.model import ShallowPromptTransformer

shallow_prompt_dict = dict(
    n_prompts = 4,    # 随便编的,需要调整
    prompt_dim = 768,
    clip_ln_lr = 1e-5, # 随便编的,需要调整
    prompt_lr = 1e-3,  # 随便编的,需要调整
)
shallow_prompt_args = argparse.Namespace(**shallow_prompt_dict)


    
if __name__ == "__main__":

    s_prompt = ShallowPromptTransformer(shallow_prompt_args)
    s_prompt.training_step()
    s_prompt.testing_step()