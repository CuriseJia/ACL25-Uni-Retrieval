import torch
import torch.nn as nn
import numpy as np


def upsample(img):
    m = nn.UpsamplingNearest2d(img.shape, 16)
    return m(img)

def downsample(img):
    m = nn.MaxPool2d()