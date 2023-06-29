import argparse

import datasets
from torch.utils.data import DataLoader, DistributedSampler
import os.path as osp
import torch


# img_path = osp.join('ln_data/ms-cxr/images', 'file/p10/s1231241241/2124234123.jpg')
# print(img_path)

a = torch.range(1, 24).resize(2, 3, 4)
b = torch.range(13, 24).resize(3, 4)
print(a)
print(a.view(2, 12))
# print(torch.cat([a, b], dim=0).shape)
# print(torch.stack([a, b], dim=0).shape)
