import argparse

import datasets
from torch.utils.data import DataLoader, DistributedSampler
import os.path as osp


img_path = osp.join('ln_data/ms-cxr/images', 'file/p10/s1231241241/2124234123.jpg')
print(img_path)