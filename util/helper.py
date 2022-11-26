import numpy as np
import torch
import random
from functools import reduce
import os.path as osp

def composePath(*pathArr):
    return reduce((lambda prePath, cur: osp.join(prePath, cur)), pathArr, '')

def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

