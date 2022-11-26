import numpy as np
import torch
import random
from functools import reduce
import os.path as osp
import argparse

def composePath(*pathArr):
    return reduce((lambda prePath, cur: osp.join(prePath, cur)), pathArr, '')

def fix_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = int(value)

def mergeDict(lhs, rhs):
  assert type(lhs) is dict and type(rhs) is dict
  res = lhs
  for key, _ in rhs.items():
    if key is 'model_config':
      res['model_config'] = mergeDict(lhs['model_config'], rhs['model_config'])
    elif rhs[key] is None:
      continue
    else:
      res[key] = rhs[key]
  return res

def splitNested(dc):
  assert type(dc) is dict
  res = {}
  for key, value in dc.items():
    if key.find('.') != -1:
      arr = key.split('.')
      k = arr[0]
      v = arr[1]
      if res[k] and type(res[k]) is dict:
        res[k][v] = value
      else:
        res[k] = { v: value }
    else:
      res[key] = value
  return res
