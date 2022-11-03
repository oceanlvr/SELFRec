import scipy.sparse as sp
import torch

# print(sp.diags([1,2,3,5]).toarray())

a = torch.randn(3,4)
print('a', a)
b = torch.stack([a,a], dim=0)
print(b, b.shape)
print('=====')
c = torch.stack([a,a], dim=1)
print(c, c.shape)
