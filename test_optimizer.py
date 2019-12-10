import torch
import torch
import torch.optim as optim
from optimizer import CGD
w1 = torch.randn(3, 3)
w2 = torch.randn(2, 2)
o = CGD([w1, w2])
print(o.param_groups)