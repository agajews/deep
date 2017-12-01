import torch

x = torch.ones(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
print(x + y)

print(x.cuda() + y.cuda())
