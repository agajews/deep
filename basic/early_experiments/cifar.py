import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import cifar10, loader
from logger import struct, Logger
from train import train, acc_metric, nll_metric
from models import vgg

H = struct(
    batch_size=128,
    val_batch_size=100,
    epochs=10,
    lr=0.1,
    momentum=0.9,
)
S = struct(epoch=1, bn=1)

log = Logger('cifar10_standard', H, S, overwrite=True, metric_show_freq=100)

tn_loader, val_loader = loader(cifar10(), H.batch_size, H.val_batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = vgg('vgg11')
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = F.log_softmax(out)
        return out


model = Net()
# model = Net()
model.cuda()

optimizer = optim.SGD(model.parameters(), lr=H.lr, momentum=H.momentum)

print('Beginning training')
train(
    model,
    optimizer,
    tn_loader,
    val_loader,
    H,
    S,
    log,
    struct(
        val_acc=acc_metric(model, val_loader),
        val_loss=nll_metric(model, val_loader)),
    param_log_freq=3)
