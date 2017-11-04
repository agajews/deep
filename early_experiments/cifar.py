import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.models import vgg11
from datasets import cifar10, loader
from logger import struct, Logger
from train import train, acc_metric, nll_metric

H = struct(
    batch_size=128,
    val_batch_size=100,
    epochs=10,
    lr=0.01,
    momentum=0.5,
)
S = struct(epoch=1, bn=1)

log = Logger('cifar10_standard', H, S, overwrite=True, metric_show_freq=100)

tn_loader, val_loader = loader(cifar10(), H.batch_size, H.val_batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# model = vgg11()
model = Net()
model.cuda()

optimizer = optim.RMSprop(model.parameters(), lr=H.lr, momentum=H.momentum)

print('Beginning training')
train(model, optimizer, tn_loader, val_loader, H, S, log,
      struct(
          val_acc=acc_metric(model, val_loader),
          val_loss=nll_metric(model, val_loader)))
