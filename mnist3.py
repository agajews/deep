import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets import mnist, loader
from logger import struct, Logger

H = struct(
    batch_size=64,
    val_batch_size=1000,
    epochs=10,
    lr=0.01,
    momentum=0.5,
    seed=1,
)
S = struct(epoch=1, bn=1)
log_freq = 100,

tn_loader, val_loader = loader(mnist(), H.batch_size, H.val_batch_size)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
model.cuda()

log = Logger(
    'mnist_standard', H, S, params=model.named_parameters(), overwrite=True)

optimizer = optim.SGD(model.parameters(), lr=H.lr, momentum=H.momentum)


def train():
    model.train()
    for bn, (data, target) in enumerate(tn_loader):
        S.bn = bn + 1
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        log.log_metrics(dict(loss=loss.data[0]))
    S.bn += 1
    log.log_params()


def val():
    model.eval()
    val_loss = 0
    correct = 0
    N = len(val_loader.dataset)
    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(dim=1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    val_loss /= N
    val_acc = 100. * correct / N
    log.log_metrics(
        dict(val_loss=val_loss, val_acc=val_acc), show=True, desc='Val')


log.log_params()
for epoch in range(S.epoch, H.epochs + 1):
    S.epoch = epoch
    train()
    val()
