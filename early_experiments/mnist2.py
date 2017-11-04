import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from datasets import mnist, loader

batch_size = 64
ts_batch_size = 1000
epochs = 2
lr = 0.01
momentum = 0.5
seed = 1
log_freq = 100

tn_loader, ts_loader = loader(mnist(), batch_size, ts_batch_size)


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

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
print(list(model.named_parameters()))


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tn_loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(tn_loader.dataset), 100. * batch_idx / len(tn_loader),
                loss.data[0]))


def test():
    model.eval()
    ts_loss = 0
    correct = 0
    N = len(ts_loader.dataset)
    for data, target in ts_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        ts_loss += F.nll_loss(
            output, target, size_average=False).data[0]  # sum up batch loss
        pred = output.data.max(
            dim=1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    ts_loss /= N
    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            ts_loss, correct, N, 100. * correct / N))


for epoch in range(1, epochs + 1):
    train(epoch)
    test()
