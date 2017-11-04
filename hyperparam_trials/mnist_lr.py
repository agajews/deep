import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import mnist, loader
from logger import struct, Logger
from train import train

H_proto = struct(
    batch_size=64,
    val_batch_size=1000,
    epochs=10,
    lrs=[
        0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.5,
        1
    ],
    # lr=0.01,
    momentum=0.5,
)
S_proto = struct(epoch=0, bn=0)

log = Logger('mnist_lr2', H_proto, S_proto, load=True, metric_show_freq=1)
S_proto.epoch = 0

tn_loader, val_loader = loader(mnist(), H_proto.batch_size,
                               H_proto.val_batch_size)

for epoch in range(S_proto.epoch, len(H_proto.lrs)):
    S_proto.epoch = epoch
    log.flush()
    lr = H_proto.lrs[epoch]
    H = H_proto.copy()
    H.lr = lr
    S = struct(epoch=1, bn=1)
    inner_log = Logger(
        'lr{}'.format(lr),
        H,
        S,
        # overwrite=True,
        load=True,
        metric_show_freq=0,
        parent=log)

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

    optimizer = optim.SGD(model.parameters(), lr=H.lr, momentum=H.momentum)

    train(model, optimizer, tn_loader, val_loader, H, S, inner_log)

    best_acc = max(
        batch.val_acc for batch in inner_log.metrics() if 'val_acc' in batch)
    best_loss = min(
        batch.val_loss for batch in inner_log.metrics() if 'val_loss' in batch)
    log.log_metrics(
        struct(best_acc=best_acc, best_loss=best_loss, lr=lr),
        'LR {}'.format(lr))
