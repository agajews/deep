import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import mnist, loader
from logger import struct, Logger
from train import train, acc_metric, nll_metric
import numpy as np

H_proto = struct(
    batch_size=64,
    val_batch_size=1000,
    epochs=10,
    lr=0.01,
    caps=[10, 20, 50, 70, 100, 200, 500],
    momentum=0.5,
)
S_proto = struct(epoch=0, bn=0)

log = Logger('mnist_capacity', H_proto, S_proto, load=True, metric_show_freq=1)

tn_loader, val_loader = loader(mnist(), H_proto.batch_size,
                               H_proto.val_batch_size)

for epoch in range(S_proto.epoch, len(H_proto.caps)):
    S_proto.epoch = epoch
    log.flush()
    cap = H_proto.caps[epoch]
    H = H_proto.copy()
    H.cap = cap
    S = struct(epoch=1, bn=1)
    inner_log = Logger(
        'cap{}'.format(cap),
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
            self.fc1 = nn.Linear(320, cap)
            self.fc2 = nn.Linear(cap, 10)

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

    train(model, optimizer, tn_loader, val_loader, H, S, inner_log,
          struct(
              val_acc=acc_metric(model, val_loader),
              val_loss=nll_metric(model, val_loader),
              tn_loss=nll_metric(model, tn_loader),
              tn_acc=acc_metric(model, tn_loader)))

    val_accs = [
        batch.val_acc for batch in inner_log.metrics() if 'val_acc' in batch
    ]
    val_losses = [
        batch.val_loss for batch in inner_log.metrics() if 'val_loss' in batch
    ]
    tn_accs = [
        batch.tn_acc for batch in inner_log.metrics() if 'tn_acc' in batch
    ]
    tn_losses = [
        batch.tn_loss for batch in inner_log.metrics() if 'tn_loss' in batch
    ]

    best_acc = max(val_accs)
    best_acc_idx = int(np.argmax(val_accs))

    best_loss = min(val_losses)
    best_loss_idx = int(np.argmin(val_losses))

    best_tn_acc = max(tn_accs)
    best_tn_acc_idx = int(np.argmax(tn_accs))

    best_tn_loss = min(tn_losses)
    best_tn_loss_idx = int(np.argmin(tn_losses))

    log.log_metrics(
        struct(
            best_acc=best_acc,
            best_loss=best_loss,
            best_acc_idx=best_acc_idx,
            best_loss_idx=best_loss_idx,
            best_tn_acc=best_tn_acc,
            best_tn_loss=best_tn_loss,
            best_tn_acc_idx=best_tn_acc_idx,
            best_tn_loss_idx=best_tn_loss_idx,
            cap=cap), 'Cap {}'.format(cap))

if S_proto.epoch != len(H_proto.caps) + 1:
    S_proto.epoch = len(H_proto.caps) + 1
    log.flush()
