import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datasets import mnist, loader
from logger import struct
from train import (cycle_hyperparams, acc_metric, nll_metric, max_metametric,
                   argmax_metametric, min_metametric, argmin_metametric)

H_proto = struct(
    batch_size=64,
    val_batch_size=1000,
    epochs=10,
    lrs=[
        0.001, 0.003, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.3, 0.5,
        1
    ],
    momentum=0.5)


def model(H):
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

    return Net()


def optimizer(H, params):
    return optim.SGD(params, lr=H.lr, momentum=H.momentum)


loader_fn = lambda H: loader(mnist(), H.batch_size, H.val_batch_size)

metrics = lambda model, tn_loader, val_loader: struct(
    val_acc=acc_metric(model, val_loader),
    val_loss=nll_metric(model, val_loader),
    tn_loss=nll_metric(model, tn_loader),
    tn_acc=acc_metric(model, tn_loader))

meta_metrics = struct(
    best_acc=max_metametric('val_acc'),
    best_acc_idx=argmax_metametric('val_acc'),
    best_loss=min_metametric('val_loss'),
    best_loss_idx=argmin_metametric('val_loss'),
    best_tn_acc=max_metametric('tn_acc'),
    best_tn_acc_idx=argmax_metametric('tn_acc'),
    best_tn_loss=min_metametric('tn_loss'),
    best_tn_loss_idx=argmin_metametric('tn_loss'))

cycle_hyperparams(
    'mnist_lr2_clean',
    'lr',
    H_proto,
    H_proto.lrs,
    model,
    optimizer,
    loader_fn,
    metrics,
    meta_metrics,
    load=True)
