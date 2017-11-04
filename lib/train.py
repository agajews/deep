import torch.nn.functional as F
from torch.autograd import Variable
from logger import struct


def loss_metric(model, loader, loss_fn):
    def metric():
        loss = 0
        N = len(loader.dataset)
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss += loss_fn(output, target)
        loss /= N
        return loss

    return metric


def nll_metric(model, loader):
    return loss_metric(
        model, loader,
        lambda output, target: F.nll_loss(output, target, size_average=False).data[0]
    )


def acc_metric(model, loader):
    def metric():
        correct = 0
        N = len(loader.dataset)
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            pred = output.data.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        acc = 100. * correct / N
        return acc

    return metric


def train(model,
          optimizer,
          tn_loader,
          val_loader,
          H,
          S,
          log,
          val_metrics=None,
          loss_fn=F.nll_loss):
    if val_metrics is None:
        val_metrics = {}

    def params_reader():
        return struct(
            model_state=model.state_dict(), optim_state=optimizer.state_dict())

    def params_writer(params):
        model.load_state_dict(params.model_state)
        optimizer.load_state_dict(params.optim_state)

    log.set_params_reader(params_reader)
    log.set_params_writer(params_writer)

    def train():
        model.train()
        for bn, (data, target) in enumerate(tn_loader):
            S.bn = bn + 1
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            log.log_metrics(struct(loss=loss.data[0]))
        S.bn += 1
        log.log_params()

    def val():
        model.eval()
        metrics = struct()
        for name, metric in val_metrics.items():
            metrics[name] = metric()
        log.log_metrics(metrics, show=True, desc='Val')

    for epoch in range(S.epoch, H.epochs + 1):
        if epoch == 1:
            log.log_params()
        S.epoch = epoch
        log.flush()
        train()
        val()
    if S.epoch != H.epochs + 1:
        S.epoch = H.epochs + 1
        log.flush()
