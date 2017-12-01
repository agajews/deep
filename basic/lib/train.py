import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from logger import struct, Logger


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


def max_metametric(name):
    return lambda ms: max(batch[name] for batch in ms if name in batch)


def argmax_metametric(name):
    return lambda ms: int(np.argmax(batch[name] for batch in ms if name in batch))


def min_metametric(name):
    return lambda ms: min(batch[name] for batch in ms if name in batch)


def argmin_metametric(name):
    return lambda ms: int(np.argmin(batch[name] for batch in ms if name in batch))


def train(model,
          optimizer,
          tn_loader,
          val_loader,
          H,
          S,
          log,
          val_metrics=None,
          loss_fn=F.nll_loss,
          param_log_freq=1):

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
        if param_log_freq != 0 and epoch % param_log_freq == 0:
            log.log_params()

    def val():
        model.eval()
        metrics = struct()
        for name, metric in val_metrics.items():
            metrics[name] = metric()
        log.log_metrics(metrics, show=True, desc='Val')

    for epoch in range(S.epoch, H.epochs + 1):
        if param_log_freq != 0 and epoch == 1:
            log.log_params()
        S.epoch = epoch
        log.flush()
        train()
        val()
    if S.epoch != H.epochs + 1:
        S.epoch = H.epochs + 1
        log.flush()


def cycle_hyperparams(name,
                      hyper_name,
                      H_proto,
                      hyper_range,
                      model_fn,
                      optim_fn,
                      loader_fn,
                      metrics_fn,
                      meta_metrics,
                      overwrite=False,
                      load=False,
                      loss_fn=F.nll_loss):

    S_proto = struct(epoch=0, bn=0)

    log = Logger(
        name,
        H_proto,
        S_proto,
        overwrite=overwrite,
        load=load,
        metric_show_freq=1)

    for epoch in range(S_proto.epoch, len(hyper_range)):
        S_proto.epoch = epoch
        log.flush()
        hyper = hyper_range[epoch]
        H = H_proto.copy()
        H[hyper_name] = hyper
        S = struct(epoch=1, bn=1)
        inner_log = Logger(
            '{}{}'.format(hyper_name, hyper),
            H,
            S,
            # overwrite=True,
            load=True,
            metric_show_freq=0,
            parent=log)

        model = model_fn(H)
        model.cuda()

        optimizer = optim_fn(H, model.parameters())

        tn_loader, val_loader = loader_fn(H)
        train(model, optimizer, tn_loader, val_loader, H, S, inner_log,
              metrics_fn(model, tn_loader, val_loader), loss_fn)

        metrics = {}
        for name, metric in meta_metrics.items():
            metrics[name] = metric(inner_log.metrics())

        log.log_metrics(struct(**metrics), '{}{}'.format(hyper_name, hyper))

    if S_proto.epoch != len(hyper_range) + 1:
        S_proto.epoch = len(hyper_range) + 1
        log.flush()
