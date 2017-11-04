import torch.nn.functional as F
from torch.autograd import Variable
from logger import struct


def train(model, optimizer, tn_loader, val_loader, H, S, log):
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
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            log.log_metrics(struct(loss=loss.data[0]))
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
            struct(val_loss=val_loss, val_acc=val_acc), show=True, desc='Val')

    for epoch in range(S.epoch, H.epochs + 1):
        if epoch == 1:
            log.log_params()
        S.epoch = epoch
        log.flush()
        train()
        val()
    S.epoch += 1
    log.flush()
