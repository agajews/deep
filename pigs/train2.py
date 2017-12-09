import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import torchsample
import time
import shutil

num_pigs = 30

print_freq = 200

batch_size = 32
epochs = 100

lr = 0.01
momentum = 0.9
weight_decay = 1e-4

train_tf = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
    transforms.ToTensor(),
    torchsample.transforms.RandomAffine(
        rotation_range=20,
        translation_range=(0.2, 0.2),
        zoom_range=(0.7, 1.3),
        shear_range=20),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('/mnt/data/pigs/imgs', train_tf)
val_data = datasets.ImageFolder('/mnt/data/pigs/val_imgs', val_tf)

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(
    val_data, batch_size=batch_size, shuffle=True, num_workers=2)

arch = 'resnet34-augment3'


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pretrained = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(pretrained.children())[:-1])
        # for layer in list(self.features.children())[:-2]:
        #     for param in layer.parameters():
        #         param.requires_grad = False
        # for sublayer in list(list(
        #         self.features.children())[-2].children())[:-1]:
        #     for param in sublayer.parameters():
        #         param.requires_grad = False
        self.classifier = nn.Linear(512, num_pigs)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_pigs),
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 4, 512),
        #     # nn.ReLU(True),
        #     # nn.Dropout(),
        #     # nn.Linear(512, 256),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_pigs),
        # )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if i > int(train_prop * len(train_loader)):
        #     break
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = Variable(input).cuda()
        target_var = Variable(target).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(
        top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state,
                    is_best,
                    filename='checkpoint-{}.pth.tar'.format(arch)):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best-{}.pth.tar'.format(arch))


model = Model().cuda()
optimizer = SGD(
    filter(lambda p: p.requires_grad, model.parameters()), lr, momentum,
    weight_decay)
criterion = nn.CrossEntropyLoss().cuda()

best_prec1 = 0
for epoch in range(epochs):
    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    prec1 = validate(val_loader, model, criterion)

    # remember best prec@1 and save checkpoint
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint({
        'epoch': epoch + 1,
        'arch': arch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer': optimizer.state_dict(),
    }, is_best)
