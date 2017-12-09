import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models, transforms
from torchvision import datasets
# import torchsample
import numpy as np
import csv
import os
from PIL import Image

num_pigs = 30
batch_size = 64

fnm = "checkpoint-resnet50-augment2-lambda5e-1.pth.tar"

# train_tf = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.Resize((224, 224)),
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
#     transforms.ToTensor(),
#     torchsample.transforms.RandomAffine(
#         rotation_range=20,
#         translation_range=(0.2, 0.2),
#         zoom_range=(0.7, 1.0),
#         shear_range=20),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

test_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# test_data = datasets.ImageFolder('/home/alex/Downloads/test_A', test_tf)
val_data = datasets.ImageFolder('/mnt/data/pigs/val_imgs', test_tf)

# test_loader = DataLoader(
#     test_data, batch_size=batch_size, shuffle=False, num_workers=2)
val_loader = DataLoader(
    val_data, batch_size=batch_size, shuffle=False, num_workers=2)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        pretrained = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(pretrained.children())[:-1])
        self.classifier = nn.Linear(512 * 4, num_pigs)
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, num_pigs),
        # )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


model = Model().cuda()
checkpoint = torch.load(fnm)
model.load_state_dict(checkpoint['state_dict'])

# assert len(test_data) == len(test_data.classes)
# probs = np.zeros((len(test_data), num_pigs))
# print(probs.shape)


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


def validate(val_loader, model, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = Variable(input, volatile=True).cuda()
        target_var = Variable(target, volatile=True).cuda()

        # compute output
        output = F.log_softmax(model(input_var))
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))
        print(loss.data[0], losses.avg)

        # measure elapsed time

        # if i % print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               i,
        #               len(val_loader),
        #               batch_time=batch_time,
        #               loss=losses,
        #               top1=top1,
        #               top5=top5))

    print('Loss {losses.avg:.3f}'.format(losses=losses))


base_fnm = '/home/alex/Downloads/test_B'
num_imgs = len(os.listdir(base_fnm))

probs = np.zeros((num_imgs, num_pigs))
all_img_nums = np.zeros((num_imgs, ))


def test():
    model.eval()
    # for i, (input, target) in enumerate(test_loader):
    for i, filename in enumerate(os.listdir(base_fnm)):
        # target = target.cuda(async=True)
        img_num = int(os.path.splitext(filename)[0])
        all_img_nums[i] = img_num
        if i % 100 == 0:
            print(img_num)
        img = Image.open(os.path.join(base_fnm, filename))
        img = test_tf(img)
        input_var = Variable(img, volatile=True).cuda()

        output = F.softmax(model(torch.unsqueeze(input_var, 0)))
        # assert target[0] == i * batch_size
        # assert target[-1] == (i + 1) * batch_size - 1
        probs[i] = output.cpu().data.numpy()
        # print(output[0].cpu().data.numpy())


criterion = nn.NLLLoss().cuda()

if __name__ == '__main__':
    # validate(val_loader, model, criterion)
    test()
    img_nums = np.array(
        [[img_num] * num_pigs
         for img_num in all_img_nums], dtype='int32').flatten()
    pigs = [int(c) + 1 for c in val_data.classes] * num_imgs
    preds = probs.flatten()

    with open('test_preds6.csv', 'w') as f:
        writer = csv.writer(f)
        for row in zip(img_nums, pigs, preds):
            writer.writerow(row)
