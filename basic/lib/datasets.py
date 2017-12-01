from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from settings import data_dir
import os


def mnist(transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    mnist_dir = os.path.join(data_dir, 'mnist')
    tn_data = datasets.MNIST(
        mnist_dir, train=True, download=True, transform=transform)
    ts_data = datasets.MNIST(
        mnist_dir, train=False, download=True, transform=transform)

    return tn_data, ts_data


def cifar10(transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    cifar10_dir = os.path.join(data_dir, 'cifar10')
    tn_data = datasets.CIFAR10(
        cifar10_dir, train=True, download=True, transform=transform)
    ts_data = datasets.CIFAR10(
        cifar10_dir, train=False, download=True, transform=transform)

    return tn_data, ts_data


def loader(data, batch_size, ts_batch_size):
    tn_data, ts_data = data
    tn_loader = DataLoader(tn_data, batch_size=batch_size, shuffle=True)
    ts_loader = DataLoader(ts_data, batch_size=ts_batch_size, shuffle=True)

    return tn_loader, ts_loader
