import torch
import torchvision
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)