from models import ResNet18
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(root='./data', train=False, 
                               download=True, transform=transform)
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                               shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

def main():

    batch_size = 128
    epochs = 30
    learning_rate = 1E-4
    train_loader, test_loader, classes = dataset(batch_size)

    model = ResNet18()

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

