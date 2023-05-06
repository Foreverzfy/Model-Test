import sys, os
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets,  transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision.io import read_image
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import timeit
import pandas as pd
from torch.autograd import Variable
import numpy as np
from torch import nn,optim
import GPUtil
import json
import torch.backends.cudnn as cudnn

batch_size = 64  #1 2 4 8 16 32 64

# ----- Transform image scale
transform_train = transforms.Compose([
    #transforms.RandomHorizontalFilp(),
    transforms.RandomCrop(32),
    #transforms.RdandomResizedCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/cifar100/cifar100_data_train', train=True,
                                        download=True, transform=transform_train)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/cifar100/cifar100_data_test', train=False,
                                       download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# ----- AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=5, stride=2, padding=2),
            # kernel_size = 11,stride = 4,padding = 2
            nn.ReLU(),
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size = 3,stride = 2
            # IN : 96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size = 3,stride = 2
            # IN : 256*4*4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),  # kernel_size = 3,stride = 2
            # OUT : 384*2*2
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=100),
        )

    def forward(self, x):
        x = self.Conv(x)
        x = x.view(-1, 384 * 2 * 2)
        x = self.linear(x)
        return x


model = AlexNet().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.95, weight_decay=5e-4)


# ----- Train
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        correct = 0
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            correct /= len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}], Accuracy: {(100*correct):>0.1f}% ")

# ----- Test
def evaluteTop1(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    correct /= size
    print(f"Top1 Error: \n Accuracy: {(100*correct):>0.1f}% \n")
    Gpus = GPUtil.getGPUs()
    for gpu in Gpus:
        print('Total GPU',gpu.memoryTotal)
        print('GPU usage',gpu.memoryUsed)

def evaluteTop5(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            maxk = max((1,5))
            y_resize = y.view(-1,1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct += torch.eq(pred, y_resize).sum().float().item()
    correct /= size
    print(f"Top5 Error: \n Accuracy: {(100*correct):>0.1f}% \n")
    Gpus = GPUtil.getGPUs()
    for gpu in Gpus:
        print('Total GPU',gpu.memoryTotal)
        print('GPU usage',gpu.memoryUsed)

# ----- epochs
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    #train(train_dataloader, model, loss_fn, optimizer)     #--- train model
    startTime = timeit.default_timer() 
    evaluteTop1(test_dataloader, model)
    StartTime = timeit.default_timer()
    evaluteTop5(test_dataloader, model)
    stopTime = timeit.default_timer() 
    print('Top1 time: %5.1fs.'%(StartTime - startTime))
    print('Top5 time: %5.1fs.'%(stopTime - StartTime))
    #torch.save(model.state_dict(), 'Alexnet.pth')
    #torch.save(model, 'Alexnet.pth')
print("Done!")