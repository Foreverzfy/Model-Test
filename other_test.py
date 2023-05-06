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

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
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
                                          shuffle=True, num_workers=2)

test_dataset = torchvision.datasets.CIFAR100(root='/content/drive/MyDrive/cifar100/cifar100_data_test', train=False,
                                       download=True, transform=transform_test)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))



model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(device) #VGG series model, Resnet series model and so on
#model = torch.hub.load('PingoLH/Pytorch-HarDNet', 'hardnet68', pretrained=True).to(device) #68 85 68ds 39ds
#model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn131', pretrained=True).to(device) #68 92 98 107 131
model.eval()
#print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)



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
    train(train_dataloader, model, loss_fn, optimizer)    #train model
    startTime = timeit.default_timer() 
    evaluteTop1(test_dataloader, model)
    StartTime = timeit.default_timer()
    evaluteTop5(test_dataloader, model)
    stopTime = timeit.default_timer() 
    print('Top1 time: %5.1fs.'%(StartTime - startTime))
    print('Top5 time: %5.1fs.'%(stopTime - StartTime))
    #torch.save(model.state_dict(), 'vgg11.pth')
    #torch.save(model, 'vgg11.pth')
print("Done!")
