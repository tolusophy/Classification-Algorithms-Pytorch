import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms
#from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
#from torchsummary import summary

from models import *
from models.ResNet import ResNet, PreAct_ResNet, ResNet20, ResNet32, ResNet44, ResNet56, ResNet110, ResNet164, ResNet1001, ResNet1202, preact_ResNet110, preact_ResNet164, preact_ResNet1001
from models.VGGNet import VGG_net
from models.AlexNet import AlexNet
from models.DenseNet import DenseNet, denseNet
from models.EfficientNet import EfficientNet
from models.InceptionNet import InceptionNet
from models.ResNext import ResNeXt, resneXt
from models.Wide_ResNet import Wide_ResNet, wide_ResNet

train_loader=torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100("data",train=True,download=True,
                                    transform=transforms.Compose([transforms.Resize((224,224)),
                                                                  transforms.RandomHorizontalFlip(),
                                                                  transforms.ToTensor(),])),batch_size=128,shuffle=True)
test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR100("data",train=False,download=True,
                                    transform=transforms.Compose([transforms.Resize((224,224)),
                                                                  transforms.ToTensor(),])),batch_size=128,shuffle=False)
print(len(train_loader))
print(len(test_loader))

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

#define your model here
model = VGG_net('VGG11',3,100)
model = model.to(device)

criterion=torch.nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=4, gamma=0.1)

for epoch in range(15):
    
    train_loss = 0
    train_accuracy = 0
    train_samples = 0
    for batch_idx, (data, targets) in (enumerate(train_loader)):
        data = data.to(device=device)
        targets = targets.to(device=device)
        ## Forward Pass
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores,targets)
        loss.backward()
        optimizer.step()
        predictions = torch.argmax(scores, dim=-1)
        train_samples += predictions.size(0)
        train_accuracy += (predictions == targets).sum()
        train_loss += loss.item()
    print(f"{epoch+1}. Accuracy in epoch {epoch}: {(train_accuracy/train_samples)*100:.2f}")
    print(f"{epoch+1}. TrainLoss in epoch {epoch}: {train_loss/len(train_loader):.3f}")
    
    scheduler.step()

    with torch.no_grad():
        #model.eval()
        test_loss = 0
        test_accuracy = 0
        test_samples = 0
        for batch_idx, (data,targets) in (enumerate(test_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            ## Forward Pass
            scores = model(data)
            loss = criterion(scores,targets)
            predictions = torch.argmax(scores, dim=-1)
            test_accuracy += (predictions == targets).sum()
            test_samples += predictions.size(0)
            test_loss += loss.item()
        print(f"{epoch+1}. Accuracy in epoch {epoch}: {(test_accuracy / test_samples)*100:.2f}")
        print(f"{epoch+1}. TestLoss in epoch {epoch}: {test_loss / len(test_loader):.3f}")  