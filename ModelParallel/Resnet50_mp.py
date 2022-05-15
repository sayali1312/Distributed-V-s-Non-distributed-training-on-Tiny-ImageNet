'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
# !pip3 install wandb
# !pip3 install torchsummary

#import os
import sys
import time
import math
import shutil 

import torch.nn as nn
import torch.nn.init as init

import wandb
import warnings
import torch

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import sys
import matplotlib.pyplot as plt
from torchsummary import summary
import numpy as np
import time

wandb.init(project="TinyImagenetModelParallel", name="Resnet50ModelParallel")



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

_, term_width = shutil.get_terminal_size()
#_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

warnings.simplefilter(action='ignore', category=FutureWarning)

pkgpath = './'
save_path = './results/'

if os.path.isdir(save_path) == False:
    os.makedirs(save_path)

sys.path.append(pkgpath)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

train_dir = '/scratch/mnk2978/hpml/finalproj/tiny-imagenet-200/train'

trainset = torchvision.datasets.ImageFolder(
    train_dir, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True, num_workers=0)

test_dir = '/scratch/mnk2978/hpml/finalproj/tiny-imagenet-200/val/images'
testset = torchvision.datasets.ImageFolder(
    test_dir, transform=transform_test) 
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)
classes = 200
img_size = 64

# Model
print('==> Building model..')
import torchvision.models as models

from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 200

print(ResNet)
class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
            Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)

        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,

            self.layer1,
            self.layer2
        ).to('cuda:0')

        self.seq2 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to('cuda:1')

        self.fc.to('cuda:1')

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
    
net = ModelParallelResNet50()


#Finetune Final few layers to adjust for tiny imagenet input
net.avgpool = nn.AdaptiveAvgPool2d(1)
net.fc.out_features = 200


# net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
print(net)
#%%


# Training
wandb.watch(net)


epochs = 90
def train(epoch):
    print('Epoch:{0}/{1}'.format(epoch, epochs))
    net.train()
    
    train_loss = 0 
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs = inputs.to('cuda:0')
#         targets = targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)   
        targets = targets.to(outputs.device) #match the outputs device
        loss = criterion(outputs, targets)       
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'train_Loss: %.3f | train_Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    return train_loss/(batch_idx+1), 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc_list = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to('cuda:0')
            
            outputs = net(inputs)
            targets = targets.to(outputs.device)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc_list.append(100.*correct/total)
            
            progress_bar(batch_idx, len(testloader), 'test_Loss: %.3f | test_Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            print('>>>best acc: {0}, mean: {1}, std: {2}'.format(best_acc, round(np.mean(acc_list), 2), round(np.std(acc_list), 2)))
            
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(save_path+'/checkpoint'):
            os.mkdir(save_path+'/checkpoint')
        torch.save(state, save_path+'/checkpoint/ckpt.pth')
        best_acc = acc
        print('>>>best acc:', best_acc)
    
    return test_loss/(batch_idx+1), 100.*correct/total, best_acc

test_loss = 0
test_list = []
train_list = []
epoch_list = []
train_acc_list = []
test_acc_list = []
a = time.perf_counter()
for epoch in range(start_epoch, start_epoch+epochs):
   
    epoch_list.append(epoch)
    
    train_loss, train_acc = train(epoch)
    train_list.append(train_loss)
    train_acc_list.append(train_acc)
    
    test_loss, test_acc, best_acc = test(epoch)
    test_list.append(test_loss)
    test_acc_list.append(test_acc)
    
    epoch_line = 'epoch: {0}/ total epoch: {1} '.format(epoch, epochs) 
    best_acc_line = 'best_acc: {0} '.format(best_acc)
    accuracy_line = 'train_acc: {0} %, test_acc: {1} % '.format(train_acc, test_acc)
    loss_line = 'train_loss: {0},e test_loss: {1} '.format(train_loss, test_loss)
    wandb.log({"train accuracy" : train_acc, "test accuracy" : test_acc, "train_loss" : train_loss, "test_loss" : test_loss}, step=epoch+1)

    if epoch % 1 == 0:
        plt.figure()
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(epoch_list, train_list, c = 'blue', label = 'train loss')
        ax1.plot(epoch_list, test_list, c = 'red', label = 'test loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        ax1.legend(loc=0)
        
        ax2 = plt.subplot(2, 1, 2)
        ax2.plot(epoch_list, train_acc_list, c = 'blue', label = 'train acc')
        ax2.plot(epoch_list, test_acc_list, c = 'red', label = 'test acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        ax2.legend(loc=0)
        
        plt.savefig(save_path+'/train_history.png')

    
    with open(save_path+'/logs.txt', 'a') as f:
        f.write(epoch_line + best_acc_line + accuracy_line + loss_line + '\n')
    scheduler.step()
b = time.perf_counter()
print(b-a)

