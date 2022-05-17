import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision
import wandb
from torchgpipe import GPipe
from torchgpipe.balance import balance_by_time

from typing import cast
import time
from collections import OrderedDict

from resnet import *

wandb.init(project="TinyImagenetGpipe", name="ResNet50")

model_names = {
    'resnet18'   : inet_resnet18(),
    'resnet50'   : inet_resnet50(),
    'resnet152'  : inet_resnet152(),
    
}


epochs          = 90
microbatch_size = 100
log_inter       = 1
cores_gpu       = 8
microbatches    = 8
batch_size      = microbatch_size * microbatches

def main():
    parser = argparse.ArgumentParser(description='D-DNN imagenet benchmark')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # Value of args.synthetic_data may seem confusing, but those values 
    # come from bash and there 0=true and all else =false
    parser.add_argument('-s', '--synthetic_data', type=int, default=0,
                        help="Use synthetic data")
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    #---------------------------------------------------------------------------------
    # Move model to GPU.
    print("=> creating model '{}'".format(args.arch))
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
#     model = model_names[args.arch].cuda()
    model = model_names[args.arch].to(device)
    batch_size = 100
    partitions = torch.cuda.device_count()
    if args.synthetic_data == -1:
        sample = torch.empty(batch_size, 3, 512, 512)
    else:
        sample = torch.empty(batch_size, 3, 64, 64)
    balance = balance_by_time(partitions, model, sample)
    model = GPipe(model, balance, chunks=microbatches)
    # Training
    wandb.watch(model)

    #---------------------------------------------------------------------------------
    devices = list(model.devices)
    in_device  = devices[0]
    out_device = devices[-1]
    torch.cuda.set_device(in_device)

    throughputs = []
    elapsed_times = []
    #---------------------------------------------------------------------------------

    # define optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    #---------------------------------------------------------------------------------
    cudnn.benchmark = True

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

#     train_dir = '/scratch/mnk2978/finalproj_old/tiny-imagenet-200/train'
    train_dir = '../tiny-imagenet-200/train'
    trainset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_train)
    

#     test_dir = '/scratch/mnk2978/finalproj_old/tiny-imagenet-200/val/images'
    test_dir = '../tiny-imagenet-200/val/images'
    testset = torchvision.datasets.ImageFolder(
        test_dir, transform=transform_test) 
    val_loader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False,
        num_workers=2, pin_memory=True)
    classes = 200
    img_size = 64
    num_classes = 200

    ############################################################
    # makes sure that each process gets a different slice of the training data
    # during distributed training
#     train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    # notice we turn off shuffling and use distributed data sampler
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True,
        num_workers=2, pin_memory=True)
    #---------------------------------------------------------------------------------

    for epoch in range(epochs):
        throughput, elapsed_time, loss, train_acc = run_epoch(train_loader, val_loader, model, optimizer, epoch, args, in_device, out_device)

        throughputs.append(throughput)
        elapsed_times.append(elapsed_time)

        _, valid_accuracy = evaluate(val_loader, model, args, in_device, out_device)
        wandb.log({"train accuracy" : train_acc, "test accuracy" : valid_accuracy, "train_loss" : loss}, step=epoch+1)
    
    n = len(throughputs)
    throughput = sum(throughputs) / n if n > 0 else 0.0
    elapsed_time = sum(elapsed_times) / n if n > 0 else 0.0
    print('valid accuracy: %.4f | %.3f samples/sec, %.3f sec/epoch (average)'
               '' % (valid_accuracy, throughput, elapsed_time))


def run_epoch(train_loader, test_loader, model, optimizer, epoch, args, in_device, out_device):
    torch.cuda.synchronize(in_device)
    tick = time.time()

    steps = len(train_loader)
    data_trained = 0
    loss_sum = torch.zeros(1, device=out_device)
    model.train()
    total = 0
    correct = 0
    for i, (input, target) in enumerate(train_loader):
        data_trained += batch_size
        input = input.to(device=in_device, non_blocking=True)
        target = target.to(device=out_device, non_blocking=True)

        output = model(input)
        loss = F.cross_entropy(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.detach() * batch_size
        total += target.size(0)
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        if i % log_inter == 0:
            percent = i / steps * 100
            throughput = data_trained / (time.time()-tick)

            dev = torch.cuda.current_device()
            stats = torch.cuda.memory_stats(device=dev)
            max_mem = torch.cuda.get_device_properties(dev).total_memory
            print('train | %d/%d epoch (%d%%) | %.3f samples/sec (estimated) | mem (GB): %.3f (%.3f) / %.3f'
                '' % (epoch+1, epochs, percent, throughput, 
                      stats["allocated_bytes.all.peak"] / 10**9,
                      stats["reserved_bytes.all.peak"] / 10**9,
                      float(max_mem) / 10**9))
    
    torch.cuda.synchronize(in_device)
    tock = time.time()

    train_loss = loss_sum.item() / data_trained
    valid_loss, valid_accuracy = evaluate(test_loader, model, args, in_device, out_device)
    torch.cuda.synchronize(in_device)

    elapsed_time = tock - tick
    throughput = data_trained / elapsed_time
    print('%d/%d epoch | train loss:%.3f %.3f samples/sec | '
        'valid loss:%.3f accuracy:%.3f'
        '' % (epoch+1, epochs, train_loss, throughput,
                valid_loss, valid_accuracy))

    return throughput, elapsed_time, loss_sum/(i+1), 100.*(correct/total)
            


def evaluate(test_loader, model, args, in_device, out_device):
    tick = time.time()
    steps = len(test_loader)
    data_tested = 0
    loss_sum = torch.zeros(1, device=out_device)
    accuracy_sum = torch.zeros(1, device=out_device)
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            current_batch = input.size(0)
            data_tested += current_batch
            input = input.to(device=in_device)
            target = target.to(device=out_device)

            output = model(input)

            loss = F.cross_entropy(output, target)
            loss_sum += loss * current_batch

            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum()
            accuracy_sum += correct

            if i % log_inter == 0:
                percent = i / steps * 100
                throughput = data_tested / (time.time() - tick)
                print('valid | %d%% | %.3f samples/sec (estimated)'
                    '' % (percent, throughput))

    loss = loss_sum / data_tested
    accuracy = (accuracy_sum / data_tested) * 100

    return loss.item(), accuracy.item()


if __name__ == '__main__':
    main()