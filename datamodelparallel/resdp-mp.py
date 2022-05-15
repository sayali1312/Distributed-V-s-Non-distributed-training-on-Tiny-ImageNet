import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision
import wandb
from torchsummary import summary
import numpy as np
import time

# wandb.init()

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch TinyImageNet Training')
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N',
                    help='mini-batch size (default: 1024), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--desired-acc', default=None, type=float,
                    help='Training will stop after desired-acc is reached.')

best_acc1 = 0


from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 200



    
def main():
    args = parser.parse_args()

    ############################################################
    ngpus_per_node = torch.cuda.device_count()
    
    # on each node we have: ngpus_per_node processes and ngpus_per_node gpus
    # that is, 1 process for each gpu on each node.
    # world_size is the total number of processes to run
    args.world_size = ngpus_per_node * args.world_size

    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    ############################################################


def main_worker(gpu, ngpus_per_node, args):
    """
    :param gpu: this is the process index, mp.spawn will assign this for you, goes from 0 to ngpus - 1 for the curr node
    :param ngpus_per_node:
    :param args:
    :return:
    """
    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        run = wandb.init(
        project="TinyImagenetDataParallel", name="Resnet50DataParallel",
        )
        
    global best_acc1
    print("Use GPU: {} for training".format(gpu))

    ############################################################
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes across all nodes
    # This is “blocking,” meaning that no process will continue until all processes have joined.
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend,
                            world_size=args.world_size, rank=args.rank)
    ############################################################

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)

    print("=> creating model '{}'".format(args.arch))


    # setup mp_model and devices for this process
    dev0 = (args.rank * 2) % args.world_size
    dev1 = (args.rank * 2 + 1) % args.world_size
    num_classes = 200
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
            ).cuda(dev0)

            self.seq2 = nn.Sequential(
                self.layer3,
                self.layer4,
                self.avgpool,
            ).cuda(dev1)

            self.fc.cuda(dev1)

        def forward(self, x):
            x = x.to(dev0)
            print("input device:", x.get_device())
            x = self.seq2(self.seq1(x).to(dev1))
            return self.fc(x.view(x.size(0), -1))

    mp_model = ModelParallelResNet50()


    #Finetune Final few layers to adjust for tiny imagenet input
#     mp_model.avgpool = nn.AdaptiveAvgPool2d(1)
#     mp_model.fc.out_features = 200

        
    ############################################################
    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(gpu)
    print(gpu)
    mp_model.cuda(gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    args.batch_size = int(args.batch_size / ngpus_per_node)  # calculate local batch size for each GPU
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    model = torch.nn.parallel.DistributedDataParallel(mp_model)
    ############################################################

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # best_acc1 may be from a checkpoint from a different GPU
            best_acc1 = best_acc1.to(gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn will look for the optimal set of algorithms for that
    # particular configuration. this will have faster runtime if
    # your input sizes does not change at each iteration
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

    train_dir = '/scratch/mnk2978/hpml/finalproj/tiny-imagenet-200/train'

    trainset = torchvision.datasets.ImageFolder(
        train_dir, transform=transform_train)
    

    test_dir = '/scratch/mnk2978/hpml/finalproj/tiny-imagenet-200/val/images'
    testset = torchvision.datasets.ImageFolder(
        test_dir, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    classes = 200
    img_size = 64
    num_classes = 200

    ############################################################
    # makes sure that each process gets a different slice of the training data
    # during distributed training
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    # notice we turn off shuffling and use distributed data sampler
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    ############################################################

    
    end = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_acc, train_l = train(train_loader, model, criterion, optimizer, epoch, gpu, args, run, dev0)

        # evaluate on validation set
        acc1, testloss = validate(testloader, model, criterion, gpu, args, dev0)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        do_log = run is not None
        if do_log:
            run.log({"train accuracy" : train_acc, "test accuracy" : acc1, "train_loss" : train_l, "test_loss" : testloss}, step=epoch+1)
        if args.rank % ngpus_per_node == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        # stop training once reach desired accuracy
        if args.desired_acc and best_acc1 >= args.desired_acc:
            time_elapsed = time.time() - end
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'training_time': time_elapsed,
            }, is_best)
            print("Reached acc of: {:6.2f}; Time elapsed: {:6.3f}".format(args.desired_acc, time_elapsed))
            break


def train(train_loader, model, criterion, optimizer, epoch, gpu, args, run, dev0):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    is_master = args.local_rank == 0
    do_log = run is not None
    
    

    # switch to train mode
    model.train()
    
    # watch gradients only for rank 0
    if is_master:
        run.watch(model)

    end = time.time()
    train_loss = 0
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(dev0, non_blocking=True)
#         target = target.cuda(gpu, non_blocking=True)
        # compute output
        
        output = model(images)
        print(output)
        targets = targets.cuda(output.device, non_blocking=True)
        
        loss = criterion(output, target)
        train_loss += loss.item()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return top1.avg, train_loss/(i+1)

def validate(val_loader, model, criterion, gpu, args, dev0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    test_loss = 0
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(dev0, non_blocking=True)
#             target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            targets = targets.cuda(output.device, non_blocking=True)
            loss = criterion(output, target)
            test_loss += loss.item()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, test_loss/(i+1)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    a = time.perf_counter()
    main()
    b = time.perf_counter()
    print(b-a)
    cleanup()