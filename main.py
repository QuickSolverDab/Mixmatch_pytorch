#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import random
import warnings
import sys
import math
import shutil
from datetime import datetime
from collections import OrderedDict

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
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from tensorboardX import SummaryWriter
from torchsummary import summary

"""Private functions"""
from datasetload import *
from utils import *
from submain import *
from ema import *

""" networks """
from models.networks import *

""" argparse
    python -W ignore main.py --gpu 0
"""
dims   = {'cifar10': (3,32,32), 'cifar100':(3,32,32), 'ImageNet':(3,224,224)}
classes = {'cifar10': 10, 'cifar100':100, 'ImageNet':1000}
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--network', default='wideResNet28_2',
                    choices=['vgg11', 'vgg13', 'vgg16', 'vgg19',
                            'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnet200',
                            'preresnet56', 'preresnet101', 'preresnet110',
                            'wideResNet28_10', 'wideResNet28_4','wideResNet28_2',
                            'wideResNet16_10', 'wideResNet16_4','wideResNet16_2',
                            'pyramidnet200',],
                    type=str, help='name of dataset') # required=True,
parser.add_argument('--dataset', default='cifar10',
                    choices=['cifar10', 'cifar100', 'ImageNet'],
                    type=str, help='name of dataset') # required=True,
parser.add_argument('--workers', default=2, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--total_epochs', default=200, type=int, metavar='N',
                    help='total epoch of training')
parser.add_argument('--batch_size', default=64, type=int,
                    help='mini-batch size (default: 64), this is the total'
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num_labeled_per_classes', default=250, type=int,
                    help='# of labeled images per classes'
                         '(default: 250)')
parser.add_argument('--lr', '--learning_rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--print_test_freq', default=5, type=int,
                    metavar='N', help='print test frequency (default: per 5 epoch)')
parser.add_argument('--resume', default='./best_ckpt/model_best.pth.tar',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world_size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--port', default='8888', type=str)
parser.add_argument('--gpu', default=None, type=str, help='GPU id to use.')
parser.add_argument('--time', default=None, type=str, help='start time of training')
parser.add_argument('--mp', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--data_dir', default='/home/Databases/cifar/', type=str, metavar='FILE',
                    help='directory of datasets, example'
                         'cifar: /home/Databases/cifar/'
                         'imagenet: /home/Databases/ILSVR2012/')

# hyper-parameters of Mixmatch
parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha value for beta distribution of mixup')
parser.add_argument('--ema', default=0.999, type=float,
                    metavar='M', help='exponential moving average',)
parser.add_argument('--T', default=0.5, type=float,
                    help='temperature for sharpening')
parser.add_argument('--K', default=2, type=float,
                    help='# of iteration for unlabeld dataset augmentation')
parser.add_argument('--w', default=75, type=float,
                    help='ratio between l_x and l_u'
                         '75 for cifar10')

# best test accuracy
best_acc1 = 0

def main(args):
    # check the number and id of gpu for training
    ngpus_per_node = torch.cuda.device_count()
    args.distributed = args.world_size > 1 or args.mp
    if args.mp:
        args.world_size = ngpus_per_node * args.world_size
        print(mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:'+args.port,
                                world_size=args.world_size, rank=args.rank)
    """ Create models
        define models / check both the arcitecture and parameters
        torchvision: model = models.__dict__[args.arch]()
        print("=> creating model '{}'".format(args.arch))
    """
    print("=> creating model")
    network = networks(args.network, dataset=args.dataset)
    ema_net = networks(args.network, dataset=args.dataset)
    map(lambda p: p.detach_(), ema_net.parameters())

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            network.cuda(args.gpu)
            ema_net.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            network = torch.nn.parallel.DistributedDataParallel(network,
                                                            device_ids=[args.gpu])
        else:
            network.cuda()
            ema_net.cuda()
            network = torch.nn.parallel.DistributedDataParallel(network)

    else: #args.gpu is not None: args.gpu = 0
        args.gpu = int(args.gpu)
        torch.cuda.set_device(args.gpu)
        network.cuda(args.gpu)
        ema_net.cuda(args.gpu)

    print()
    summary(network, dims[args.dataset])
    print()

    """ Training scheme
        select optimizer
    """
    network_optm = torch.optim.Adam(network.parameters(), lr=args.lr)
    ema_optm = WeightEMA(network, ema_net, args)

    """ Optionally resume from a checkpoint
        resume: dir of check point
        load_state_dict: load pre-trained parameters
    """
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:{}'.format(args.gpu)) #map_location='cuda:{}'.format(args.gpu)
            args.start_epoch = checkpoint['epoch']
            if args.gpu is not None:
                best_acc1 = checkpoint['best_acc1']
            if args.distributed:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = 'module.' + k # remove `module.`
                    new_state_dict[name] = v
                network.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(checkpoint['state_dict'])
            network_optm.load_state_dict(checkpoint['network_optm'])
            ema_net.load_state_dict(checkpoint['ema_net'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            load_network = True
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            load_network = False

    """ Data loading code
        train/valication dataset load
    """
    train_dataset_l, train_dataset_u, train_sampler_l, train_sampler_u, val_dataset = datasetload_SSL(args.dataset, args)

    train_loader_l = DataLoader(train_dataset_l, sampler=train_sampler_l, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    train_loader_u = DataLoader(train_dataset_u, sampler=train_sampler_u, batch_size=args.batch_size,
                                num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    """ network training
    """
    if not load_network:
        for epoch in range(args.start_epoch, args.total_epochs):
            if epoch == 0:
                network_save_dir = make_save_dir(args, network=args.network)
                shutil.copy('./main.py', network_save_dir)
                shutil.copy('./submain.py', network_save_dir)
                writer = SummaryWriter(log_dir=network_save_dir)

            # train for one epoch
            tr_acc1 = network_train(train_loader_l, train_loader_u, network,
                                    network_optm, ema_optm, epoch, args)
            # evaluate on validation set
            if epoch % args.print_test_freq == 0 or epoch + 1 == args.total_epochs:
                acc1 = validate(val_loader, ema_net, args)
                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                best_acc1 = max(acc1, best_acc1)
                save_checkpoint(network_save_dir, args, is_best,
                                {'epoch': epoch + 1,
                                'state_dict': network.state_dict(),
                                'best_acc1': best_acc1,
                                'network_optm': network_optm.state_dict(),
                                'ema_net': ema_net.state_dict()
                                })
                record(network_save_dir, args,
                       {'epoch': epoch + 1, 'train_acc': tr_acc1, 'test_acc': acc1})
                writer.add_scalar('Student/train/acc1', torch.tensor(tr_acc1), epoch)
                writer.add_scalar('Student/test/acc1', torch.tensor(acc1), epoch)

if __name__ == '__main__':
    args = parser.parse_args()
    # For multi-processing
    if len(args.gpu) != 1:
        gpu_devices = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        args.mp = True
        args.gpu = None
        print('Multi_process mode is {} using {}'.format(args.mp, gpu_devices))

    args.num_classes = classes[args.dataset]
    for i in range(1):
        args.repeat = str(i)
        args.time = datetime.now().strftime("%y%m%d_%H-%M-%S")
        main(args)
