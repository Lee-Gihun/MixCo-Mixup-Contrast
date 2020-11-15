import argparse

__all__ = ['parse_args']


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
       
    # basic configs:
    parser.add_argument('--exp-name', default='test', type=str,
                        help='experiment_name')
    parser.add_argument('--data-path', metavar='DIR', default='./data/tiny-imagenet',
                        help='path to dataset')
    parser.add_argument('--dataset', metavar='DATA', default='tiny-imagenet',
                        choices=['cifar10', 'cifar100', 'tiny-imagenet', 'imagenet'])
    parser.add_argument('-j', '--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--download', action='store_true',
                        help='whether to download dataset')
    
    # train configs:
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.015, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--save-freq', default=-1, type=int,
                        help='model save frequency (default: -1)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
                        
    # distributed configs:
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=[0], nargs='*', type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # moco, mixco specific configs:
    parser.add_argument('--algo', default='moco', 
                        help='pretrain algorithm to use')
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--mix-param', default=1, type=float)
    parser.add_argument('--mixco-t', default=0.05, type=float,
                        help='softmax temperature (default: 0.05)')

    # lincls specific configs:
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to moco pretrained checkpoint')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])

    return parser.parse_args()
