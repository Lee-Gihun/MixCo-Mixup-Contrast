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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum value')
    parser.add_argument('--schedule', default=[60, 80], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--wd', '--weight-decay', default=1e-6, type=float,
                        metavar='W', help='weight decay (default: 1e-6)',
                        dest='weight_decay')
    parser.add_argument('--log_every_n_steps', default=10, type=int,
                        help='model save frequency (default: -1)')
    parser.add_argument("--fp16_precision", action='store_true',
                        help="whether to train with mixed precision or not")
                        
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

    # mixup specific configs:
    parser.add_argument('--mix', action='store_true', 
                        help='whether to use mixup or not')
    parser.add_argument('--out-dim', default=128, type=int,
                        help='feature dimension (default: 256)')
    parser.add_argument('--temperature', default=0.5, type=float,
                        help='softmax temperature (default: 0.5)')
    parser.add_argument('--mix-temperature', default=0.5, type=float,
                        help='softmax temperature (default: 0.5)')

    # lincls specific configs:
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to simclr pretrained checkpoint')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--optimizer', default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--cos', action='store_true')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    
    parser.add_argument('--resume', type=str, default='',
                        help='path to simclr pretrained checkpoint')
    parser.add_argument('--save-freq', type=int, default=-1,
                        help='save frequency')

    return parser.parse_args()
