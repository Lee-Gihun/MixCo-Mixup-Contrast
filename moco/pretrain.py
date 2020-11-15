import torch
import torch.nn as nn
import torch.nn.functional as F
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

import os, math, random, time, shutil, builtins, argparse, warnings, json

from data_utils import *
from models import *
from utils import *

MODEL_NAMES = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def main():
    args = parse_args()
    
    # check the base_encoder architecture
    assert args.arch in MODEL_NAMES

    if args.seed is not None:
        fix_seed(args.seed)
        
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in args.gpu])
    if len(args.gpu) > 1:
        args.gpu = None
        args.single_gpu = False
    else:
        args.gpu = [0]
        args.single_gpu = True
        
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print(args.gpu)
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    model = Encoder(models.__dict__[args.arch], 
                    args.algo, 
                    dim=args.moco_dim, 
                    num_splits=int(args.batch_size/2),
                    K=args.moco_k,
                    m=args.moco_m,
                    T=args.moco_t,
                    mix_T=args.mixco_t,
                    mlp=args.mlp,
                    single_gpu=args.single_gpu)
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    #else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        #raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    args.mix_param = 0.0 if args.algo == 'moco' else args.mix_param
    criterion = MixcoLoss(args.mix_param).cuda(args.gpu)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loader
    train_loader, train_sampler = data_loader(args.dataset, 
                                              args.data_path, 
                                              args.batch_size, 
                                              args.num_workers, 
                                              download=args.download, 
                                              distributed=args.distributed, 
                                              aug_plus=args.aug_plus, 
                                              supervised=False)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1, acc5 = train(train_loader, model, optimizer, criterion, epoch+1, args)
        
        if args.save_freq != -1 and (epoch+1) % args.save_freq == 0: 
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch+1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    }, is_best=False, path='./results/pretrained', filename='{}_{:04d}.pth.tar'.format(args.exp_name, epoch+1))

    # always saves at the end of training    
    else:
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch+1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, path='./results/pretrained', filename='{}.pth.tar'.format(args.exp_name))
            
            update_json(args.exp_name, 'pretrain', [acc1.item(), acc5.item()])

            
def train(train_loader, model, optimizer, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        # compute output
        outputs = model(im_q=images[0], im_k=images[1])
        loss = criterion(outputs)

        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs[0], outputs[1], topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
