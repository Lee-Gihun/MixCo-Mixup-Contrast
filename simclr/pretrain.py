import torch
import torch.nn as nn
from models.resnet_simclr import ResNetSimCLR
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from loss.nt_xent import NTXentLoss
from loss.mix_loss import SoftCrossEntropy
import os
import shutil
import sys
import sys
import builtins
from data_aug import *
from utils import *

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np
from utils import *


def main():
    args = parse_args()
    
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
    
    model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim).to(args.gpu)
    
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
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
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

    # Data loader
    train_loader, train_sampler = data_loader(args.dataset, 
                                              args.data_path, 
                                              args.batch_size, 
                                              args.num_workers, 
                                              download=args.download, 
                                              distributed=args.distributed, 
                                              supervised=False)
    
    #optimizer = torch.optim.Adam(model.parameters(), 3e-4, weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0,
                                                           last_epoch=-1)
    
    criterion = NTXentLoss(args.gpu, args.batch_size, args.temperature, True).cuda(args.gpu)
    
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
            
    if apex_support and args.fp16_precision:
        model, optimizer = amp.initialize(model, optimizer,
                                          opt_level='O2',
                                          keep_batchnorm_fp32=True)

    cudnn.benchmark = True
    
    train(model, train_loader, train_sampler, criterion, optimizer, scheduler, args, ngpus_per_node)
           

def _step(model, xis, xjs, criterion):

    # get the representations and the projections
    ris, zis = model(xis)  # [N,C]

    # get the representations and the projections
    rjs, zjs = model(xjs)  # [N,C]

    # normalize projection feature vectors
    zis = F.normalize(zis, dim=1)
    zjs = F.normalize(zjs, dim=1)

    logits, labels, loss = criterion(zis, zjs)
    return zis, zjs, loss, (logits, labels)


def _mix_step(model, xis, xjs, zis, zjs, loss, args):
    crit = SoftCrossEntropy()
    B = xis.shape[0]
    assert B % 2 == 0
    sid = int(B/2)
    
    for x, z in zip([xis, xjs], [zjs, zis]):
        x_1, x_2 = x[:sid], x[sid:]

        # each image get different lambda
        lam = torch.from_numpy(np.random.uniform(0, 1, size=(sid,1,1,1))).float().to(x.device)
        imgs_mix = lam * x_1 + (1-lam) * x_2

        _, z_mix = model(imgs_mix)
        lbls_mix = torch.cat((torch.diag(lam.squeeze()), torch.diag((1-lam).squeeze())), dim=1)
        z_mix = F.normalize(z_mix, dim=1)

        logits_mix = torch.mm(z_mix, z.transpose(0, 1)) # N/2 * N
        logits_mix /= args.mix_temperature
        loss += crit(logits_mix, lbls_mix) / 2
        
    return loss
    

def train(model, train_loader, train_sampler, criterion, optimizer, scheduler, args, ngpus_per_node):

    n_iter = 0
    valid_n_iter = 0
    avg_loss = avg_mix_loss = avg_acc = 0
    best_valid_loss = np.inf
    
    model.train()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                train_sampler.set_epoch(epoch)

            for (xis, xjs), _ in train_loader:
                optimizer.zero_grad()

                xis = xis.to(args.gpu)
                xjs = xjs.to(args.gpu)

                zis, zjs, loss, logit_label = _step(model, xis, xjs, criterion)
                avg_loss += loss.item()
                
                if args.mix:
                    loss = _mix_step(model, xis, xjs, zis, zjs, loss, args)
                    avg_mix_loss += loss.item()

                acc1 = accuracy(logit_label[0], logit_label[1])
                avg_acc += acc1[0].item()

                # logging
                if args.log_every_n_steps != -1 and n_iter % args.log_every_n_steps == 0:
                    print('iter {:d}: train loss {:.6f}, mixed loss {:.6f}, acc@1 {:.2f}'.format(
                        n_iter, avg_loss/args.log_every_n_steps, avg_mix_loss/args.log_every_n_steps, avg_acc/args.log_every_n_steps))
                    avg_loss = avg_mix_loss = avg_acc = 0
                    
                if apex_support and args.fp16_precision:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # warmup for the first 10 epochs
            if epoch >= 10:
                scheduler.step()
            print('Learning rate of epoch %d : %s' % (epoch, scheduler.get_last_lr()[0]))
            
            if args.save_freq != -1 and (epoch+1) % args.save_freq == 0: 
                if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                    save_checkpoint({
                        'epoch': epoch+1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        }, is_best=False, path='./results/pretrained', filename='{}_{:04d}.pth.tar'.format(args.exp_name, epoch+1))

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': args.epochs,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, path='./results/pretrained', filename='{}.pth.tar'.format(args.exp_name))
     
    
if __name__ == "__main__":
    main()
