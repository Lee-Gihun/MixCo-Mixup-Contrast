#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="../dataset/ILSVRC2015/Data/CLS-LOC"
exp_name="exp_mixco_res50_imagenet"

# pretrain encoder
python ./pretrain.py -a resnet50 --algo mixco --batch-size 256 --lr 0.03 --epochs 200 --cos --dataset imagenet --data-path $data_path --mlp --moco-t 0.2 --mixco-t 1 --aug-plus --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 2 3 4 5 6 7 --save-freq 10

# lincls
python ./lincls.py -a resnet50 --lr 30.0 --batch-size 256 --epochs 100 --schedule 60 80 --dataset imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 2 3 4 5 6 7


