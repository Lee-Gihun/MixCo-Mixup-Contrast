#!/bin/sh

# make directories
mkdir -p ./results/pretrained
mkdir -p ./results/lincls

# set configs (let other configs unchanged if not needed)
data_path="../dataset/tiny-imagenet-200"
exp_name="exp_mixco_res18_tinyimg"

# pretrain encoder
python ./pretrain.py -a resnet18 --algo mixco --batch-size 128 --lr 0.015 --epochs 100 --cos --dataset tiny-imagenet --data-path $data_path --mlp --moco-t 0.2 --aug-plus --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1

# linear classification protocol
python ./lincls.py -a resnet18 --lr 3.0 --epochs 100 --schedule 60 80  --dataset tiny-imagenet --data-path $data_path --pretrained ./results/pretrained/${exp_name}.pth.tar --exp-name $exp_name --multiprocessing-distributed --dist-url tcp://localhost:10001 --gpu 0 1 --wd 0.0 
