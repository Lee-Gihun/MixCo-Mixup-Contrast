## How to Reproduce our Results

This repository contains PyTorch implementation code for the paper **MixCo: Mix-up Contrastive Learning for Visual Representation (https://arxiv.org/abs/2010.06300)** that is accepted in [NeurIPS 2020 Workshop on Self-Supervised Learning: Theory and Practice](https://nips.cc/Conferences/2020/ScheduleMultitrack?event=16146).

This is an instruction to reproduce our results, based on the source code we have provided.

### Prerequisites

1. You should download the Tiny-ImageNet dataset. To download the images, go to [https://tiny-imagenet.herokuapp.com/](https://tiny-imagenet.herokuapp.com/) and click 'Download Tiny ImageNet' button. Equivalently, try
```sh
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip 
```
2. Unzip the file and place the folder into `[your_data_path]`.
3. Also, download the ImageNet train dataset.
4. For linear classification, you may also need CIFAR10 and CIFAR100 dataset. If you do not have them, give the argument `--download` when running `lincls.py`. Then, it will download the dataset before training.

### Structure
This repository contains python files that can train the model with mixup-based representaion learning.

`pretrain.py` pretrains the model in unsupervised manner, and saves the encoder part (without classification layers). 

`lincls.py` loads and freezes the pretrained model, and then train the classifier part on the target dataset.

### Experiments

1. In `./moco/experiments/` and `./simclr/scripts/`, there are `.sh` files which include the commands that can reproduce our experimental results. Open and set the configs.
```sh
data_path="[your_data_path]"
exp_name="[experiment_name]"
```
2. Run the file. For example, if you want to pretrain the ResNet18 model with Tiny-ImageNet, and then see the linear classification results, run `exp_mix_res18_tinyimg.sh`.
```sh
bash experiments/exp_mix_res18_tinyimg.sh
```
3. You can find the pretraining and linear evaluation results in `results/results.json` file.

## Model Checkpoints
You can also download the checkpoint of MixCo with ResNet-18 architecture. Link: [google drive](https://drive.google.com/file/d/1Dg_SNGBmpyPCRIvrt8EIfQrUEIvtWasW/view?usp=sharing)

## Citing this work
```
@article{kim2020mixco,
  title={Mixco: Mix-up contrastive learning for visual representation},
  author={Kim, Sungnyun and Lee, Gihun and Bae, Sangmin and Yun, Se-Young},
  journal={arXiv preprint arXiv:2010.06300},
  year={2020}
}
```
