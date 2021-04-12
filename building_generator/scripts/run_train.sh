#!/usr/bin/env bash


label_dir=XXX
image_dir=XXX
num_labels=2
batchSize=8
name=inria
python train.py --name $name --nThreads 0 --gpu_ids 0 --model pix2pix --dataset_mode custom --label_dir $label_dir --image_dir ${image_dir} --label_nc $num_labels --batchSize $batchSize --load_size 284 --crop_size 256 --no_instance --niter_decay 55 --niter 50 --display_freq 10