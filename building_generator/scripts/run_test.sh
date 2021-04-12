#!/usr/bin/env bash

# Inria
label_dir=G:/program/CD/IAug_public/spade/samples/inria/label
results_dir=./result/
num_labels=2
name=inria_pretrained
which_epoch=latest
size=64
#python test.py --name $name  --model pix2pix --results_dir $results_dir --dataset_mode custom --label_dir $label_dir --label_nc $num_labels --batchSize 4 --load_size ${size} --crop_size ${size} --no_instance --which_epoch $which_epoch


# AIRS
label_dir=G:/program/CD/IAug_public/spade/samples/airs/label
results_dir=./result/
num_labels=2
name=airs_pretrained
which_epoch=latest
size=256
python test.py --name $name  --model pix2pix --results_dir $results_dir --dataset_mode custom --label_dir $label_dir --label_nc $num_labels --batchSize 4 --load_size ${size} --crop_size ${size} --no_instance --which_epoch $which_epoch

