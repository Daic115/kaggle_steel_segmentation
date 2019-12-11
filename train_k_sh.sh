#!/usr/bin/env bash

cd kaggle/segmentation
conda activate bert

#840
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k1.json \
--train_name seg_se50_radam_k1 \
--gpu_id 2

#847
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k2.json \
--train_name seg_se50_radam_k2 \
--gpu_id 3
#830
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k3.json \
--train_name seg_se50_radam_k3 \
--gpu_id 1 --lr 8e-5
#839
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k4.json \
--train_name seg_se50_radam_k4 \
--gpu_id 2 --lr 8e-5
#836
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k5.json \
--train_name seg_se50_radam_k5 \
--gpu_id 3 --lr 8e-5

#resume k3 k4 k5
python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k3.json \
--train_name seg_se50_radam_k3 \
--gpu_id 1 --lr 0.000002 \
--resume_path /home/daic/kaggle/steel/data/save/seg_se50_radam_k3/modelk3.pth \
--max_epoch 10

python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k4.json \
--train_name seg_se50_radam_k4 \
--gpu_id 2 --lr 0.000002 \
--resume_path /home/daic/kaggle/steel/data/save/seg_se50_radam_k4/modelk4.pth \
--max_epoch 10

python3 main.py --input_json /home/daic/kaggle/steel/data/dataset_k5.json \
--train_name seg_se50_radam_k5 \
--gpu_id 3 --lr 0.000002 \
--resume_path /home/daic/kaggle/steel/data/save/seg_se50_radam_k5/modelk5.pth \
--max_epoch 10