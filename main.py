#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   main.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:37   Daic       1.0        
'''
import os
import argparse
from untils import *
from models import *

parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--input_json', type=str, default='/home/daic/kaggle/steel/data/dataset_k1.json')
parser.add_argument('--image_path', type=str, default='/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_images')
parser.add_argument('--mask_path', type=str, default='/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_masks')

#output_path
parser.add_argument('--out_path', type=str,  default='/home/daic/kaggle/steel/data/save')
parser.add_argument('--train_name', type=str,  default='seg_se50_radam_k1')

#optimizer
parser.add_argument('--optimizer', type=str,  default='radam')
parser.add_argument('--lr', type=float,  default=1e-4)
parser.add_argument('--weight_decay', type=float,  default=0)
parser.add_argument('--momentum', type=float,  default=0.9)
##if use cosine decay:
parser.add_argument('--warmup_lr', type=float,  default=1e-6)
parser.add_argument('--min_lr', type=float,  default=1e-6)
parser.add_argument('--min_lr_epoch', type=int,  default=2)
##if use epoch schedule
parser.add_argument('--decay_step', type=int,  default=5)
parser.add_argument('--decay_rate', type=float,  default=0.6)

#model
parser.add_argument('--encoder', type=str,  default='se_resnext50_32x4d')
parser.add_argument('--num_class', type=int,  default=4)
parser.add_argument('--loss', type=str,  default='Dice')#Jaccard #Dice #BCEJaccard #BCEDice
parser.add_argument('--activation', type=str,  default='sigmoid')#sigmoid, softmax

#train and eval
parser.add_argument('--train_bch', type=int,  default=4)
parser.add_argument('--val_bch', type=int,  default=4)
parser.add_argument('--num_worker', type=int,  default=8)
parser.add_argument('--max_epoch', type=int,  default=60)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--seed', type=int,  default=6666)
parser.add_argument('--gpu_id', type=str,  default='3')

#resume
parser.add_argument('--resume_path', type=str,  default='')


opt = parser.parse_args()

#os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)  # if you are using multi-GPU.
np.random.seed(opt.seed)  # Numpy module.
os.environ["PYTHONHASHSEED"] = str(opt.seed)
np.random.seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.backends.cudnn.deterministic = True


# # segmentation
# trainer = SegTrainer(opt)
#
# if len(opt.resume_path)!=0:
#     trainer.resum_load(opt.resume_path)
#
# trainer.train_model()

#classfication
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
opt.input_json = '/home/daic/kaggle/steel/data/dataset_k5.json'
trainer = ClsTrainer(opt,activa='relu',save_path='/home/daic/kaggle/steel/data/save/exp190929/cls/cls_model5.pth')
trainer.train_model()