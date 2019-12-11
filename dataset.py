#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   dataset.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:36   Daic       1.0        
'''
import os
import cv2
import json
import torch
import numpy as np
import albumentations as albu
from torch.utils.data import  DataLoader
from torch.utils.data import  Dataset as BaseDB
def get_training_augmentation():
    train_transform = [
        albu.OneOf(
            [
                albu.NoOp(p=1),
                albu.RandomResizedCrop(256, 1600, scale=(0.7, 1.0), ratio=(5.5, 6.6), p=1.0),
                albu.Rotate(limit=(-10, 10), p=0.8),
            ],
            p=0.8,
        ),

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
    ]
    return albu.Compose(train_transform)

def preprocess(img):
    img = ((img / 255.)-0.5)*2
    img = img[::,::,0].copy()
    return img

def argue_sampler(infos):
    argue_times = [3,7,0,3]
    for xi in range(len(infos)):
        tmp = infos[xi]
        if tmp['defects'][1] == 1:
            for time in range(argue_times[1]):
                infos.append(tmp)
        elif tmp['defects'][3] == 1:
            for time in range(argue_times[3]):
                infos.append(tmp)
        elif tmp['defects'][0] == 1:
            for time in range(argue_times[0]):
                infos.append(tmp)
    return infos


class SegSteelDataset(BaseDB):
    def __init__(self,opt, phase ,augmentation=None):
        self.phase = phase
        self.image_path = opt.image_path
        self.mask_path = opt.mask_path

        self.infos = json.load(open(opt.input_json))
        if self.phase == 'train':
            self.infos = argue_sampler(self.infos)

        self.dataset = [tmp for tmp in self.infos if tmp['split'] == self.phase]
        self.augmentation = augmentation

    def __getitem__(self, idx):

        id = self.dataset[idx]['id']
        img = cv2.imread(os.path.join(self.image_path, id +'.jpg'))
        mask = np.load(os.path.join(self.mask_path,self.dataset[idx]['id'] +'.npy'))#256 1600 4
        if self.phase == 'train':
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        img = torch.from_numpy(preprocess(img)).unsqueeze(0).float()      # 1 256 1600
        mask = torch.from_numpy(mask).permute(2,0,1).float()                # 4 256 1600

        return img, mask

    def __len__(self):
        return len(self.dataset)


class ClsSteelDataset(BaseDB):
    def __init__(self,opt, phase ,augmentation=None):
        self.phase = phase
        self.image_path = opt.image_path
        #self.mask_path = opt.mask_path

        self.infos = json.load(open(opt.input_json))
        if self.phase == 'train':
            self.infos = argue_sampler(self.infos)

        self.dataset = [tmp for tmp in self.infos if tmp['split'] == self.phase]
        if self.phase == 'train':
            self.no_defect = [tmp for tmp in self.infos if tmp['split'] == 'cls_train']
        else:
            self.no_defect = [tmp for tmp in self.infos if tmp['split'] == 'cls_val']
        self.dataset = self.dataset + self.no_defect

        self.augmentation = augmentation

    def __getitem__(self, idx):

        id = self.dataset[idx]['id']
        img = cv2.imread(os.path.join(self.image_path, id +'.jpg'))
        if 1 in self.dataset[idx]['defects']:
            label = 1
        else:
            label = 0
        #mask = np.load(os.path.join(self.mask_path,self.dataset[idx]['id'] +'.npy'))#256 1600 4
        if self.phase == 'train':
            sample = self.augmentation(image=img)
            img = sample['image']

        img = torch.from_numpy(preprocess(img)).unsqueeze(0).float()      # 1 256 1600
        #mask = torch.from_numpy(mask).permute(2,0,1).float()                # 4 256 1600

        return img , label#, mask

    def __len__(self):
        return len(self.dataset)
# if __name__ == '__main__':
#     import imageio
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     # Input paths
#     parser.add_argument('--input_json', type=str,
#                         default='/home/daic/kaggle/steel/data/dataset_k1.json')
#     parser.add_argument('--image_path', type=str,
#                         default='/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_images')
#     parser.add_argument('--mask_path', type=str,
#                         default='/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_masks')
#     opt = parser.parse_args()
#
#     argumentation = get_training_augmentation()
#     dataset = SegSteelDataset(opt,'train',augmentation=argumentation)
#     print(len(dataset))
#
#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,
#         num_workers=1,
#         # pin_memory=True,
#         shuffle=True,
#     )
#     for i ,data in enumerate(dataloader):
#         img, mask = data
#
#         img = img.squeeze().numpy()
#         mask = mask.squeeze().numpy()[2]
#         if mask.max() > 0.:
#             imageio.imwrite("/home/daic/kaggle/steel/data/visual/"+str(i)+'im.jpg',img)
#             imageio.imwrite("/home/daic/kaggle/steel/data/visual/" + str(i) + 'ms.jpg', mask)