#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   run_kaggle_mask_minsize_counting.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/30 20:29   Daic       1.0        
'''
import os
import json
import numpy as np
import cv2

'''
cate0    min:1.000    mean:1180.486   max:17459.000
cate1    min:1.000    mean:1958.852   max:9641.000
cate2    min:1.000    mean:7385.090   max:329012.000
cate3    min:1.000    mean:12049.702   max:168151.000
'''
def count_size(mask_):
    mask_ = cv2.threshold(mask_, 0.5, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask_.astype(np.uint8))
    sizes_ = []
    for c in range(1, num_component):
        p = (component == c)
        sizes_.append(p.sum().astype(float))
    return sizes_

dataset = json.load(open('/home/daic/kaggle/steel/data/dataset_k1.json'))
PATH = '/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_masks'

all_size = [[],[],[],[],[]]

for tmp in dataset:
    if tmp['split'] in ['train','val']:
        mask = np.load(os.path.join(PATH, tmp['id']+'.npy')).transpose(2,0,1).copy()
        for x in range(4):
            if tmp['defects'][x]==1:
                sizes = count_size(mask[x])
                all_size[x]+=sizes

print("cate0    min:%.3f    mean:%.3f   max:%.3f"%(min(all_size[0]),sum(all_size[0])/len(all_size[0]),max(all_size[0])))
print("cate1    min:%.3f    mean:%.3f   max:%.3f"%(min(all_size[1]),sum(all_size[1])/len(all_size[1]),max(all_size[1])))
print("cate2    min:%.3f    mean:%.3f   max:%.3f"%(min(all_size[2]),sum(all_size[2])/len(all_size[2]),max(all_size[2])))
print("cate3    min:%.3f    mean:%.3f   max:%.3f"%(min(all_size[3]),sum(all_size[3])/len(all_size[3]),max(all_size[3])))