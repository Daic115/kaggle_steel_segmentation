#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   kaggle_sub.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/28 18:44   Daic       1.0        
'''
import os
import json
import torch
from PIL import Image
import torch.nn as nn
import pretrainedmodels
import torchvision.transforms as trn

#acc1angle  0.99395
#acc4angle  0.99435

def get_cls_model(model_name):
    model=pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained=None)
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
    return model

prepro1 = trn.Compose([
            trn.Resize([128,800]),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
prepro2 = trn.Compose([
            trn.Resize([128,800]),
            trn.RandomHorizontalFlip(1.),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
prepro3 = trn.Compose([
            trn.Resize([128,800]),
            trn.RandomVerticalFlip(1.),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
prepro4 = trn.Compose([
            trn.Resize([128,800]),
            trn.RandomHorizontalFlip(1.),
            trn.RandomVerticalFlip(1.),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


PATH = '/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_images'
datas = json.load(open('/home/daic/kaggle/steel/data/dataset_k1.json'))

model = get_cls_model("se_resnext50_32x4d")
model.load_state_dict(torch.load("/home/daic/kaggle/steel/data/save/exp190923/cls_model.pth"))
model.cuda()
model.eval()

gts = []
pre = []
def load_process_img(img_path,process):
    img = Image.open(img_path)
    img = process(img)
    return img
def load_process_img1(img_path,pro1,pro2,pro3,pro4):
    img = Image.open(img_path)
    img1 = pro1(img).unsqueeze(0)
    img2 = pro2(img).unsqueeze(0)
    img3 = pro3(img).unsqueeze(0)
    img4 = pro4(img).unsqueeze(0)
    img = torch.cat((img1,img2,img3,img4),0)
    return img

for tmp in datas:
    if 1 in tmp['defects']:
        gts.append(0)
    else:
        gts.append(1)

    #input = load_process_img(os.path.join(PATH,tmp['id']+'.jpg'),prepro1)
    input = load_process_img1(os.path.join(PATH, tmp['id'] + '.jpg'), prepro1, prepro2, prepro3, prepro4)
    #input = input.unsqueeze(0).cuda()# 1,3,xx.,xx
    input = input.cuda()
    output = model(input).squeeze()#1,1
    output = torch.sigmoid(output).data.cpu().numpy().tolist()

    pre.append(output)
    print(len(pre),len(datas))

json.dump({'pre':pre,'gt':gts},open('/home/daic/kaggle/steel/data/save/exp190923/pre_4an.json','w'))