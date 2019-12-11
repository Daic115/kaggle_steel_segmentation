#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   run_kaggle_steel_split.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 19:39   Daic       1.0        
'''
import json
import copy
import random


da = json.load(open('/home/daic/kaggle/steel/data/dataset_no_split.json'))
###
#   cate            0       1       2       3
#   mask_num        897     247     5150    801
#   image_num       860     247     4759    800
#   sp              172     50      952     160

###
# cate0    min:1.000    mean:1180.486   max:17459.000
# cate1    min:1.000    mean:1958.852   max:9641.000
# cate2    min:1.000    mean:7385.090   max:329012.000
# cate3    min:1.000    mean:12049.702   max:168151.000

cate0 = []
cate1 = []
cate2 = []
cate3 = []
for x in range(len(da)):
    if da[x]['defects'][1]==1:
        cate1.append(x)
    elif da[x]['defects'][3]==1:
        cate3.append(x)
    elif da[x]['defects'][0]==1:
        cate0.append(x)
    elif da[x]['defects'][2]==1:
        cate2.append(x)

print("dataset having %d defected images!"%(len(cate3)+len(cate2)+len(cate1)+len(cate0)))

###do split

da1 = copy.deepcopy(da)
da2 = copy.deepcopy(da)
da3 = copy.deepcopy(da)
da4 = copy.deepcopy(da)
da5 = copy.deepcopy(da)
random.shuffle(cate0)
random.shuffle(cate1)
random.shuffle(cate2)
random.shuffle(cate3)

sp1 = cate0[172*4:] +cate1[50*4:]+cate2[952*4:]+cate3[160*4:]
sp2 = cate0[172*3:172*4] +cate1[50*3:50*4]+cate2[952*3:952*4]+cate3[160*3:160*4]
sp3 = cate0[172*2:172*3] +cate1[50*2:50*3]+cate2[952*2:952*3]+cate3[160*2:160*3]
sp4 = cate0[172*1:172*2] +cate1[50*1:50*2]+cate2[952*1:952*2]+cate3[160*1:160*2]
sp5 = cate0[172*0:172*1] +cate1[50*0:50*1]+cate2[952*0:952*1]+cate3[160*0:160*1]

def split(datas, splits):
    for x in range(len(datas)):
        if 1 in datas[x]['defects']:
            if x in splits:
                datas[x]['split'] = 'val'
            else:
                datas[x]['split'] = 'train'

        else:
            datas[x]['split'] = 'no_split'

    return datas

da1 = split(da1,sp1)
da2 = split(da2,sp2)
da3 = split(da3,sp3)
da4 = split(da4,sp4)
da5 = split(da5,sp5)

json.dump(da1,open('/home/daic/kaggle/steel/data/dataset_k1.json','w'))
json.dump(da2,open('/home/daic/kaggle/steel/data/dataset_k2.json','w'))
json.dump(da3,open('/home/daic/kaggle/steel/data/dataset_k3.json','w'))
json.dump(da4,open('/home/daic/kaggle/steel/data/dataset_k4.json','w'))
json.dump(da5,open('/home/daic/kaggle/steel/data/dataset_k5.json','w'))