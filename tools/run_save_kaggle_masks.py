#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   run_save_masks.py
@Desciption     :   Kaggle encoding mask to nparray as save them
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:38   Daic       1.0        
'''
import os
import json
import pandas as pd
import numpy as np

def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df):
    '''Given a row index, return image_id and mask (256, 1600, 4)'''
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((256, 1600, 4), dtype=np.float32)  # float32 is V.Imp
    # 4:class 1～4 (ch:0～3)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(256 * 1600, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(256, 1600, order='F')

    return masks

def load_df(df_path):
    df = pd.read_csv(df_path)
    # some preprocessing
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    return df

def process_label(labels):
    my_label =[]
    for label_ in labels:
        if label_ is np.nan:
            my_label.append(0)
        else:
            my_label.append(1)
    return my_label

SAVE_PATH = '/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_masks/'
IMAGE_PATH = '/home/daic/kaggle/steel/data/severstal-steel-defect-detection/train_images'
DF_PATH = '/home/daic/kaggle/steel/data/train.csv'
NOT_NULL = 1
if __name__ == '__main__':
    dataset = []
    dataf = load_df(DF_PATH)
    names = dataf.index.tolist()
    for x in range(len(names)):
        if names[x] != dataf.iloc[x].name:
            print("Index is not correct!! ")
            break
        id = names[x][:-4]
        defects = process_label(dataf.iloc[x][:4].values)
        tmp = {
            'id':id,
            'defects':defects
        }
        dataset.append(tmp)
        if NOT_NULL in defects:
            mask  = make_mask(x,dataf)
            np.save(os.path.join(SAVE_PATH,names[x][:-4]+'.npy'),mask)


        if x%20 ==0:
            print("%d / %d"%(x,len(names)))

    json.dump(dataset,open('/home/daic/kaggle/steel/data/dataset_no_split.json','w'))



