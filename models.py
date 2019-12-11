#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   models.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:38   Daic       1.0        
'''
import torch
import torch.nn as nn
import numpy as np
from untils import *
from dataset import *
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR

class SegTrainer(object):
    def __init__(self, opt):
        self.opt = opt
        self.train_bch =opt.train_bch
        self.val_bch = opt.val_bch
        self.num_worker  = opt.num_worker
        self.model = smp.Unet(
            encoder_name=self.opt.encoder,
            encoder_weights='imagenet',
            classes=self.opt.num_class,
            activation=self.opt.activation,
        )
        self.model.encoder.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                               bias=False)
        self.device = 'cuda'
        self.val_dataset = SegSteelDataset(self.opt, 'val')
        self.train_arguementation = get_training_augmentation()
        self.train_dataset = SegSteelDataset(self.opt,'train',self.train_arguementation)
        print("Initial dataset success! The train length: %d  The val length: %d"%
              (len(self.train_dataset),len(self.val_dataset)))

        self.train_loader = DataLoader(dataset=self.train_dataset,shuffle=True ,
                                       batch_size=self.train_bch, num_workers=self.num_worker)
        self.val_loader   = DataLoader(dataset=self.train_dataset,shuffle=False ,
                                       batch_size=self.val_bch, num_workers=self.num_worker-4)

        self.loss = smp.utils.losses.DiceLoss(eps=1.)#JaccardLoss #DiceLoss #BCEJaccardLoss #BCEDiceLoss
        self.metrics = [
            smp.utils.metrics.IoUMetric(eps=1.),
            smp.utils.metrics.FscoreMetric(eps=1.),
        ]
        self.optimizer = build_optimizer(self.model.parameters(), opt)

        self.max_epoch = self.opt.max_epoch
        self.max_score = 0.

    def resum_load(self,path):
        self.model =torch.load(path)
        print("Loading success from %s"%path)

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.model(images)
        loss = self.loss(outputs, masks)
        return loss, outputs

    def train_model(self):
        train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
        )
        for i in range(self.max_epoch):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(self.train_loader)
            valid_logs = valid_epoch.run(self.val_loader)

            # do something (save model, change lr, etc.)
            if self.max_score < valid_logs['iou']:
                self.max_score = valid_logs['iou']
                torch.save(self.model, os.path.join(self.opt.out_path,self.opt.train_name,'model-best.pth'))
                print('Model saved!')
            if i%self.opt.decay_step == 0 and i>1:
                current_lr = get_lr(self.optimizer)
                set_lr(self.optimizer,current_lr*self.opt.decay_rate)
                print("Learning rate dropping %.6f ->>>  %.6f"%(current_lr,current_lr*self.opt.decay_rate))


class ClsTrainer(object):
    def __init__(self, opt,activa,save_path):
        self.opt = opt
        self.save_path = save_path
        self.train_bch = 32
        self.val_bch = 32
        self.num_worker  = 8
        self.model = build_cls_model('resnet18',active = activa)#relu
        self.model.cuda()
        self.val_dataset = ClsSteelDataset(self.opt, 'val')
        self.train_arguementation = get_training_augmentation()
        self.train_dataset = ClsSteelDataset(self.opt,'train',self.train_arguementation)
        print("Initial dataset success! The train length: %d  The val length: %d"%
              (len(self.train_dataset),len(self.val_dataset)))

        self.train_loader = DataLoader(dataset=self.train_dataset,shuffle=True ,
                                       batch_size=self.train_bch, num_workers=self.num_worker)
        self.val_loader   = DataLoader(dataset=self.train_dataset,shuffle=False ,
                                       batch_size=self.val_bch, num_workers=self.num_worker-4)

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = build_optimizer(self.model.parameters(), opt)
        self.scheduler = MultiStepLR(self.optimizer, milestones=[10, 20, 30 , 40], gamma=0.5)

        self.max_epoch = self.opt.max_epoch
        self.max_score = 0.

        self.best_acc = 0.5

    def resum_load(self,path):
        self.model.load_state_dict(torch.load(path))
        print("Loading success from %s"%path)

    def threshold_predict(self,out,gt,threshold):
        acc = 0
        for ix_ in range(out.shape[0]):
            pred = 0
            if out[ix_] >= threshold:
                pred = 1
            if pred == gt[ix_]:
                acc +=1
        return acc

    def eval_model(self):
        self.model.eval()
        sample_num = 0
        acc_num = 0
        for i, data in enumerate(self.val_loader):
            img = data[0].cuda()
            label = data[1]
            sample_num += img.size(0)
            outputs = self.model(img)
            outputs = torch.sigmoid(outputs).data.cpu().numpy()
            acc_num+=self.threshold_predict(outputs, label, 0.5)

        acc_ = float(acc_num)/float(sample_num)
        return acc_

    def train_model(self):
        running_loss = 0.
        item_num = 0
        for epoch in range(self.max_epoch):
            print('\nEpoch: {}'.format(epoch))
            for i, data in enumerate(self.train_loader):
                img = data[0].cuda()
                label = data[1].float().unsqueeze(1).cuda()
                item_num += label.size(0)
                outputs = self.model(img)

                loss = self.loss(outputs, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if i% 50 == 0:
                    print('[%d, %d]   loss: %.5f    lr:%.5f' %
                          (epoch, i, running_loss / item_num,self.optimizer.param_groups[0]['lr']))
                    running_loss = 0.
                    item_num = 0
            acc = self.eval_model()
            print("epoch %d:  ACC: %.3f"%
                  (epoch,   acc))
            if self.best_acc<=acc:
                torch.save(self.model.state_dict(),self.save_path)
                print("saving model....")
                self.best_acc = acc
            self.scheduler.step()