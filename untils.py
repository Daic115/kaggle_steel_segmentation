#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File           :   untils.py
@Desciption     :   None
@Modify Time      @Author    @Version 
------------      -------    --------  
2019/9/26 13:36   Daic       1.0        
'''
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torch.optim as optim
from torch.optim import Optimizer
import segmentation_models_pytorch as smp
import pretrainedmodels

class RAdam(Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.buffer = [[None, None, None] for ind in range(10)]
        super(RAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)
                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                buffered = self.buffer[int(state['step'] % 10)]
                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = group['lr'] * math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = group['lr'] / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                if group['weight_decay'] != 0:
                    p_data_fp32.add_(-group['weight_decay'] * group['lr'], p_data_fp32)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p_data_fp32.addcdiv_(-step_size, exp_avg, denom)
                else:
                    p_data_fp32.add_(-step_size, exp_avg)

                p.data.copy_(p_data_fp32)

        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,long)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''
    return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return mish(input)


def build_optimizer(params, opt):
    if opt.optimizer == 'rmsprop':
        return optim.RMSprop(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'radam':
        return RAdam(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adagrad':
        return optim.Adagrad(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgd':
        return optim.SGD(params, opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdm':
        return optim.SGD(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'sgdmom':
        return optim.SGD(params, opt.lr, opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    elif opt.optimizer == 'adam':
        return optim.Adam(params, opt.lr,  weight_decay=opt.weight_decay)
    else:
        raise Exception("bad option opt.optim: {}".format(opt.optim))

def build_loss_function(opt):
    # Jaccard #Dice #BCEJaccard #BCEDice
    if opt.loss == 'Jaccard':
        return smp.utils.losses.JaccardLoss()
    elif opt.loss == 'Dice':
        return smp.utils.losses.DiceLoss()
    elif opt.loss == 'BCEJaccard':
        return smp.utils.losses.BCEJaccardLoss()
    elif opt.loss == 'BCEDice':
        return smp.utils.losses.BCEDiceLoss()
    elif opt.loss == 'Focal':
        return  FocalLoss()
    else:
        raise Exception("bad option opt.loss: {}".format(opt.loss))

def convert_relu_to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            convert_relu_to_Mish(child)

def build_cls_model(name,active = 'relu'):
    model = pretrainedmodels.__dict__[name](num_classes=1000, pretrained='imagenet')
    if name in ['resnet18','resnet32']:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.last_linear = nn.Linear(in_features=512, out_features=1, bias=True)
    elif name in ['resnet50','resnet101','resnet152']:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
    elif name in ['senet154',  'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d']:
        model.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Linear(in_features=2048, out_features=1, bias=True)
    else:
        raise Exception("unsupported model! {}".format(name))

    if active == 'relu':
        return model
    elif active == 'mish':
        convert_relu_to_Mish(model)
        return model

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def resave_model(src,savep):
    m = torch.load(src)
    torch.save(m.state_dict(),savep)



