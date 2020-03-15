import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

from utils import *

def concat(batches,num):
    list=[]
    batches_list=[]
    for idx in range(batches.shape[1]):
        batches_list.append(batches[:,idx,:,:,:])
    for j in range(len(batches_list) // num):
        list.append(torch.cat(batches_list[num*j:num*(j+1)], dim=3))
    return torch.cat(list,dim=2)

def tensor_clamp(t, min, max, in_place=True):
    if not in_place:
        res = t.clone()
    else:
        res = t
    idx = res.data < min
    res.data[idx] = min[idx]
    idx = res.data > max
    res.data[idx] = max[idx]

    return res

def linfball_proj(center, radius, t, in_place=True):
    return tensor_clamp(t, min=center - radius, max=center + radius, in_place=in_place)

def PGD_attack(x, selfie_model, P, criterion, seq, eps=(8/255), steps=3, gamma=(2/255), randinit=True):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape) - 1.0) * eps

    x_adv=x_adv.requires_grad_(True).cuda()
    x = x.cuda()

    num_split=8
    total=16
    batch_size=x.size(0)

    t = seq[:(total // 4)]
    v = seq[(total // 4):]
    v = torch.from_numpy(v).cuda()
    pos = t

    t = torch.from_numpy(np.array(pos)).cuda()
    
    for iteration in range(steps):


        batches = split_image(x_adv, num_split)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = P(input_batches)
        output_batches = output_batches.unsqueeze(1)
        output_batches = torch.split(output_batches, batch_size, 0)
        output_batches = torch.cat(output_batches,1)

        output_decoder = output_batches.index_select(1, t)
        
        output_encoder = output_batches.index_select(1, v)
        output_encoder = selfie_model(output_encoder, pos)

        features = []
        for i in range(len(pos)):
            feature = output_decoder[:, i, :]
            feature = feature.unsqueeze(2)
            features.append(feature)

        features = torch.cat(features, 2) # (B, F, NP)
        patch_loss = 0

        for i in range(len(t)):
            activate = output_encoder[:, i, :].unsqueeze(1)
            pre = torch.bmm(activate, features)
            logit = nn.functional.softmax(pre, 2).view(-1, len(t))
            temptarget = torch.ones(logit.shape[0]).cuda() * i
            temptarget = temptarget.long()
            loss_ = criterion(logit, temptarget)

            patch_loss += loss_
        
        loss_adv=patch_loss

        grad0 = torch.autograd.grad(loss_adv, x_adv, only_inputs=True)[0]

        x_adv.data.add_(gamma * torch.sign(grad0.data))

        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)
    
    return x_adv.data

def PGD_normal(x, loss_fn, y=None, eps=None, model=None, steps=3, gamma=None, randinit=False):

    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps
    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()


    for t in range(steps):
        out = model(x_adv)
        loss_adv0 = loss_fn(out, y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]
        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv 
        
