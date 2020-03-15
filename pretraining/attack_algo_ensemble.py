import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pdb

def split_image_selfie(image, N):
    """
    image: (B, C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=2)):
        batches.extend(list(torch.split(i, N, dim=3)))

    return batches

def concat_selfie(batches,num):
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

def calculate_log_det(grad1,grad2,grad3):

    grad1 = grad1.view(grad1.size(0),-1)
    grad2 = grad2.view(grad2.size(0),-1)
    grad3 = grad3.view(grad3.size(0),-1)

    I=[1e-5*torch.eye(3).unsqueeze(0) for i in range(grad1.size(0))]
    I=torch.cat(I,0).cuda()

    norm1 = torch.norm(grad1, p=2, dim=1, keepdim=True)+1e-10
    norm2 = torch.norm(grad2, p=2, dim=1, keepdim=True)+1e-10
    norm3 = torch.norm(grad3, p=2, dim=1, keepdim=True)+1e-10


    grad1 = grad1.div(norm1).unsqueeze(2)
    grad2 = grad2.div(norm2).unsqueeze(2)
    grad3 = grad3.div(norm3).unsqueeze(2)

    G = torch.cat([grad1,grad2,grad3],2)

    det = torch.bmm(torch.transpose(G, 1, 2), G)+I

    logdet = det.det().log().mean()

    return logdet

def PGD_attack_3module(x, y_jig, y_rot, selfie_model, model, criterion, seq, eps=(8/255), steps=3, gamma=(2/255), randinit=True):
    
    #input list of three input image 
    lambda_h=0.1

    # Compute loss
    x_adv1 = x[0].clone()
    x_adv2 = x[1].clone()
    x_adv3 = x[2].clone()

    if randinit:
        x_adv1 += (2.0 * torch.rand(x_adv1.shape).cuda() - 1.0) * eps
        x_adv2 += (2.0 * torch.rand(x_adv2.shape).cuda() - 1.0) * eps
        x_adv3 += (2.0 * torch.rand(x_adv3.shape).cuda() - 1.0) * eps

    x_adv1=x_adv1.requires_grad_(True).cuda()
    x_adv2=x_adv2.requires_grad_(True).cuda()
    x_adv3=x_adv3.requires_grad_(True).cuda()

    x1 = x[0].cuda()
    x2 = x[1].cuda()
    x3 = x[2].cuda()

    num_split=8
    total=16
    batch_size=x1.size(0)

    t = seq[:(total // 4)]
    v = seq[(total // 4):]
    v = torch.from_numpy(v).cuda()
    pos = t

    t = torch.from_numpy(np.array(pos)).cuda()
    
    for iteration in range(steps):

        #selfie forward
        batches = split_image_selfie(x_adv1, num_split)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = model(input_batches)
        output_batches = model.avgpool1(output_batches)
        output_batches = output_batches.view(output_batches.size(0),-1)
        
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


        #jigsaw forward 
        jig_pre = model(x_adv2)
        jig_pre = model.layer4_jigsaw(jig_pre)
        jig_pre = model.avgpool3(jig_pre)
        jig_pre = jig_pre.view(jig_pre.size(0), -1)
        jig_out = model.fc_jigsaw(jig_pre)
        loss_jigsaw_adv = criterion(jig_out, y_jig)

        #rotation forward 
        rot_pre = model(x_adv3)
        rot_pre = model.layer4_rotation(rot_pre)
        rot_pre = model.avgpool2(rot_pre)
        rot_pre = rot_pre.view(rot_pre.size(0), -1)
        rot_out = model.fc_rotation(rot_pre)
        loss_rot_adv = criterion(rot_out, y_rot)
   
        grad_selfie = torch.autograd.grad(patch_loss, x_adv1, create_graph=True)[0]
        grad_jig = torch.autograd.grad(loss_jigsaw_adv, x_adv2, create_graph=True)[0]
        grad_rot = torch.autograd.grad(loss_rot_adv, x_adv3, create_graph=True)[0]

        #compute log_det of three gradient 

        h_loss = calculate_log_det(grad_selfie, grad_jig, grad_rot)

        # compute h_loss gradient
        h_grad_selfie = torch.autograd.grad(h_loss, x_adv1, retain_graph=True)[0]
        h_grad_jig = torch.autograd.grad(h_loss, x_adv2, retain_graph=True)[0]
        h_grad_rot = torch.autograd.grad(h_loss, x_adv3, retain_graph=False)[0]


        all_grad_selfie = grad_selfie + lambda_h*h_grad_selfie
        all_grad_jig = grad_jig + lambda_h*h_grad_jig
        all_grad_rot = grad_rot + lambda_h*h_grad_rot


        x_adv1.data.add_(gamma * torch.sign(all_grad_selfie.data))
        x_adv2.data.add_(gamma * torch.sign(all_grad_jig.data))
        x_adv3.data.add_(gamma * torch.sign(all_grad_rot.data))

        linfball_proj(x1, eps, x_adv1, in_place=True)
        linfball_proj(x2, eps, x_adv2, in_place=True)
        linfball_proj(x3, eps, x_adv3, in_place=True)

        x_adv1 = torch.clamp(x_adv1, 0, 1)
        x_adv2 = torch.clamp(x_adv2, 0, 1)
        x_adv3 = torch.clamp(x_adv3, 0, 1)


    return x_adv1.data, x_adv2.data, x_adv3.data

def PGD_attack_selfie(x, selfie_model, P, criterion, seq, eps=(8/255), steps=3, gamma=(2/255), randinit=True):
    
    # Compute loss
    x_adv = x.clone()
    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps

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


        batches = split_image_selfie(x_adv, num_split)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = P(input_batches)
        output_batches = P.avgpool1(output_batches)
        output_batches = output_batches.view(output_batches.size(0),-1)

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

        grad0 = torch.autograd.grad(loss_adv, batches, only_inputs=True)[0]

        batches.data.add_(gamma * torch.sign(grad0.data))

        x_adv=concat_selfie(batches,4)

        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.data

def PGD_attack_rotation(x, y, model, criterion, eps=(8/255), steps=3, gamma=(2/255), randinit=True):

    x_adv = x.clone()

    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps

    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()
    y = y.cuda()

    for t in range(steps):
        #rotation

        adv_feature = model(x_adv)
        adv_feature = model.layer4_rotation(adv_feature)
        adv_feature = model.avgpool2(adv_feature)
        adv_feature = adv_feature.view(adv_feature.size(0),-1)
        output = model.fc_rotation(adv_feature)

        loss_rotation = criterion(output,y)
        grad0 = torch.autograd.grad(loss_rotation, x_adv, only_inputs=True)[0]

        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv

def PGD_attack_jigsaw(x, y, model, criterion, eps=(8/255), steps=3, gamma=(2/255), randinit=True):

    x_adv = x.clone()

    if randinit:
        x_adv += (2.0 * torch.rand(x_adv.shape).cuda() - 1.0) * eps

    x_adv = Variable(x_adv.cuda(), requires_grad=True)
    x = x.cuda()


    for t in range(steps):
        pre = model(x_adv)
        pre = model.layer4_jigsaw(pre)
        pre = model.avgpool3(pre)
        pre = pre.view(pre.size(0), -1)
        loss_adv0 = criterion(model.fc_jigsaw(pre), y)
        grad0 = torch.autograd.grad(loss_adv0, x_adv, only_inputs=True)[0]

        x_adv.data.add_(gamma * torch.sign(grad0.data))
        linfball_proj(x, eps, x_adv, in_place=True)
        x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv


