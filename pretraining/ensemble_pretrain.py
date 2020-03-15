'''
    ensmeble pretrain with penalty (penalty is try to magnify the gradients of input image by different self-supervised task)
'''

import argparse
import os
import pdb
import time
import torch
import random
import pickle
import numpy as np  
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import PIL.Image as Image
import matplotlib.pyplot as plt

import model_ensemble
from attention_pooling.attention_pooling import SelfieModel
from attack_algo_ensemble import PGD_attack_3module,PGD_attack_jigsaw,PGD_attack_rotation,PGD_attack_selfie

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('--data', type=str, default='/datadrive/data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=80, help='batch size')
parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=150, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=10, type=int, help='print frequency')

#parameters need input
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='/datadrive/models/pretrain_penalty_ensemble_onegpu', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=22, help='random seed')
parser.add_argument('--local_rank', type=int, help='local-rank')
parser.add_argument('--attack_eps', type=float, default=(8/255), help='perturbation radius for attack phase')
parser.add_argument('--attack_gamma', type=float, default=(2/255), help='perturbation radius for attack phase')
parser.add_argument('--attack_randinit', type=bool, default=True, help="randinit flag for attack algo")
parser.add_argument('--adv_iter', type=int, default=10,  help='how many epochs to wait before another test')

best_prec1 = 0
best_ata = 0

def main():
    global args, best_prec1, best_ata
    args = parser.parse_args()
    print(args)

    torch.cuda.set_device(args.gpu)

    setup_seed(args.seed)

    model = model_ensemble.PretrainEnsembleModel()
    model = model.cuda()

    n_split = 4
    selfie_model = get_selfie_model(n_split)
    selfie_model = selfie_model.cuda()

    cudnn.benchmark = True

    train_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()
        ])

    val_trans = transforms.Compose([
            transforms.ToTensor()
        ])

    #dataset process
    train_dataset = datasets.CIFAR10(args.data, train=True, transform=train_trans, download=True)
    test_dataset = datasets.CIFAR10(args.data, train=False, transform=val_trans, download=True)

    valid_size = 0.1
    indices = list(range(len(train_dataset)))
    split = int(np.floor(valid_size*len(train_dataset)))
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.Subset(train_dataset, train_idx)
    valid_sampler = torch.utils.data.Subset(train_dataset, valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_sampler,
        batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valid_sampler,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True)


    criterion = nn.CrossEntropyLoss().cuda()


    params_list = [{'params': selfie_model.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]

    params_list.append({'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay, nesterov = True)

                                
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  
            1e-8 / args.lr))

    print('adv training')

    alltrain_acc_jig=[]
    allta_jig=[]
    allata_jig=[]

    alltrain_acc_rot=[]
    allta_rot=[]
    allata_rot=[]

    alltrain_acc_selfie=[]
    allta_selfie=[]
    allata_selfie=[]

    if os.path.exists(args.save_dir) is not True:
        os.mkdir(args.save_dir)

    permutation = np.array([np.random.permutation(16) for i in range(30)]) 
    np.save(os.path.join(args.save_dir, 'permutation.npy'), permutation)
    if permutation.min()==1:
        permutation=permutation-1

    all_seq=[np.random.permutation(16) for ind in range(400)]
    pickle.dump(all_seq, open(os.path.join(args.save_dir, 'img_test_seq.pkl'),'wb'))
    # all_seq=pickle.load(open('img_test_seq.pkl','rb'))
 
    for epoch in range(args.epochs):
        print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))
        # train for one epoch
        train_acc_jig, train_acc_rot, train_acc_selfie, _ = train(train_loader, model, selfie_model, criterion, optimizer, epoch, scheduler, permutation)


        # evaluate on validation set
        ta_jig, ta_rot, ta_selfie , _ = val(val_loader, model, selfie_model, criterion, permutation, all_seq)
        ata_jig, ata_rot, ata_selfie , _ = val_pgd(val_loader, model, selfie_model, criterion, permutation, all_seq)

        alltrain_acc_selfie.append(train_acc_selfie)
        allta_selfie.append(ta_selfie)
        allata_selfie.append(ata_selfie)

        alltrain_acc_jig.append(train_acc_jig)
        allta_jig.append(ta_jig)
        allata_jig.append(ata_jig)

        alltrain_acc_rot.append(train_acc_rot)
        allta_rot.append(ta_rot)
        allata_rot.append(ata_rot)

        sum_ta=ta_jig+ta_rot+ta_selfie
        sum_ata=ata_jig+ata_rot+ata_selfie


        # remember best prec@1 and save checkpoint
        is_best = sum_ta  > best_prec1
        best_prec1 = max(sum_ta, best_prec1)

        ata_is_best = sum_ata > best_ata
        best_ata = max(sum_ata,best_ata)
        
        if is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'selfie_state': selfie_model.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pt'))

        if ata_is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'selfie_state': selfie_model.state_dict(),
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'ata_best_model.pt'))

        save_checkpoint({
            'epoch': epoch + 1,
            'selfie_state': selfie_model.state_dict(),
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.pt'))
    


        plt.plot(alltrain_acc_selfie, label='train_acc')
        plt.plot(allta_selfie, label='TA')
        plt.plot(allata_selfie, label='ATA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'selfie.png'))
        plt.close()

        plt.plot(alltrain_acc_rot, label='train_acc')
        plt.plot(allta_rot, label='TA')
        plt.plot(allata_rot, label='ATA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'rotation.png'))
        plt.close()

        plt.plot(alltrain_acc_jig, label='train_acc')
        plt.plot(allta_jig, label='TA')
        plt.plot(allata_jig, label='ATA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'jigsaw.png'))
        plt.close()

    print('start testing ATA best model')
    model_path = os.path.join(args.save_dir, 'ata_best_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    selfie_model.load_state_dict(model_dict_best['selfie_state'])
    ta_jig, ta_rot, ta_selfie , _ = val(val_loader, model, selfie_model, criterion, permutation, all_seq)
    ata_jig, ata_rot, ata_selfie , _ = val_pgd(val_loader, model, selfie_model, criterion, permutation, all_seq)

    print('start testing TA best model')
    model_path = os.path.join(args.save_dir, 'best_model.pt')
    model_dict_best = torch.load(model_path)
    model.load_state_dict(model_dict_best['state_dict'])
    selfie_model.load_state_dict(model_dict_best['selfie_state'])
    ta_jig, ta_rot, ta_selfie , _ = val(val_loader, model, selfie_model, criterion, permutation, all_seq)
    ata_jig, ata_rot, ata_selfie , _ = val_pgd(val_loader, model, selfie_model, criterion, permutation, all_seq)


def train(train_loader, model, selfie_model, criterion, optimizer, epoch, scheduler, permutation):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_selfie = AverageMeter()
    top1_rotation = AverageMeter()
    top1_jigsaw = AverageMeter()

    end = time.time()


    # # switch to train mode
    model.train()
    selfie_model.train()

    bias=0.9

    for index, (input, label) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.cuda()

        cur_batch_size=input.size(0)

        jig_input,jig_target = jigsaw(input, permutation, bias)

        jig_target=jig_target.long().cuda()
        jig_input = jig_input.cuda()

        #rotation process
        rot_input, rot_target = rotation(input)

        #selfie forward 
        total=16
        seq = np.random.permutation(total)
        t = seq[:(total // 4)]
        v = seq[(total // 4):]
        v = torch.from_numpy(v).cuda()
        pos = t

        t = torch.from_numpy(np.array(pos)).cuda()

        input_adv, jig_input_adv, rot_input_adv = PGD_attack_3module(
            x = [input, jig_input, rot_input],
            y_jig = jig_target,
            y_rot = rot_target,
            selfie_model = selfie_model,
            model = model,
            criterion = criterion,
            seq = seq,
            eps = (8/255),
            steps = 10,
            gamma = (2/255),
            randinit = True 
        )
        
        input_adv = input_adv.requires_grad_(True).cuda()
        jig_input_adv = jig_input_adv.requires_grad_(True).cuda()
        rot_input_adv = rot_input_adv.requires_grad_(True).cuda()

        #jigsaw forward
        jig_feature = model(jig_input)
        jig_feature = model.layer4_jigsaw(jig_feature)
        jig_feature = model.avgpool3(jig_feature)
        jig_feature = jig_feature.view(jig_feature.size(0), -1)
        jig_out = model.fc_jigsaw(jig_feature)
        loss_jigsaw = criterion(jig_out,jig_target)

        #adv jigsaw forward
        jig_feature_adv = model(jig_input_adv)
        jig_feature_adv = model.layer4_jigsaw(jig_feature_adv)
        jig_feature_adv = model.avgpool3(jig_feature_adv)
        jig_feature_adv = jig_feature_adv.view(jig_feature_adv.size(0), -1)
        jig_out_adv = model.fc_jigsaw(jig_feature_adv)
        loss_jigsaw_adv = criterion(jig_out_adv,jig_target)

        #adv rotation forward
        rot_feature_adv = model(rot_input_adv)
        rot_feature_adv = model.layer4_rotation(rot_feature_adv)
        rot_feature_adv = model.avgpool2(rot_feature_adv)
        rot_feature_adv = rot_feature_adv.view(rot_feature_adv.size(0), -1)
        rot_out_adv = model.fc_rotation(rot_feature_adv)
        loss_rotation_adv = criterion(rot_out_adv,rot_target)

        # rotation forward
        rot_feature = model(rot_input)
        rot_feature = model.layer4_rotation(rot_feature)
        rot_feature = model.avgpool2(rot_feature)
        rot_feature = rot_feature.view(rot_feature.size(0), -1)
        rot_out = model.fc_rotation(rot_feature)
        loss_rotation = criterion(rot_out,rot_target)


        
        #selfie forward
        batches = split_image_selfie(input, 8)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = model(input_batches)
        output_batches = model.avgpool1(output_batches)
        output_batches = output_batches.view(output_batches.size(0),-1)

        output_batches = output_batches.unsqueeze(1)
        output_batches = torch.split(output_batches, cur_batch_size, 0)
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

        #adv selfie forward
        batches_adv = split_image_selfie(input_adv, 8)

        batches_adv = list(map(lambda x: x.unsqueeze(1), batches_adv))
        batches_adv = torch.cat(batches_adv, 1) # (B, L, C, H, W)

        input_batches_adv = torch.split(batches_adv, 1, 1)
        input_batches_adv = list(map(lambda x: x.squeeze(1), input_batches_adv))
        input_batches_adv = torch.cat(input_batches_adv, 0)

        output_batches_adv = model(input_batches_adv)
        output_batches_adv = model.avgpool1(output_batches_adv)
        output_batches_adv = output_batches_adv.view(output_batches_adv.size(0),-1)
        output_batches_adv = output_batches_adv.unsqueeze(1)
        output_batches_adv = torch.split(output_batches_adv, cur_batch_size, 0)
        output_batches_adv = torch.cat(output_batches_adv,1)

        output_decoder_adv = output_batches_adv.index_select(1, t)
        
        output_encoder_adv = output_batches_adv.index_select(1, v)
        output_encoder_adv = selfie_model(output_encoder_adv, pos)

        features_adv = []
        for i in range(len(pos)):
            feature_adv = output_decoder_adv[:, i, :]
            feature_adv = feature_adv.unsqueeze(2)
            features_adv.append(feature_adv)

        features_adv = torch.cat(features_adv, 2) # (B, F, NP)
        patch_loss_adv = 0

        for i in range(len(t)):
            activate_adv = output_encoder_adv[:, i, :].unsqueeze(1)
            pre_adv = torch.bmm(activate_adv, features_adv)
            logit_adv = nn.functional.softmax(pre_adv, 2).view(-1, len(t))
            temptarget_adv = torch.ones(logit_adv.shape[0]).cuda() * i
            temptarget_adv = temptarget_adv.long()
            loss__adv = criterion(logit_adv, temptarget_adv)

            patch_loss_adv += loss__adv

            prec1_adv, _ = accuracy(logit_adv, temptarget_adv, topk=(1,3))
            top1_selfie.update(prec1_adv.item(), 1)


        grad_selfie = torch.autograd.grad(patch_loss_adv, input_adv, create_graph=True)[0]
        grad_jig = torch.autograd.grad(loss_jigsaw_adv, jig_input_adv, create_graph=True)[0]
        grad_rot = torch.autograd.grad(loss_rotation_adv, rot_input_adv, create_graph=True)[0]
        
        h_loss = calculate_log_det(grad_selfie, grad_jig, grad_rot)

        all_loss = (patch_loss+loss_jigsaw+loss_rotation+patch_loss_adv+loss_jigsaw_adv+loss_rotation_adv)/6 + 0.1*h_loss

        optimizer.zero_grad()
        all_loss.backward()
        optimizer.step()

        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        jig_out_adv = jig_out_adv.float()
        rot_out_adv = rot_out_adv.float()

        all_loss = all_loss.float()

        # measure accuracy and record loss
        prec1_jig = accuracy(jig_out_adv.data, jig_target)[0]
        prec1_rot = accuracy(rot_out_adv.data, rot_target)[0]

        losses.update(all_loss.item(), input.size(0))
        top1_jigsaw.update(prec1_jig.item(), input.size(0))
        top1_rotation.update(prec1_rot.item(), input.size(0))

        if index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_selfie {atop1.val:.3f} ({atop1.avg:.3f})\t'
                  'Acc_jig {btop1.val:.3f} ({btop1.avg:.3f})\t'
                  'Acc_rot {ctop1.val:.3f} ({ctop1.avg:.3f})'.format(
                      epoch, index, len(train_loader),batch_time=batch_time, loss=losses, atop1=top1_selfie, btop1=top1_jigsaw, ctop1=top1_rotation))


    print('train_jigsaw_accuracy {top1.avg:.3f}'.format(top1=top1_jigsaw))
    print('train_rotation_accuracy {top1.avg:.3f}'.format(top1=top1_rotation))
    print('train_selfie_accuracy {top1.avg:.3f}'.format(top1=top1_selfie))

    return top1_jigsaw.avg, top1_rotation.avg, top1_selfie.avg, losses.avg

def val(val_loader, model, selfie_model, criterion, permutation, all_seq):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_selfie = AverageMeter()
    top1_rotation = AverageMeter()
    top1_jigsaw = AverageMeter()
    end = time.time()
    model.eval()
    selfie_model.eval()

    bias=0.9

    for index, (input, label) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda()

        cur_batch_size=input.size(0)

        # jigsaw process 
        jig_input,jig_target = jigsaw(input, permutation, bias)

        jig_target = jig_target.long().cuda()
        jig_input = jig_input.cuda()

        
        #rotation process
        rot_input, rot_target = rotation(input)

        
        #selfie forward 
        total=16
        seq = all_seq[index]
        t = seq[:(total // 4)]
        v = seq[(total // 4):]
        v = torch.from_numpy(v).cuda()
        pos = t

        t = torch.from_numpy(np.array(pos)).cuda()

        #jigsaw forward
        jig_feature = model(jig_input)
        jig_feature = model.layer4_jigsaw(jig_feature)
        jig_feature = model.avgpool3(jig_feature)
        jig_feature = jig_feature.view(jig_feature.size(0), -1)
        jig_out = model.fc_jigsaw(jig_feature)
        loss_jigsaw = criterion(jig_out,jig_target)

        #rotation forward
        rot_feature = model(rot_input)
        rot_feature = model.layer4_rotation(rot_feature)
        rot_feature = model.avgpool2(rot_feature)
        rot_feature = rot_feature.view(rot_feature.size(0), -1)
        rot_out = model.fc_rotation(rot_feature)
        loss_rotation = criterion(rot_out,rot_target)


        
        #selfie forward
        batches = split_image_selfie(input, 8)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = model(input_batches)
        output_batches = model.avgpool1(output_batches)
        output_batches = output_batches.view(output_batches.size(0),-1)

        output_batches = output_batches.unsqueeze(1)
        output_batches = torch.split(output_batches, cur_batch_size, 0)
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

            prec1, _ = accuracy(logit, temptarget, topk=(1,3))

            top1_selfie.update(prec1.item(), 1)


        all_loss = patch_loss+loss_jigsaw+loss_rotation


        jig_out = jig_out.float()
        rot_out = rot_out.float()

        all_loss = all_loss.float()

        # measure accuracy and record loss
        prec1_jig = accuracy(jig_out.data, jig_target)[0]
        prec1_rot = accuracy(rot_out.data, rot_target)[0]

        losses.update(all_loss.item(), input.size(0))
        top1_jigsaw.update(prec1_jig.item(), input.size(0))
        top1_rotation.update(prec1_rot.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_selfie {atop1.val:.3f} ({atop1.avg:.3f})\t'
                  'Acc_jig {btop1.val:.3f} ({btop1.avg:.3f})\t'
                  'Acc_rot {ctop1.val:.3f} ({ctop1.avg:.3f})'.format(
                    index, len(val_loader), batch_time=batch_time, loss=losses, atop1=top1_selfie, btop1=top1_jigsaw, ctop1=top1_rotation))

    print('val_jigsaw_accuracy {top1.avg:.3f}'.format(top1=top1_jigsaw))
    print('val_rotation_accuracy {top1.avg:.3f}'.format(top1=top1_rotation))
    print('val_selfie_accuracy {top1.avg:.3f}'.format(top1=top1_selfie))

    return top1_jigsaw.avg, top1_rotation.avg, top1_selfie.avg, losses.avg  

def val_pgd(val_loader, model, selfie_model, criterion, permutation, all_seq):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1_selfie = AverageMeter()
    top1_rotation = AverageMeter()
    top1_jigsaw = AverageMeter()
    end = time.time()
    model.eval()
    selfie_model.eval()
    bias=0.9

    for index, (input, label) in enumerate(val_loader):
        data_time.update(time.time() - end)
        input = input.cuda()

        cur_batch_size=input.size(0)

        # jigsaw process 
        jig_input,jig_target = jigsaw(input, permutation, bias)

        jig_target=jig_target.long().cuda()
        jig_input = jig_input.cuda()

        jig_input_adv = PGD_attack_jigsaw(jig_input,jig_target,
            model=model,
            criterion=criterion,
            eps=args.attack_eps,
            steps=20,
            gamma=args.attack_gamma,
            randinit=args.attack_randinit).data
        with torch.no_grad():
            jig_input=jig_input_adv.cuda()

        #rotation process
        rot_input, rot_target = rotation(input)

        rot_input_adv = PGD_attack_rotation(rot_input,rot_target,
            model=model,
            criterion=criterion,
            eps=args.attack_eps,
            steps=20,
            gamma=args.attack_gamma,
            randinit=args.attack_randinit).data
        with torch.no_grad():
            rot_input=rot_input_adv.cuda()

        #selfie forward 
        total=16
        seq = all_seq[index]
        t = seq[:(total // 4)]
        v = seq[(total // 4):]
        v = torch.from_numpy(v).cuda()
        pos = t

        t = torch.from_numpy(np.array(pos)).cuda()

        input_adv = PGD_attack_selfie(input,
            selfie_model=selfie_model,
            P=model,
            criterion=criterion,
            seq=seq,
            eps=(8/255),
            steps=20,
            gamma=(2/255),
            randinit=True).data
        with torch.no_grad():
            input=input_adv.cuda()

        #jigsaw forward
        jig_feature = model(jig_input)
        jig_feature = model.layer4_jigsaw(jig_feature)
        jig_feature = model.avgpool3(jig_feature)
        jig_feature = jig_feature.view(jig_feature.size(0), -1)
        jig_out = model.fc_jigsaw(jig_feature)
        loss_jigsaw = criterion(jig_out,jig_target)

        #rotation forward
        rot_feature = model(rot_input)
        rot_feature = model.layer4_rotation(rot_feature)
        rot_feature = model.avgpool2(rot_feature)
        rot_feature = rot_feature.view(rot_feature.size(0), -1)
        rot_out = model.fc_rotation(rot_feature)
        loss_rotation = criterion(rot_out,rot_target)


        
        #selfie forward
        batches = split_image_selfie(input, 8)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = model(input_batches)
        output_batches = model.avgpool1(output_batches)
        output_batches = output_batches.view(output_batches.size(0),-1)

        output_batches = output_batches.unsqueeze(1)
        output_batches = torch.split(output_batches, cur_batch_size, 0)
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

            prec1, _ = accuracy(logit, temptarget, topk=(1,3))

            top1_selfie.update(prec1.item(), 1)


        all_loss = patch_loss+loss_jigsaw+loss_rotation


        jig_out = jig_out.float()
        rot_out = rot_out.float()

        all_loss = all_loss.float()

        # measure accuracy and record loss
        prec1_jig = accuracy(jig_out.data, jig_target)[0]
        prec1_rot = accuracy(rot_out.data, rot_target)[0]

        losses.update(all_loss.item(), input.size(0))
        top1_jigsaw.update(prec1_jig.item(), input.size(0))
        top1_rotation.update(prec1_rot.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc_selfie {atop1.val:.3f} ({atop1.avg:.3f})\t'
                  'Acc_jig {btop1.val:.3f} ({btop1.avg:.3f})\t'
                  'Acc_rot {ctop1.val:.3f} ({ctop1.avg:.3f})'.format(
                    index, len(val_loader), batch_time=batch_time, loss=losses, atop1=top1_selfie, btop1=top1_jigsaw, ctop1=top1_rotation))

    print('adv_jigsaw_accuracy {top1.avg:.3f}'.format(top1=top1_jigsaw))
    print('adv_rotation_accuracy {top1.avg:.3f}'.format(top1=top1_rotation))
    print('adv_selfie_accuracy {top1.avg:.3f}'.format(top1=top1_selfie))

    return top1_jigsaw.avg, top1_rotation.avg, top1_selfie.avg, losses.avg
    
def save_checkpoint(state, is_best, filename='weight.pt'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def get_selfie_model(n_split):
    n_layers = 12
    d_model = 1024 #vector length after the patch routed in P
    d_in = 64
    n_heads = d_model// d_in
    d_ff = 2048
    model = SelfieModel(n_layers, n_heads, d_in, d_model, d_ff, n_split)
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def split_image(image, N=8):
    """
    image: (C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=1)):
        batches.extend(list(torch.split(i, N, dim=2)))

    return batches

def split_image_selfie(image, N):
    """
    image: (B, C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=2)):
        batches.extend(list(torch.split(i, N, dim=3)))

    return batches

def concat(batches, num=4):
    """
    batches: [(C,W1,H1)]
    """

    batches_list=[]

    for j in range(len(batches) // num):
        batches_list.append(torch.cat(batches[num*j:num*(j+1)], dim=2))
    return torch.cat(batches_list,dim=1)

def jigsaw(input, permutation, bias):

    cur_batch_size=input.size(0)

    jig_input=torch.zeros_like(input)
    jig_target=torch.zeros(cur_batch_size)

    for idx in range(cur_batch_size):
        img=input[idx,:]
        batches=split_image(img)
        order=np.random.randint(len(permutation)+1)
    
        rate=np.random.rand()
        if rate>bias:
            order=0

        if order == 0:
            new_batches=batches
        else:
            new_batches=[batches[permutation[order-1][j]] for j in range(16)]

        jig_input[idx]=concat(new_batches)
        jig_target[idx]=order

    return jig_input, jig_target

def rotation(input):
    batch = input.shape[0]
    target = torch.tensor(np.random.permutation([0,1,2,3] * (int(batch / 4) + 1)), device = input.device)[:batch]
    target = target.long()
    image = torch.zeros_like(input)
    image.copy_(input)
    for i in range(batch):
        image[i, :, :, :] = torch.rot90(input[i, :, :, :], target[i], [1, 2])

    return image, target

def compute_cosine(x,x1,x2,x3):

    g1=torch.flatten(x1-x, 1)
    g2=torch.flatten(x2-x, 1)
    g3=torch.flatten(x3-x, 1)

    cos12=F.cosine_similarity(g1,g2,dim=1)
    cos13=F.cosine_similarity(g1,g3,dim=1)
    cos23=F.cosine_similarity(g2,g3,dim=1)

    return cos12,cos13,cos23

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

def warmup_lr(step, optimizer, speed):
    lr = 0.01+step*(0.1-0.01)/speed
    lr = min(lr,0.1)
    for p in optimizer.param_groups:
        p['lr']=lr

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 


if __name__ == '__main__':
    main()