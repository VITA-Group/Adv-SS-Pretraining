import argparse
import os
import sys
import random
import shutil
import time
import warnings
import numpy as np
import pickle
import inspect

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torchvision import transforms
from torchvision import datasets
from dataset import CifarDataset, ImageNetDataset
from resnetv2 import *
from utils import *
from functions import *
from attention_pooling.attention_pooling import SelfieModel

from attack_algo import PGD_attack
from advertorch.utils import NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(description='Selfie')
  
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    help='model architecture: ')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of steps of selfie')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-method', default='step', type=str,
                    help='method of learning rate')
parser.add_argument('--lr-params', default=[], dest='lr_params',nargs='*',type=float,
                    action='append', help='params of lr method')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--data',default="../../data/",
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default="cifar") 
parser.add_argument('--modeldir', default="imagenet_adv_selfie", type=str,
                    help='director of checkpoint')
parser.add_argument('--store-model-everyepoch', dest='store_model_everyepoch', action='store_true',
                    help='store checkpoint in every epoch')
parser.add_argument('--percent', type=float,
                    help="Used data percent", default=1.0)
parser.add_argument('--evaluation', action="store_true")
parser.add_argument('--classification-model', type=str, default="")
parser.add_argument('--split-gpu', action="store_true")
parser.add_argument('--resume', action="store_true")
parser.add_argument('--finetune', action="store_true")
parser.add_argument('--evaluation-selfie', action="store_true")
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--seed', type=int, default=10)
best_prec1 = 0
ata_best_prec1 = 0


def main():
    global args, best_prec1, ata_best_prec1

    args = parser.parse_args()
    print(args)
    
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(int(args.gpu))

    setup_seed(args.seed)

    # Data Preprocess
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.RandomCrop(32),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(5),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Pad(2),
            #transforms.RandomCrop(32),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    } 
    if args.dataset == 'cifar':
        train_dataset = CifarDataset(args.data, True, data_transforms['train'], args.percent)
        test_dataset = CifarDataset(args.data, False, data_transforms['val'], 1)
    elif args.dataset == 'imagenet':
        train_dataset = ImageNetDataset(args.data, True, data_transforms['train'], args.percent)
        test_dataset = ImageNetDataset(args.data, False, data_transforms['val'], 1)

    elif args.dataset == 'imagenet224':
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        } 
        train_dataset = datasets.ImageNet(args.data, 'train', True, data_transforms['train'])
        test_dataset = datasets.ImageNet(args.data, 'train', True, data_transforms['val'])

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
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        valid_sampler,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # define model 
    n_split = 4
    selfie_model = get_selfie_model(n_split)
    selfie_model = selfie_model.cuda()

    P=get_P_model()
    normalize = NormalizeByChannelMeanStd(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    P = nn.Sequential(normalize, P)
    P = P.cuda()

    #define optimizer and scheduler 
    params_list = [{'params': selfie_model.parameters(), 'lr': args.lr,
                        'weight_decay': args.weight_decay},]
    params_list.append({'params': P.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay})
    optimizer = torch.optim.SGD(params_list, lr=args.lr,
                                momentum=0.9,
                                weight_decay=args.weight_decay, nesterov = True)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-7 / args.lr))

    print("Training model.")
    step = 0
    if os.path.exists(args.modeldir) is not True:
        os.mkdir(args.modeldir)
    stats_ = stats(args.modeldir, args.start_epoch)

    if args.epochs > 0:

        #order of patches 
        all_seq=[np.random.permutation(16) for ind in range(400)]
        pickle.dump(all_seq, open(os.path.join(args.modeldir, 'img_test_seq.pkl'),'wb'))
        # all_seq=pickle.load(open(os.path.join(args.modeldir, 'img_test_seq.pkl'),'rb'))
      
        print("Begin selfie training...")
        for epoch in range(args.start_epoch, args.epochs):
            print("The learning rate is {}".format(optimizer.param_groups[0]['lr']))
            trainObj, top1 = train_selfie_adv(train_loader, selfie_model, P, criterion, optimizer, epoch, scheduler)

            valObj, prec1 = val_selfie(val_loader, selfie_model, P, criterion, all_seq)

            adv_valObj, adv_prec1 = val_pgd_selfie(val_loader, selfie_model, P, criterion, all_seq)

            stats_._update(trainObj, top1, valObj, prec1,adv_valObj, adv_prec1)

            
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            ata_is_best = adv_prec1 > ata_best_prec1
            ata_best_prec1 = max(adv_prec1, ata_best_prec1)
            if is_best:
                torch.save(
                    {
                    'epoch': epoch,
                    'P_state': P.state_dict(),
                    'selfie_state': selfie_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    }, os.path.join(args.modeldir, 'adv_selfie_TA_model_best.pth.tar'))

            if ata_is_best:
                torch.save(
                    {
                    'epoch': epoch,
                    'P_state': P.state_dict(),
                    'selfie_state': selfie_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    }, os.path.join(args.modeldir, 'adv_selfie_ATA_model_best.pth.tar'))

            torch.save(
                    {
                    'epoch': epoch,
                    'P_state': P.state_dict(),
                    'selfie_state': selfie_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    }, os.path.join(args.modeldir, 'adv_selfie_checkpoint.pth.tar'))

            plot_curve(stats_, args.modeldir, True)
            data = stats_
            sio.savemat(os.path.join(args.modeldir,'stats.mat'), {'data':data})
   

        print("testing ATA best selfie model from checkpoint...")
        model_path = os.path.join(args.modeldir, 'adv_selfie_ATA_model_best.pth.tar')
        model_loaded = torch.load(model_path)

        P.load_state_dict(model_loaded['P_state'])
        selfie_model.load_state_dict(model_loaded['selfie_state'])
        print("Best ATA selfie model loaded! ")

        valObj, prec1 = val_selfie(test_loader, selfie_model, P, criterion, all_seq)
        adv_valObj, adv_prec1 = val_pgd_selfie(test_loader, selfie_model, P, criterion, all_seq)


        print("testing TA best selfie model from checkpoint...")
        model_path = os.path.join(args.modeldir, 'adv_selfie_TA_model_best.pth.tar')
        model_loaded = torch.load(model_path)

        P.load_state_dict(model_loaded['P_state'])
        selfie_model.load_state_dict(model_loaded['selfie_state'])
        print("Best TA selfie model loaded! ")
        
        valObj, prec1 = val_selfie(test_loader, selfie_model, P, criterion, all_seq)
        adv_valObj, adv_prec1 = val_pgd_selfie(test_loader, selfie_model, P, criterion, all_seq)


def train_selfie_adv(train_loader, selfie_model, P, criterion, optimizer, epoch, scheduler):
    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    selfie_model.train()
    P.train()

    for index, (input, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        cur_batch_size = input.size(0)

        # if epoch == 0:
        #     warmup_lr(index, optimizer, 200)
        
        total=16
        seq = np.random.permutation(total)
        t = seq[:(total // 4)]
        v = seq[(total // 4):]
        v = torch.from_numpy(v).cuda()
        pos = t
        t = torch.from_numpy(np.array(pos)).cuda()

        input_adv=PGD_attack(input,
            selfie_model=selfie_model,
            P=P,
            criterion=criterion,
            seq=seq,
            eps=(8/255),
            steps=10,
            gamma=(2/255),
            randinit=True)
            
        input = input.cuda()

        #selfie forward
        batches = split_image_selfie(input, 8)

        batches = list(map(lambda x: x.unsqueeze(1), batches))
        batches = torch.cat(batches, 1) # (B, L, C, H, W)

        input_batches = torch.split(batches, 1, 1)
        input_batches = list(map(lambda x: x.squeeze(1), input_batches))
        input_batches = torch.cat(input_batches, 0)

        output_batches = P(input_batches)

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

        output_batches_adv = P(input_batches_adv)

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
            top1.update(prec1_adv[0], 1)
        
        
        patch_loss=0.5*(patch_loss+patch_loss_adv)

        optimizer.zero_grad()
        patch_loss.backward()
        optimizer.step()
        scheduler.step()

        all_loss = patch_loss.float()
        losses.update(all_loss.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if index % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, index, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    return losses.avg, top1.avg

def val_selfie(val_loader, selfie_model, P, criterion, all_seq):

    '''
    print("Preheating ... ")
    with torch.no_grad():
        h = 0
        while h < 10:
            for index, (input, _) in enumerate(val_loader):
                input = input.cuda(args.gpu, non_blocking = True)
                if input.shape[-1] == 32:
                    num_split = 8
                else:
                    num_split = 32
                batches = split_image(input, num_split)
                batches = list(map(lambda x: x.unsqueeze(1), batches))
                batches = torch.cat(batches, 1)
                total = batches.shape[1]
                seq = np.random.permutation(total)
                t = seq[:int(np.trunc(total / 4.0))]
                v = seq[int(np.trunc(total / 4.0)):]
                v = torch.from_numpy(v).cuda(args.gpu, non_blocking = True)
                pos = t
                t = torch.from_numpy(np.array(pos)).cuda(args.gpu, non_blocking = True)
                input_for_transformer = batches.index_select(1, v)
                shape = input_for_transformer.shape
                output_for_transformer = []
                output_for_decoder = []
                input_for_decoder = batches.index_select(1, t)
                for i in range(shape[1]):
                    d = P(input_for_transformer[:, i, :, :, :]).unsqueeze(1)
                    output_for_transformer.append(d)
                output_for_transformer = torch.cat(output_for_transformer, 1)
                after_position_embeddings = selfie_model(output_for_transformer, pos)
                for i in range(input_for_decoder.shape[1]):
                    output_for_decoder.append(P(input_for_decoder[:, i, :, :, :]).unsqueeze(1)) 
            h += 1
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    selfie_model.eval()
    P.eval()
    with torch.no_grad():
        for index, (input, _) in enumerate(val_loader):
            #print(input)
            data_time.update(time.time() - end)
            input = input.cuda()
            cur_batch_size = input.size(0)

            total=16
            seq = all_seq[index]
            t = seq[:(total // 4)]
            v = seq[(total // 4):]
            v = torch.from_numpy(v).cuda()
            pos = t

            t = torch.from_numpy(np.array(pos)).cuda()

            #selfie forward
            batches = split_image_selfie(input, 8)

            batches = list(map(lambda x: x.unsqueeze(1), batches))
            batches = torch.cat(batches, 1) # (B, L, C, H, W)

            input_batches = torch.split(batches, 1, 1)
            input_batches = list(map(lambda x: x.squeeze(1), input_batches))
            input_batches = torch.cat(input_batches, 0)

            output_batches = P(input_batches)

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

                prec1, _ = accuracy(logit, temptarget, topk=(1,3))

                losses.update(loss_.item(), 1)
                top1.update(prec1[0], 1)

            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                           index, len(val_loader), batch_time=batch_time, loss=losses,
                           top1=top1))
            #raise NotImplementedError
        print(' * Prec@1 {top1.avg:.3f}'
              .format(top1=top1))
        return losses.avg, top1.avg

def val_pgd_selfie(val_loader, selfie_model, P, criterion, all_seq):

    '''
    print("Preheating ... ")
    with torch.no_grad():
        h = 0
        while h < 10:
            for index, (input, _) in enumerate(val_loader):
                input = input.cuda(args.gpu, non_blocking = True)
                if input.shape[-1] == 32:
                    num_split = 8
                else:
                    num_split = 32
                batches = split_image(input, num_split)
                batches = list(map(lambda x: x.unsqueeze(1), batches))
                batches = torch.cat(batches, 1)
                total = batches.shape[1]
                seq = np.random.permutation(total)
                t = seq[:int(np.trunc(total / 4.0))]
                v = seq[int(np.trunc(total / 4.0)):]
                v = torch.from_numpy(v).cuda(args.gpu, non_blocking = True)
                pos = t
                t = torch.from_numpy(np.array(pos)).cuda(args.gpu, non_blocking = True)
                input_for_transformer = batches.index_select(1, v)
                shape = input_for_transformer.shape
                output_for_transformer = []
                output_for_decoder = []
                input_for_decoder = batches.index_select(1, t)
                for i in range(shape[1]):
                    d = P(input_for_transformer[:, i, :, :, :]).unsqueeze(1)
                    output_for_transformer.append(d)
                output_for_transformer = torch.cat(output_for_transformer, 1)
                after_position_embeddings = selfie_model(output_for_transformer, pos)
                for i in range(input_for_decoder.shape[1]):
                    output_for_decoder.append(P(input_for_decoder[:, i, :, :, :]).unsqueeze(1)) 
            h += 1
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    selfie_model.eval()
    P.eval()
    
    for index, (input, _) in enumerate(val_loader):
        #print(input)
        data_time.update(time.time() - end)

        cur_batch_size = input.size(0)
        
        total=16
        seq = all_seq[index]
        t = seq[:(total // 4)]
        v = seq[(total // 4):]
        v = torch.from_numpy(v).cuda()
        pos = t

        t = torch.from_numpy(np.array(pos)).cuda()

        input_adv=PGD_attack(input,
            selfie_model=selfie_model,
            P=P,
            criterion=criterion,
            seq=seq,
            eps=(8/255),
            steps=20,
            gamma=(2/255),
            randinit=True)
            
        with torch.no_grad():
            input = input_adv.cuda()
            batches = split_image_selfie(input, 8)

            batches = list(map(lambda x: x.unsqueeze(1), batches))
            batches = torch.cat(batches, 1) # (B, L, C, H, W)

            input_batches = torch.split(batches, 1, 1)
            input_batches = list(map(lambda x: x.squeeze(1), input_batches))
            input_batches = torch.cat(input_batches, 0)

            output_batches = P(input_batches)

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

                prec1, _ = accuracy(logit, temptarget, topk=(1,3))

                losses.update(loss_.item(), 1)
                top1.update(prec1[0], 1)
            
            batch_time.update(time.time() - end)
            end = time.time()

            if index % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                            index, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
            .format(top1=top1))
    return losses.avg, top1.avg

def get_selfie_model(n_split):
    n_layers = 12
    d_model = 1024 #vector length after the patch routed in P
    d_in = 64
    n_heads = d_model// d_in
    d_ff = 2048
    model = SelfieModel(n_layers, n_heads, d_in, d_model, d_ff, n_split)
    return model

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

def accuracy(output, target, topk=(1,)):
    #print(output.shape)
    #print(target.shape)
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        #print(target)
        if (target.dim() > 1):
            target = torch.argmax(target, 1)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename[0])
    if is_best:
        shutil.copyfile(filename[0], filename[1])

def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def warmup_lr(step, optimizer, speed):
    lr = 0.01+step*(0.1-0.01)/speed
    lr = min(lr,0.1)
    for p in optimizer.param_groups:
        p['lr']=lr

def split_image_selfie(image, N):
    """
    image: (B, C, W, H)
    """
    batches = []

    for i in list(torch.split(image, N, dim=2)):
        batches.extend(list(torch.split(i, N, dim=3)))

    return batches

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

if __name__ == '__main__':
    main()