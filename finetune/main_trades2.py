import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import pickle
import numpy as np  
import PIL.Image as Image
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
import random

from resnetv2 import ResNet50
from attack_algo import Trades
from advertorch.utils import NormalizeByChannelMeanStd

parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')

parser.add_argument('--data', type=str, default='/data4/zzy/data/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=3e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--print_freq', default=50, type=int, help='print frequency')
parser.add_argument('--decreasing_lr', default='40,60', help='decreasing strategy')

parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model path')
parser.add_argument('--save_dir', help='The directory used to save the trained models', default='adv', type=str)
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=20, help='random seed')

best_prec1 = 0
best_ata = 0

def main():
    global args, best_prec1, best_ata
    args = parser.parse_args()
    print(args)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    torch.cuda.set_device(int(args.gpu))
    setup_seed(args.seed)

    model = ResNet50()
    normalize = NormalizeByChannelMeanStd(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    model = nn.Sequential(normalize, model)
    model.cuda()
    cudnn.benchmark = True

    if args.pretrained_model:
        model_dict_pretrain = torch.load(args.pretrained_model, map_location=torch.device('cuda:'+str(args.gpu)))
        model.load_state_dict(model_dict_pretrain, strict=False)
        print('model loaded:', args.pretrained_model)

    #dataset process
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
    
    decreasing_lr = list(map(int, args.decreasing_lr.split(',')))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
                                
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=decreasing_lr, gamma=0.1)


    print('adv training')

    train_acc=[]
    ta=[]
    ata=[]

    if os.path.exists(args.save_dir) is not True:
        os.mkdir(args.save_dir)


    for epoch in range(args.epochs):

        print(optimizer.state_dict()['param_groups'][0]['lr'])
        acc,loss = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        tacc,tloss = validate(val_loader, model, criterion)

        atacc,atloss = validate_adv(val_loader, model, criterion)

        scheduler.step()

        train_acc.append(acc)
        ta.append(tacc)
        ata.append(atacc)


        # remember best prec@1 and save checkpoint
        is_best = tacc  > best_prec1
        best_prec1 = max(tacc, best_prec1)

        ata_is_best = atacc > best_ata
        best_ata = max(atacc,best_ata)
        
        if is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'best_model.pt'))

        if ata_is_best:

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'ata_best_model.pt'))

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.pt'))
    
        plt.plot(train_acc, label='train_acc')
        plt.plot(ta, label='TA')
        plt.plot(ata, label='ATA')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, 'net_train.png'))
        plt.close()

    best_model_path = os.path.join(args.save_dir, 'ata_best_model.pt')
    print('start testing ATA best model')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    tacc,tloss = validate(test_loader, model, criterion)
    atacc,atloss = validate_adv(test_loader, model, criterion)

    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    print('start testing TA best model')
    model.load_state_dict(torch.load(best_model_path)['state_dict'])
    tacc,tloss = validate(test_loader, model, criterion)
    atacc,atloss = validate_adv(test_loader, model, criterion)
        
        

def train(train_loader, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    top1 = AverageMeter()

    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):

        #warm up
        if epoch == 0:
            warmup_lr(i, optimizer, 200)

        input = input.cuda()
        target=target.cuda()

        #adv samples
        input_adv = Trades(input,
            eps=(8/255),
            model=model,
            steps=10,
            gamma=(2/255),
            randinit=True)
        
        input_adv = input_adv.cuda()

        # compute output
        output_adv = model(input_adv)
        output_clean = model(input)

        loss = criterion(output_clean, target) + criterion_kl(F.log_softmax(output_adv, dim=1), F.softmax(output_clean, dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_adv.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))


        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                      epoch, i, len(train_loader), loss=losses, top1=top1))

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg, losses.avg
    
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1))

    print('valid_accuracy {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def validate_adv(val_loader, model, criterion):
    """
    Run evaluation
    """
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()

        #adv samples
        input_adv = PGD_normal(input, criterion,
            y=target,
            eps=(8/255),
            model=model,
            steps=20,
            gamma=(2/255),
            randinit=True)

        input_adv = input_adv.cuda()
        # compute output
        with torch.no_grad():
            output = model(input_adv)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), loss=losses, top1=top1))

    print('ATA {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

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


if __name__ == '__main__':
    main()


