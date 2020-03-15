import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np

class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
           stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
           data = stats_['data']
           content = data[0,0]
           self.trainObj = content['trainObj'][:,:start_epoch].squeeze().tolist()
           self.trainTop1 = content['trainTop1'][:,:start_epoch].squeeze().tolist()
           #self.trainTop5 = content['trainTop5'][:,:start_epoch].squeeze().tolist()
           self.valObj = content['valObj'][:,:start_epoch].squeeze().tolist()
           self.valTop1 = content['valTop1'][:,:start_epoch].squeeze().tolist()

           self.avalObj = content['adv_valObj'][:,:start_epoch].squeeze().tolist()
           self.avalTop1 = content['adv_prec1'][:,:start_epoch].squeeze().tolist()
           #self.valTop5 = content['valTop5'][:,:start_epoch].squeeze().tolist()
           if start_epoch is 1:
               self.trainObj = [self.trainObj]
               self.trainTop1 = [self.trainTop1]
               #self.trainTop5 = [self.trainTop5]
               self.valObj = [self.valObj]
               self.valTop1 = [self.valTop1]

               self.avalObj = [self.avalObj]
               self.avalTop1 = [self.avalTop1]
               #self.valTop5 = [self.valTop5]
        else:
           self.trainObj = []
           self.trainTop1 = []
           #self.trainTop5 = []
           self.valObj = []
           self.valTop1 = []

           self.avalObj = []
           self.avalTop1 = []
           #self.valTop5 = []
    def _update(self, trainObj, top1, valObj, prec1, avalObj, aprec1):
        self.trainObj.append(trainObj)
        self.trainTop1.append(top1.cpu().numpy())
        #self.trainTop5.append(top5.cpu().numpy())
        self.valObj.append(valObj)
        self.valTop1.append(prec1.cpu().numpy())

        self.avalObj.append(avalObj)
        self.avalTop1.append(aprec1.cpu().numpy())
        #self.valTop5.append(prec5.cpu().numpy())

def plot_curve(stats, path, iserr):
    trainObj = np.array(stats.trainObj)
    valObj = np.array(stats.valObj)
    avalObj = np.array(stats.avalObj)
    if iserr:
        trainTop1 = 100 - np.array(stats.trainTop1)
        #trainTop5 = 100 - np.array(stats.trainTop5)
        valTop1 = 100 - np.array(stats.valTop1)

        avalTop1 = 100 - np.array(stats.avalTop1)
        #valTop5 = 100 - np.array(stats.valTop5)
        titleName = 'error'
    else:
        trainTop1 = np.array(stats.trainTop1)
        #trainTop5 = np.array(stats.trainTop5)
        valTop1 = np.array(stats.valTop1)

        avalTop1 = np.array(stats.avalTop1)
        #valTop5 = np.array(stats.valTop5)
        titleName = 'accuracy'
    epoch = len(trainObj)
    figure = plt.figure()
    obj = plt.subplot(1,2,1)
    obj.plot(range(1,epoch+1),trainObj,'o-',label = 'train')
    obj.plot(range(1,epoch+1),valObj,'o-',label = 'val')
    obj.plot(range(1,epoch+1),avalObj,'o-',label = 'adv_val')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    top1 = plt.subplot(1,2,2)
    top1.plot(range(1,epoch+1),trainTop1,'o-',label = 'train')
    top1.plot(range(1,epoch+1),valTop1,'o-',label = 'val')
    top1.plot(range(1,epoch+1),avalTop1,'o-',label = 'adv_val')
    plt.title('top1'+titleName)
    plt.xlabel('epoch')
    handles, labels = top1.get_legend_handles_labels()
    top1.legend(handles[::-1], labels[::-1])
    '''
    top5 = plt.subplot(1,3,3)
    top5.plot(range(1,epoch+1),trainTop5,'o-',label = 'train')
    top5.plot(range(1,epoch+1),valTop5,'o-',label = 'val')
    plt.title('top5'+titleName)
    plt.xlabel('epoch')
    handles, labels = top5.get_legend_handles_labels()
    top5.legend(handles[::-1], labels[::-1])
    '''
    filename = os.path.join(path, 'net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    plt.close()