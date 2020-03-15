import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import pickle
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import os


class CifarDataset(Dataset):

    def __init__(self, _dir, train, transform, percent):

        self.dir=osp.join( _dir, 'cifar-10-batches-py')
        self.transforms=transform
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'

        if train:
            data_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
            data_labels = np.zeros(50000, dtype='int32')
            for ii, fname in enumerate(train_filenames):
                cur_images, cur_labels = self._load_datafile(osp.join(self.dir, fname))
                data_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
                data_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
            permutation = np.random.permutation(50000)
            #self.number=int(n*percent)
            self.choose_images=data_images[permutation]
            self.choose_target=data_labels[permutation]
            choose = []
            all_indexes = np.array(range(50000))
            self.number = int(50000 * percent)
            for i in range(10):
                indexes = all_indexes[self.choose_target == i]
                choose.append(indexes[:int(len(indexes) * percent)])
            choose = np.concatenate(choose, 0)
            self.choose_images = self.choose_images[choose]
            self.choose_target = self.choose_target[choose]
        else:
            data_images, data_labels = self._load_datafile(osp.join(self.dir, eval_filename))
            self.number=int(10000*percent)
            permutation = np.random.permutation(10000)
            self.choose_images=data_images[permutation[:self.number]]
            self.choose_target=data_labels[permutation[:self.number]]
            self.number = 10000
        
        
        ## Adjust the percent that model uses

    def __len__(self):

        return self.number

    def __getitem__(self, index):

        # img = Image.fromarray(self.choose_images[index])
        img=self.choose_images[index]
        target = self.choose_target[index]
        img = self.transforms(img)

        return img,target

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])

class Cifar_C_Dataset(Dataset):

    def __init__(self, _dir, file_name, transform):

        self.dir=_dir
        self.transforms=transform
        
        self.data_images = np.load(osp.join(self.dir, file_name))
        self.data_labels = np.load(osp.join(self.dir, 'labels.npy'))

        self.number=self.data_images.shape[0]
        
    def __len__(self):

        return self.number

    def __getitem__(self, index):

        img = self.data_images[index]
        target = self.data_labels[index]
        img = self.transforms(img)

        return img,target

class ImageNetDataset(Dataset):

    def __init__(self, _dir, train, transform, percent):

        self.dir=_dir
        self.transforms=transform
        train_filenames = ['train_data_batch_{}'.format(ii + 1) for ii in range(10)]
        eval_filename = 'val_data'

        if train:
            data_images = []
            data_labels = []
            for ii, fname in enumerate(train_filenames):
                cur_images, cur_labels = self._load_datafile(osp.join(self.dir, fname))
                data_images.append(cur_images)
                data_labels.append(cur_labels)
            data_images = np.concatenate(data_images, axis = 0)
            data_labels = np.concatenate(data_labels, axis = 0)
            n = data_images.shape[0]
            permutation = np.random.permutation(n)
            #self.number=int(n*percent)
            self.choose_images=data_images[permutation]
            self.choose_target=data_labels[permutation]
            choose = []
            all_indexes = np.array(range(n))
            for i in range(1000):
                indexes = all_indexes[self.choose_target == i]
                choose.append(indexes[:int(len(indexes) * percent)])
            choose = np.concatenate(choose, 0)
            self.choose_images = self.choose_images[choose]
            self.choose_target = self.choose_target[choose]
            #for i in range(1000):
            #    print(np.sum(self.choose_target == i))
        else:
            data_images, data_labels = self._load_datafile(osp.join(self.dir, eval_filename))
            n = data_images.shape[0]
            self.number=int(n*percent)
            self.choose_images=data_images[:self.number]
            self.choose_target=data_labels[:self.number]
        
        
        ## Adjust the percent that model uses

    def __len__(self):

        return len(self.choose_images)

    def __getitem__(self, index):

        img = self.choose_images[index]
        target = self.choose_target[index]
        img = self.transforms(img)

        return img,target -1

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            assert data_dict['data'].dtype == np.uint8
            image_data = data_dict['data']
            image_data = image_data.reshape((image_data.shape[0], 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict['labels'])
















