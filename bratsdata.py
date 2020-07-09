# -*- coding: utf-8 -*-

import numpy as np
import tables

import torch
import pandas as pd
#from skimage import io, transform
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler,SequentialSampler

from sklearn.model_selection import KFold


def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


# data_file = "brats_data.h5"

class BRATSDATA(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file):

        self.data_file_opened = open_data_file(data_file)
        

    def __len__(self):
        return len(self.data_file_opened.root.data)

    def __getitem__(self, idx):

        image = self.data_file_opened.root.data[idx]
        landmarks = self.data_file_opened.root.truth_per[idx].astype(np.float64)
        affine = self.data_file_opened.root.affine[idx]
        truth = self.data_file_opened.root.truth[idx]
        subject_ids = self.data_file_opened.root.subject_ids[idx]
        sample = {'image': image, 'landmarks': landmarks,'affine':affine, 'truth':truth, 'subject_ids':subject_ids}
        return sample



def get_train_valid_loader(data_dir,
                           batch_size=1,
                           random_seed=2,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=True):
    """
    refer: https://github.com/pytorch/pytorch/issues/1106
    https://gist.github.com/kevinzakka/d33bf8d6c7f06a9d8c76d97a7879f5cb

    """


    train_dataset = BRATSDATA(data_file = data_dir)


    num_train = len(train_dataset)
    indices = list(range(num_train))
    #split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    #train_idx, valid_idx = indices[split:], indices[:split]
    #train_sampler = SubsetRandomSampler(train_idx)
    #valid_sampler = SubsetRandomSampler(valid_idx)
    #print(train_idx)
    #print(valid_idx)

    #  添加交叉验证拆分数据集的代码，2018年7月11日
    kf=KFold(n_splits=10)
    train_list = []
    valid_list = []
    indices = np.array(indices)
    for train_index,test_index in kf.split(indices):
        #print("Train Index:",train_index,",Test Index:",test_index)
        train_list.append(list(indices[train_index]))
        valid_list.append(list(indices[test_index]))

    train_sampler = SubsetRandomSampler(train_list[0])
    valid_sampler = SubsetRandomSampler(valid_list[0])
    #train_sampler = SequentialSampler(train_list[1])
    #valid_sampler = SequentialSampler(valid_list[1])
    #print(train_list[1])
    #print(valid_list[1])


    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler
    )
    valid_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=valid_sampler
    )

    # visualize some images
    all_data = DataLoader(
        train_dataset, batch_size=batch_size,shuffle=True
    )

    return train_loader, valid_loader, train_dataset, all_data





class BRATSDATA_VAL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_file):

        self.data_file_opened = open_data_file(data_file)
        

    def __len__(self):
        return len(self.data_file_opened.root.data)

    def __getitem__(self, idx):

        image = self.data_file_opened.root.data[idx]
        #landmarks = self.data_file_opened.root.truth_per[idx].astype(np.float64)
        affine = self.data_file_opened.root.affine[idx]
        #truth = self.data_file_opened.root.truth[idx]
        subject_ids = self.data_file_opened.root.subject_ids[idx]
        sample = {'image': image, 'affine':affine, 'subject_ids':subject_ids}
        return sample


def get_validation_loader(data_dir,
                           batch_size=1,
                           random_seed=1,
                           valid_size=0.2,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=True):

    validation_dataset = BRATSDATA_VAL(data_file = data_dir)

    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size
    )

    return validation_loader