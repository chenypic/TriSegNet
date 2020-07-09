# -*- coding: utf-8 -*-

"""
TriSegNet on brats2015

@author: Wilson Chen
"""



import time
import argparse
import torch

import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

#import torchvision.transforms as transforms

from torch.utils.data import DataLoader


import os
import sys
import math

import shutil

import setproctitle

#import SegRes_mul
import TriSegNet
from functools import reduce
import operator
import nibabel as nib

from bratsdata import get_train_valid_loader

from loss import dice_loss
from evaluate_per import evaluate_subject

output_dir = "train_test"

def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)


def print_lr(optAlg, optimizer):
    for param_group in optimizer.param_groups:
        print("the current learning rate：",param_group['lr'])



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
        #m.bias.data.zero_()
'''
def save_checkpoint(state, is_best, path, prefix, filename='checkpoint'):
    prefix_save = os.path.join(path, prefix)
    filename = filename+'_'+ "{}".format(state['epoch'])
    filename = filename + '.pth.tar'
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')
'''

def save_checkpoint(state, is_best, path, prefix, filename='checkpoint.pth.tar'):
    prefix_save = os.path.join(path, prefix)
    name = prefix_save + '_' + filename
    torch.save(state, name)
    if is_best:
        shutil.copyfile(name, prefix_save + '_model_best.pth.tar')


'''
def get_prediction_labels(prediction, threshold=0.5, labels=None):
    prediction = prediction.cpu().detach().numpy()
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0) + 1
        label_data[np.max(prediction[sample_number], axis=0) < threshold] = 0
        if labels:
            for value in np.unique(label_data).tolist()[1:]:
                label_data[label_data == value] = labels[value - 1]
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def prediction_to_image(prediction, affine, threshold=0.5, labels=(1,2,3,4)):
    label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
    data = label_map_data[0]
    return nib.Nifti1Image(data, affine)
'''

def get_prediction_labels(prediction, threshold=0.5, labels=None):
    prediction = prediction.cpu().detach().numpy()
    n_samples = prediction.shape[0]
    label_arrays = []
    for sample_number in range(n_samples):
        label_data = np.argmax(prediction[sample_number], axis=0)
        label_arrays.append(np.array(label_data, dtype=np.uint8))
    return label_arrays


def prediction_to_image(prediction, affine, threshold=0.5, labels=(1,2,3,4)):
    label_map_data = get_prediction_labels(prediction, threshold=threshold, labels=labels)
    data = label_map_data[0]
    return nib.Nifti1Image(data, affine)


def run_validation_case(output_dir,sample_batched,prediction):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine =  sample_batched['affine'][0]
    test_data = np.asarray(sample_batched['image'])
    
    test_truth = nib.Nifti1Image(np.asarray(sample_batched['truth'][0][0]), affine)

    prediction_image = prediction_to_image(prediction, affine)

    return evaluate_subject(test_truth, prediction_image)



def train_dice(args, epoch, model, trainLoader, optimizer, trainF):
    model.train()
    nProcessed = 0
    nTrain = len(trainLoader)
    #train_loss = 0
    for batch_idx, sample_batched in enumerate(trainLoader):
        data = sample_batched['image'].cuda()
        target = sample_batched['truth'].cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = dice_loss(output, target)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        err = 100.*(loss.data[0])
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tError: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_dice(args, epoch, model, testLoader, optimizer, testF):
    model.eval()
    test_loss = 0
    incorrect = 0
    all_results = []

    with torch.no_grad():  
        for batch_idx, sample_batched  in enumerate(testLoader):
            data = sample_batched['image'].cuda()
            target = sample_batched['truth'].cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            loss = dice_loss(output, target).data[0]
            test_loss += loss
            incorrect += loss
            print(loss)

            case_directory = os.path.join(output_dir,sample_batched['subject_ids'][0].decode('utf-8'))
            result_eva = run_validation_case(case_directory,sample_batched,output)
            per_list = []
            per_list.append(result_eva["WholeTumor"])
            per_list.append(result_eva["TumorCore"])
            per_list.append(result_eva["EnhancingTumor"])
            all_results.append(per_list)

        all_results = np.array(all_results)
        rr = np.argwhere(np.isnan(all_results))
        for i in range(len(rr)):
            all_results[rr[i][0],rr[i][1]] = 1
        average = all_results.mean(axis = 0)
        print("Average: whole,core,enhanceing:",average)

        test_loss /= len(testLoader)  # loss function already averages over batch size
        print('\nTest set: Average Dice Coeff: {:.4f}, \n'.format(test_loss))
        testF.write('{},{},{},{},{}\n'.format(epoch, test_loss,average[0],average[1],average[2]))
        testF.flush()
    return test_loss




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=2)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=100)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    best_prec1 = 100.
    error_old = 100.

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    idsFolder = datestr()
    args.save = args.save or 'work/TriSegNet.base.{}'.format(idsFolder)
    weight_decay = args.weight_decay
    setproctitle.setproctitle(args.save)
    device = torch.device("cuda" if args.cuda else "cpu")

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build TriSegNet")
    model = TriSegNet.TriSegNet().to(device)
    batch_size = args.ngpu*args.batchSz
    gpu_ids = range(args.ngpu)
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)


    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            error_old = best_prec1
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        model.apply(weights_init)


    
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")

    data_file = "/media/s209/D/AiChery_data/brats2015/training_all_labels_no_hist.h5"
    trainLoader,testLoader,_,dataset, = get_train_valid_loader(data_dir = data_file,batch_size=1)
    print(len(trainLoader))
  
    print(len(testLoader))

    lr_init = 1e-3
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr_init,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=lr_init, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(),lr=lr_init, weight_decay=weight_decay)


    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    #print(model.parameters)
    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    print(trainLoader)

    err_no_reduce_count = 0
    lr_reduced_flag = 0

    for epoch in range(1,args.nEpochs+1):
        print_lr(args.opt, optimizer)
        train_dice(args, epoch, model, trainLoader, optimizer, trainF)
        err = test_dice(args, epoch, model, testLoader, optimizer, testF)
        if err < best_prec1:
            is_best = True
            save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': err},
                        is_best, args.save, "TriSegNet")
            best_prec1 = err 
            err_no_reduce_count = 0
        else:
            err_no_reduce_count+=1
            is_best = False
            save_checkpoint({'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'best_prec1': best_prec1},
                        is_best, args.save, "TriSegNet")
            if err_no_reduce_count == 10 and lr_reduced_flag == 0:
                lr_reduced_flag = 1
                lr_init = lr_init/10
                optimizer = optim.Adam(model.parameters(),lr=lr_init, weight_decay=weight_decay)
                args.resume = 'work/TriSegNet.base.{}'.format(idsFolder)+'/TriSegNet_model_best.pth.tar' 
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                args.start_epoch = checkpoint['epoch']
                best_prec1 = checkpoint['best_prec1']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.evaluate, checkpoint['epoch']))
            if err_no_reduce_count == 20 and lr_reduced_flag == 1:
                break
        os.system('./plot.py {} {} &'.format(len(trainLoader), args.save))



if __name__ == '__main__':
    main()


