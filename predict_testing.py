
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
import SimpleITK as sitk
import brainSeg_V13_end
from functools import reduce
import operator
import nibabel as nib
from glob import glob
from bratsdata import get_validation_loader
from nilearn.image import resample_to_img
from loss import dice_loss
from evaluate_per import evaluate_subject

output_dir = 'predict_dice_testing_V13'


def load_img(filename):
    nda = nib.load(filename)
    return nda

'''
def get_prediction_labels(prediction, threshold=0.001, labels=None):
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


def run_validation_case(output_dir,sample_batched,prediction,ids):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    affine =  sample_batched['affine'][0]
    test_data = np.asarray(sample_batched['image'])

    training_modalities = ["t1", "t1ce", "flair", "t2"]
    for i, modality in enumerate(training_modalities):
        image = nib.Nifti1Image(test_data[0, i], affine)
        image.to_filename(os.path.join(output_dir, "data_{0}.nii.gz".format(modality)))

    prediction_image = prediction_to_image(prediction, affine)
    prediction_image.to_filename(os.path.join(output_dir, "prediction.nii.gz"))
    '''
    dirs = '/media/s209/D/AiChery_data/brats2015/Testing_nii/HGG_LGG/'
    files = dirs+ids
    Flair = glob(os.path.join(files,"flair.nii.gz"))[0]
    img_Flair = load_img(Flair)
    final_prediction = resample_to_img(prediction_image, img_Flair, interpolation="nearest")
    final_prediction.to_filename(os.path.join(output_dir, "final_prediction.nii.gz"))

    #final_prediction_data = final_prediction.get_data()
    #print('final_prediction_data的shape:',final_prediction_data.shape)

    time.sleep(1)
    img = sitk.ReadImage(os.path.join(output_dir, "final_prediction.nii.gz"))
    #img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt16)


    mha_dirs = '/media/s209/D/AiChery_data/brats2015/BRATS2015_Testing/HGG_LGG/'
    mha_files = mha_dirs+ ids[:-6]
    print(mha_files)
    mha_Flair = glob(os.path.join(mha_files,"*","*Flair*.mha"))[0]
    print(mha_Flair)

    img_ref = sitk.ReadImage(mha_Flair)
    #fff = sitk.GetArrayFromImage(img_ref)
    #print("sitk转array的shape:",fff.shape)
    #print('sitk的shape:',img_ref.shape)
    #img = sitk.GetImageFromArray(final_prediction_data)
    img.CopyInformation(img_ref)
    #sitk.WriteImage(img, "final_prediction.mha")


    upload_dirs = 'upload'

    sitk.WriteImage(img, os.path.join(upload_dirs,"VSD.Testing."+ids.split("_")[-1] + '.mha'))
    '''


def test_dice(args, model, testLoader):
    model.eval()
    test_loss = 0
    incorrect = 0
    all_results = []


    with torch.no_grad():
        for batch_idx, sample_batched  in enumerate(testLoader):
            data = sample_batched['image'].cuda()
            data = Variable(data)
            output = model(data)
            
            case_directory = os.path.join(output_dir,sample_batched['subject_ids'][0].decode('utf-8'))
            ids = sample_batched['subject_ids'][0].decode('utf-8')
            run_validation_case(case_directory,sample_batched,output,ids)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--dice', action='store_true')
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-i', '--inference', default='', type=str, metavar='PATH',
                        help='run inference on data set and save results')

    # 1e-8 works well for lung masks but seems to prevent
    # rapid learning for nodule masks
    parser.add_argument('--weight-decay', '--wd', default=1e-8, type=float,
                        metavar='W', help='weight decay (default: 1e-8)')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    #torch.manual_seed(args.seed)
    #if args.cuda:
    #    torch.cuda.manual_seed(args.seed)

    print("build unet")
    model = brainSeg_V13_end.UNet().to(device)
    batch_size = args.ngpu*args.batchSz
    gpu_ids = range(args.ngpu)
    model = nn.parallel.DataParallel(model, device_ids=gpu_ids)


    #checkpoint = torch.load('work/unet.base.20181117_0334/vnet_model_best.pth.tar')
    #checkpoint = torch.load('work/unet.base.20181206_0942/vnet_checkpoint.pth.tar')
    #checkpoint = torch.load('work/unet.base.20181207_1500/vnet_model_best.pth.tar')
    #checkpoint = torch.load('work/unet.base.20181216_1421/vnet_checkpoint.pth.tar')
    #checkpoint = torch.load('work/unet.base.20181223_1312/vnet_model_best.pth.tar') #没有dense
    #checkpoint = torch.load('work/unet.base.20181221_1614/vnet_model_best.pth.tar') # 无 LP
    #checkpoint = torch.load('work/unet.base.20181225_0945/vnet_model_best.pth.tar') # 标准3D
    #checkpoint = torch.load('work/unet.base.20181226_0320/vnet_model_best.pth.tar') #渐变学习率
    #checkpoint = torch.load('work/unet.base.20181228_0212/vnet_model_best.pth.tar') #渐变学习率,二次
    #checkpoint = torch.load('work/unet.base.20181229_0135/vnet_checkpoint.pth.tar') #阶段学习
    #checkpoint = torch.load('work/unet.base.20190102_1250/vnet_model_best.pth.tar') #没有context
    #checkpoint = torch.load('work/unet.base.20190109_1641/vnet_checkpoint.pth.tar') #

    ####checkpoint = torch.load('work/unet.base.20190110_0829/vnet_checkpoint.pth.tar') #阶段学习

    #checkpoint = torch.load('work/unet.base.20190114_2321/vnet_checkpoint.pth.tar') 
    ######checkpoint = torch.load('work/unet.base.20190115_1453/vnet_checkpoint.pth.tar') 
    #############checkpoint = torch.load('work/unet.base.20190116_0404/vnet_model_best.pth.tar') 
    checkpoint = torch.load('work/unet.base.20190117_0025/vnet_checkpoint.pth.tar') 

    
    
    
    
    
    
    

    
    start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("best_prec1: ",best_prec1)
    print("start_epoch: ",start_epoch)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")

    data_file = "/media/s209/D/AiChery_data/brats2015/Testing_no_hist.h5"
    #data_file = "/media/s209/D/AiChery_data/brats2015/Testing_zanshide.h5"
    testLoader = get_validation_loader(data_dir = data_file,batch_size=batch_size)
    print(len(testLoader))
    test_dice(args, model,testLoader )


if __name__ == '__main__':
    main()
