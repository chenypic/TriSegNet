#!/usr/bin/env python3

import argparse
import os
import numpy as np

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
#plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nBatches', type=int)
    parser.add_argument('expDir', type=str)
    args = parser.parse_args()

    trainP = os.path.join(args.expDir, 'train.csv')
    trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
    #with open(trainP, 'r') as f:
    #    data1 = f.readlines()
    #trainData = []
    #for i in range(len(data1)):
    #    trainData.append([float(data1[i].split(',')[1:][0]),float(data1[i].split(',')[1:][1]),float(data1[i].split(',')[1:][2])])
    #trainData = np.array(trainData)
    testP = os.path.join(args.expDir, 'test.csv')
    testData = np.loadtxt(testP, delimiter=',').reshape(-1, 5)

    N = args.nBatches
    trainI, trainLoss, trainErr = np.split(trainData, [1,2], axis=1)
    trainI, trainLoss, trainErr = [x.ravel() for x in
                                   (trainI, trainLoss, trainErr)]
    trainI_, trainLoss_, trainErr_ = rolling(N, trainI, trainLoss, trainErr)

    testI, loss, whole, cores, enhancing  = np.split(testData,[1,2,3,4] ,axis=1)
    '''
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    # plt.plot(trainI, trainLoss, label='Train')
    plt.plot(trainI_, trainLoss_, label='Train')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    ax.set_yscale('log')
    loss_fname = os.path.join(args.expDir, 'loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))
    '''
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # plt.plot(trainI, trainErr, label='Train')
    #plt.plot(testI, loss, label='loss')
    plt.plot(testI, whole, label='whole')
    plt.plot(testI, cores, label='core')
    plt.plot(testI, enhancing, label='enhancing')
    plt.xlabel('Epoch')
    plt.ylabel('dice')
    #ax.set_yscale('log')
    plt.legend()
    dice_fname = os.path.join(args.expDir, 'dice.png')
    plt.savefig(dice_fname)
    print('Created {}'.format(dice_fname))
    '''
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    # plt.plot(trainI, trainErr, label='Train')
    plt.plot(testI, loss, label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('test loss')
    ax.set_yscale('log')
    plt.legend()
    test_loss_fname = os.path.join(args.expDir, 'test_loss.png')
    plt.savefig(test_loss_fname)
    print('Created {}'.format(test_loss_fname))
    '''
    tt = trainLoss[0:N*(trainI.shape[0]//N)].reshape(-1,N)
    train_loss_mean = tt.mean(1)
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    plt.plot(range(trainI.shape[0]//N), tt.mean(1), label='Train Loss')
    plt.plot(testI-1, loss, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Generalizes Dice Loss')
    plt.annotate("%.6s"%loss.min(), xy=(loss.argmin(), loss.min()), xytext=(loss.argmin(), loss.min()+0.05),arrowprops=dict(facecolor='red', shrink=0.05))
    plt.annotate("%.6s"%train_loss_mean.min(), xy=(train_loss_mean.argmin(), train_loss_mean.min()), xytext=(train_loss_mean.argmin(), train_loss_mean.min()-0.05),arrowprops=dict(facecolor='blue', shrink=0.05))
    plt.ylim(0.1,0.8,0.1)
    plt.xlim(0,trainI.shape[0]//N+2)
    plt.legend()
    gdl_fname = os.path.join(args.expDir, 'train_test_loss.png')
    plt.savefig(gdl_fname)



def rolling(N, i, loss, err):
    i_ = i[N-1:]
    K = np.full(N, 1./N)
    loss_ = np.convolve(loss, K, 'valid')
    err_ = np.convolve(err, K, 'valid')
    return i_, loss_, err_

if __name__ == '__main__':
    main()
