# -*- coding: utf-8 -*-

import torch

'''
def dice_loss(x, target, smooth=1.0):
    # pt = 2.0 * ((x.float()*target.float()).sum(2).sum(2).sum(2)).float()
    pt = x.float() * target.float()
    pt = pt.sum(2).sum(2).sum(2)
    pt = 2 * pt 
    p = x.float().sum(2).sum(2).sum(2)
    t = target.float().sum(2).sum(2).sum(2)
    dices =  ((pt + smooth) / (p + t + smooth)).mean()
    #print(x.shape)
    #print(target.shape)
    #print(t)
    losses = 1 - dices
    return losses
'''
'''
def dice_loss(x, target, smooth=1.0):
    # pt = 2.0 * ((x.float()*target.float()).sum(2).sum(2).sum(2)).float()
    pt = x.float() * target.float()
    pt = pt.sum(2).sum(2).sum(2)

    p = x.float().sum(2).sum(2).sum(2)
    t = target.float().sum(2).sum(2).sum(2)

    weights = (t ** 2 + smooth).pow(-1)

    generalised_dice_numerator = 2 * (weights *pt)

    generalised_dice_denominator = weights*(p+t)

    generalised_dice_score =(generalised_dice_numerator /generalised_dice_denominator).mean()

    GDL=1-generalised_dice_score

    return GDL
'''



def dice_loss(x, target, smooth=1e-7):
    
    #Pytorch is channel first by default. See https://pytorch.org/docs/master/nn.html

    #target 没有进行one-hot,现在target的shape为：torch.Size([1, 1, 128, 128, 128])
    
    #print('预测的shape:',x.shape)

    truth = torch.zeros(x.shape).cuda()
    #print("trurh的shape：",truth.shape)

    truth[:,0,:,:,:] = (target[:,0,:,:,:] ==0)
    truth[:,1,:,:,:] = (target[:,0,:,:,:] ==1)
    truth[:,2,:,:,:] = (target[:,0,:,:,:] ==2)
    truth[:,3,:,:,:] = (target[:,0,:,:,:] ==3)
    truth[:,4,:,:,:] = (target[:,0,:,:,:] ==4)


    pt = x.float() * truth.float()
    pt = pt.sum(0).sum(1).sum(1).sum(1)

    p = x.float().sum(0).sum(1).sum(1).sum(1)
    t = truth.float().sum(0).sum(1).sum(1).sum(1)

    weights = (t ** 2 + smooth).pow(-1)

    generalised_dice_numerator = 2 * (weights *pt)

    generalised_dice_denominator = weights*(p+t)

    generalised_dice_score =(generalised_dice_numerator /generalised_dice_denominator).mean()

    GDL=1-generalised_dice_score

    return GDL


   
#def gen_dice_loss(y_true, y_pred):
#    '''
#    computes the sum of two losses : generalised dice loss and weighted cross entropy
#    '''
#
#    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237
#    y_true_f = K.reshape(y_true,shape=(-1,4))
#    y_pred_f = K.reshape(y_pred,shape=(-1,4))
#    sum_p=K.sum(y_pred_f,axis=-2)
#    sum_r=K.sum(y_true_f,axis=-2)
#    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
#    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
#    generalised_dice_numerator =2*K.sum(weights*sum_pr)
#    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
#    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
#    GDL=1-generalised_dice_score
#    del sum_p,sum_r,sum_pr,weights
#
#    return GDL+weighted_log_loss(y_true,y_pred)