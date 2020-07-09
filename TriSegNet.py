
from __future__ import print_function, division, absolute_import
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init


class RefinementBlock(nn.Module):
    def __init__(self,cin,cout,relu=True,norm=True):
        super(RefinementBlock,self).__init__()

        self.conv1_1 = nn.Conv3d(cin,cout,3,padding=1)
        self.norm1_1 = nn.InstanceNorm3d(cout)
        self.relu1_1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()

    def forward(self,x):

        result = self.relu1_1(self.norm1_1(self.conv1_1(x)))
        return self.relu2(x + result)


class SEModule(nn.Module):

    def __init__(self, channels):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels , kernel_size=1,
                             padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels , channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEModule_1(nn.Module):

    def __init__(self, channels):
        super(SEModule_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
       

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        return module_input * x


class SEModule_mul_add(nn.Module):

    def __init__(self, channels):
        super(SEModule_mul_add, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(channels, channels, kernel_size=1,
                             padding=0)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv3d(channels, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        x = module_input * x
        #x = module_input + x
        return x



class Separable3d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(Separable3d,self).__init__()
          
        self.conv1 = nn.Conv3d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv3d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class con3dblock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(con3dblock,self).__init__()
          
        self.conv1 = nn.Conv3d(in_channels,out_channels,kernel_size,stride,padding,dilation,bias=bias)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class con3dblockup(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(con3dblockup,self).__init__()
          
        self.conv1 = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,output_padding=1,dilation=1,bias=bias)
        self.bn = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:  #如果输入不等于输出的channel，或者stride不等于1
            self.skip = nn.Conv3d(in_filters,out_filters,1,stride=strides,bias=False) #先通过1*1的卷积，修改尺寸和channels数量
            self.skipbn = nn.InstanceNorm3d(out_filters)
        else:
            self.skip=None

        self.relu = nn.LeakyReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(Separable3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(Separable3d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(Separable3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.LeakyReLU(inplace=False)

        if strides != 1:
            #rep.append(nn.MaxPool3d(3,strides,padding=1))
            rep.append(nn.Conv3d(filters,filters,kernel_size=3, stride=2, padding=1))
            rep.append(nn.InstanceNorm3d(filters))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class BlockUp(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(BlockUp, self).__init__()

        if out_filters != in_filters or strides!=1:  #如果输入不等于输出的channel，或者stride不等于1
            self.skip = nn.ConvTranspose3d(in_filters,out_filters,3,stride=strides,padding=1,output_padding=1,bias=False) #先通过1*1的卷积，修改尺寸和channels数量
            self.skipbn = nn.InstanceNorm3d(out_filters)
        else:
            self.skip=None

        self.relu = nn.LeakyReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(Separable3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(Separable3d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(Separable3d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.InstanceNorm3d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.LeakyReLU(inplace=False)

        if strides != 1:
            rep.append(nn.Upsample(scale_factor=strides,mode='trilinear'))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Isomorphism_incept(nn.Module):
    '''
    spatial-temporary-resnet
    '''
    def __init__(self,cin,co,relu=True,norm=True):
        super(Isomorphism_incept,self).__init__()
        assert(co%4==0)
        cos=[co//4]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.InstanceNorm3d(co))
        if relu:self.activa.add_module('relu',nn.LeakyReLU())
        
        self.branch1 = nn.Conv3d(cin, cos[0], 1, stride=1)
        self.branch1_bn = nn.InstanceNorm3d(cos[0])
        self.branch1_relu = nn.LeakyReLU()
        
        self.branch2_0 = nn.Conv3d(cin, cos[1], 1, stride=1)
        self.branch2_0_bn = nn.InstanceNorm3d(cos[1])
        self.branch2_0_relu = nn.LeakyReLU()
        self.branch2_1 = nn.Conv3d(cos[1], cos[1], (3,3,1), stride=(1,1,1), padding=(1,1,0))
        self.branch2_1_bn = nn.InstanceNorm3d(cos[1])
        self.branch2_1_relu = nn.LeakyReLU()
        self.branch2_2 = nn.Conv3d(cos[1], cos[1], (1,1,3), stride=(1,1,1), padding=(0,0,1))
        self.branch2_2_bn = nn.InstanceNorm3d(cos[1])
        self.branch2_2_relu = nn.LeakyReLU()

        self.branch3_0 = nn.Conv3d(cin, cos[2], 1, stride=1)
        self.branch3_0_bn = nn.InstanceNorm3d(cos[2])
        self.branch3_0_relu = nn.LeakyReLU()
        self.branch3_1 = nn.Conv3d(cos[2], cos[2], (5,5,1), stride=(1,1,1), padding=(2,2,0))
        self.branch3_1_bn = nn.InstanceNorm3d(cos[2])
        self.branch3_1_relu = nn.LeakyReLU()
        self.branch3_2 = nn.Conv3d(cos[2], cos[2], (1,1,5), stride=(1,1,1), padding=(0,0,2))
        self.branch3_2_bn = nn.InstanceNorm3d(cos[2])
        self.branch3_2_relu = nn.LeakyReLU()

        self.branch4_0 = nn.Conv3d(cin, cos[3], 1, stride=1)
        self.branch4_0_bn = nn.InstanceNorm3d(cos[3])
        self.branch4_0_relu = nn.LeakyReLU()
        self.branch4_1 = nn.Conv3d(cos[3], cos[3], (7,7,1), stride=(1,1,1), padding=(3,3,0))
        self.branch4_1_bn = nn.InstanceNorm3d(cos[3])
        self.branch4_1_relu = nn.LeakyReLU()
        self.branch4_2 = nn.Conv3d(cos[3], cos[3], (1,1,7), stride=(1,1,1), padding=(0,0,3))
        self.branch4_2_bn = nn.InstanceNorm3d(cos[3])
        self.branch4_2_relu = nn.LeakyReLU()

        self.convend=nn.Conv3d(cin,co,1,stride=1)
        self.relu_end = nn.LeakyReLU()

    def forward(self,x):

        branch1=self.branch1_relu(self.branch1_bn(self.branch1(x)))
        branch2=self.branch2_2_relu(self.branch2_2_bn(self.branch2_2(
            self.branch2_1_relu(self.branch2_1_bn(self.branch2_1(
                self.branch2_0_relu(self.branch2_0_bn(self.branch2_0(x)))))))))
        branch3=self.branch3_2_relu(self.branch3_2_bn(self.branch3_2(
            self.branch3_1_relu(self.branch3_1_bn(self.branch3_1(
                self.branch3_0_relu(self.branch3_0_bn(self.branch3_0(x)))))))))
        branch4=self.branch4_2_relu(self.branch4_2_bn(self.branch4_2(
            self.branch4_1_relu(self.branch4_1_bn(self.branch4_1(
                self.branch4_0_relu(self.branch4_0_bn(self.branch4_0(x)))))))))
        result=torch.cat((branch1,branch2,branch3,branch4),1)
        result=self.convend(result)
        return self.relu_end(x+result)



class TriSegNet(nn.Module):
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(TriSegNet, self).__init__()
        self.num_classes = num_classes

        self.leakyrelu = nn.LeakyReLU()

        self.conv1   = con3dblock(4,32,kernel_size=3,stride=1,padding=1)
        self.conv2   = con3dblock(32,64,kernel_size=3,stride=2,padding=1)
        self.conv2_2 = Isomorphism_incept(64,64)
        self.conv3 = con3dblock(64,128,kernel_size=3,stride=2,padding=1) 
        self.conv3_2 = Isomorphism_incept(128,128)
        self.conv4 = con3dblock(128,256,kernel_size=3,stride=2,padding=1)
        self.conv4_2 = Isomorphism_incept(256,256)

        self.block2_1 = Block(64,64,3,1,start_with_relu=False,grow_first=True) 

        self.block3_1 = Block(64,128,2,2,start_with_relu=True,grow_first=True)
        self.block3_2 = Block(128,128,3,1,start_with_relu=True,grow_first=True) 

        self.block4_1 = Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block4_2 = Block(256,256,3,1,start_with_relu=True,grow_first=True) 


        self.block4_se = SEModule(256) 
        self.block4_se_2 = SEModule_1(256) 

        self.fusion_conv_16 = con3dblock(512,128,kernel_size=3,stride=1,padding=1)

        self.fusion_conv_se_16 = SEModule_mul_add(128)

        self.dense_16_32 = con3dblock(128,64,kernel_size=3,stride=1,padding=1)
        self.dense_16_32_2 = con3dblock(128,32,kernel_size=3,stride=1,padding=1)

        self.fusion_conv_32 = con3dblock(192,64,kernel_size=3,stride=1,padding=1)

        self.dense_32_64 = con3dblock(64,32,kernel_size=3,stride=1,padding=1)

        self.fusion_conv_64 = con3dblock(128,32,kernel_size=3,stride=1,padding=1)

        self.upsample_2 = nn.Upsample(scale_factor=2,mode='trilinear')

        self.upsample_4 = nn.Upsample(scale_factor=4,mode='trilinear')

        self.refine = RefinementBlock(32,32)

        self.end = nn.Conv3d(32,5,1)
        
        self.out = nn.Softmax()


    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x_2_spatial = self.conv2_2(x)
        x_3_spatial = self.conv3(x_2_spatial) 
        x_3_spatial = self.conv3_2(x_3_spatial) 
        x_4_spatial = self.conv4(x_3_spatial)
        x_4_spatial = self.conv4_2(x_4_spatial)

        context_2 = self.block2_1(x)    
        context_3 = self.block3_1(context_2)
        context_3 = self.block3_2(context_3) 

        context_4 = self.block4_1(context_3)
        context_4 = self.block4_2(context_4) 


        context_4 = self.block4_se(self.leakyrelu(context_4)) 
        context_4 = self.block4_se_2(context_4)

        localization_net_4 = torch.cat((x_4_spatial, context_4),1) 

        localization_net_4 = self.fusion_conv_16(localization_net_4) 
        localization_net_4 = self.fusion_conv_se_16(localization_net_4)  

        localization_net_4 = self.upsample_2(localization_net_4) 
        localization_net_4_1 = self.dense_16_32(localization_net_4) 

        localization_net_4 = self.upsample_2(localization_net_4)
        localization_net_4_2 = self.dense_16_32_2(localization_net_4) 

        localization_net_3 = torch.cat((x_3_spatial, localization_net_4_1),1)

        localization_net_3 = self.fusion_conv_32(localization_net_3) 
        localization_net_3 = self.upsample_2(localization_net_3) 
        localization_net_3_1 = self.dense_32_64(localization_net_3) 

        localization_net_2 = torch.cat((x_2_spatial, localization_net_3_1,localization_net_4_2),1) 
        localization_net_2 = self.fusion_conv_64(localization_net_2)
        
        context_net = self.upsample_2(localization_net_2)
        context_net = self.refine(context_net)
        context_net = self.end(context_net)
        context_net = self.out(context_net)

        return context_net


        






        






