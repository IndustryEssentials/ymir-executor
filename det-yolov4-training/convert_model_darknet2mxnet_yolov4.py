# -*- encoding:utf-8 -*-
import logging
from mxnet.gluon import nn
import numpy as np
import mxnet.ndarray as nd
import mxnet as mx
import pdb
import argparse
import os
from datetime import datetime 


def ConvBNBlock(channels,kernel_size,strides,pad=0,use_bias=False,groups=1,activation='linear'):
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(channels,kernel_size,strides,pad,groups=groups,use_bias=use_bias))
    if not use_bias:
        blk.add(nn.BatchNorm(in_channels=channels))
    if activation == 'linear':
        return blk
    elif activation == "leaky":
        blk.add(nn.LeakyReLU(0.1))
    elif activation == "relu":
        blk.add(nn.Activation('relu'))
    elif activation == "swish":
        blk.add(nn.Swish())
    elif activation == "logistic":
        blk.add(nn.Activation('sigmoid'))
    else:
        print('%s is an unsupported activation!!'%s(activation))
        exit()
    return blk

class UpSampleBlock(nn.HybridBlock):
    def __init__(self, scale, sample_type="nearest"):
        super(UpSampleBlock,self).__init__()
        self.scale = scale
        self.sample_type = sample_type
    def hybrid_forward(self,F,x,*args,**kwargs):
        return F.UpSampling(x,scale=self.scale,sample_type=self.sample_type)

class TransformBlock(nn.HybridBlock):
    def __init__(self,num_classes, num_boxes, feature_size):
        super(TransformBlock, self).__init__()
        self.bbox_attrs = 5 + num_classes
        self.num_boxes = num_boxes
        self.feature_size = feature_size

    def hybrid_forward(self,F,x,*args,**kwargs):
        x = F.transpose(x.reshape((0,self.bbox_attrs * self.num_boxes, self.feature_size * self.feature_size)),(0,2,1)).reshape((0, self.feature_size * self.feature_size * self.num_boxes, self.bbox_attrs))
        xy_pred = F.sigmoid(x.slice_axis(begin=0,end=2,axis=-1))
        wh_pred = x.slice_axis(begin=2,end=4,axis=-1)
        score_pred = F.sigmoid(x.slice_axis(begin=4, end=5, axis=-1))
        cls_pred = F.sigmoid(x.slice_axis(begin=5,end=None, axis=-1))
        return F.concat(xy_pred, wh_pred, score_pred, cls_pred, dim=-1)

class Darknet(nn.HybridBlock):
    def __init__(self, num_classes=1, input_dim=608):
        super(Darknet,self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.conv_bn_block_0 = ConvBNBlock(32,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_1 = ConvBNBlock(64,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_2 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_4 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_5 = ConvBNBlock(32,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_6 = ConvBNBlock(64,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_8 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_10 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_11 = ConvBNBlock(128,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_12 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_14 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_15 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_16 = ConvBNBlock(64,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_18 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_19 = ConvBNBlock(64,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_21 = ConvBNBlock(64,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_23 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_24 = ConvBNBlock(256,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_25 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_27 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_28 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_29 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_31 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_32 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_34 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_35 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_37 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_38 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_40 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_41 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_43 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_44 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_46 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_47 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_49 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_50 = ConvBNBlock(128,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_52 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_54 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_55 = ConvBNBlock(512,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_56 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_58 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_59 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_60 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_62 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_63 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_65 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_66 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_68 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_69 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_71 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_72 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_74 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_75 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_77 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_78 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_80 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_81 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_83 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_85 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_86 = ConvBNBlock(1024,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_87 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_89 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_90 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_91 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_93 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_94 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_96 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_97 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_99 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_100 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_102 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_104 = ConvBNBlock(1024,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_105 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_106 = ConvBNBlock(1024,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_107 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.max_pool_108 = nn.MaxPool2D(5, 1, ceil_mode=True, padding=5//2)
        self.max_pool_110 = nn.MaxPool2D(9, 1, ceil_mode=True, padding=9//2)
        self.max_pool_112 = nn.MaxPool2D(13, 1, ceil_mode=True, padding=13//2)
        self.conv_bn_block_114 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_115 = ConvBNBlock(1024,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_116 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_117 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.upsample_block_118 = UpSampleBlock(scale=2)
        self.conv_bn_block_120 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_122 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_123 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_124 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_125 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_126 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_127 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.upsample_block_128 = UpSampleBlock(scale=2)
        self.conv_bn_block_130 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_132 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_133 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_134 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_135 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_136 = ConvBNBlock(128,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_137 = ConvBNBlock(256,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_138 = ConvBNBlock((num_classes+5)*3,1,1,0,use_bias=True,groups=1,activation='linear')
        self.transform_0 = TransformBlock(num_classes,3,int(input_dim/32) * 2 * 2)
        self.conv_bn_block_141 = ConvBNBlock(256,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_143 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_144 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_145 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_146 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_147 = ConvBNBlock(256,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_148 = ConvBNBlock(512,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_149 = ConvBNBlock((num_classes+5)*3,1,1,0,use_bias=True,groups=1,activation='linear')
        self.transform_1 = TransformBlock(num_classes,3,int(input_dim/32) * 2)
        self.conv_bn_block_152 = ConvBNBlock(512,3,2,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_154 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_155 = ConvBNBlock(1024,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_156 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_157 = ConvBNBlock(1024,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_158 = ConvBNBlock(512,1,1,0,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_159 = ConvBNBlock(1024,3,1,1,use_bias=False,groups=1,activation='leaky')
        self.conv_bn_block_160 = ConvBNBlock((num_classes+5)*3,1,1,0,use_bias=True,groups=1,activation='linear')
        self.transform_2 = TransformBlock(num_classes,3,int(input_dim/32))

    def hybrid_forward(self,F,x,*args,**kwargs):
        x0 = self.conv_bn_block_0(x)
        x1 = self.conv_bn_block_1(x0)
        x2 = self.conv_bn_block_2(x1)
        x3 = x1
        x4 = self.conv_bn_block_4(x3)
        x5 = self.conv_bn_block_5(x4)
        x6 = self.conv_bn_block_6(x5)
        x7 = x6 +x4
        x8 = self.conv_bn_block_8(x7)
        x9 = F.concat(x8,x2,dim=1)
        x10 = self.conv_bn_block_10(x9)
        x11 = self.conv_bn_block_11(x10)
        x12 = self.conv_bn_block_12(x11)
        x13 = x11
        x14 = self.conv_bn_block_14(x13)
        x15 = self.conv_bn_block_15(x14)
        x16 = self.conv_bn_block_16(x15)
        x17 = x16 +x14
        x18 = self.conv_bn_block_18(x17)
        x19 = self.conv_bn_block_19(x18)
        x20 = x19 +x17
        x21 = self.conv_bn_block_21(x20)
        x22 = F.concat(x21,x12,dim=1)
        x23 = self.conv_bn_block_23(x22)
        x24 = self.conv_bn_block_24(x23)
        x25 = self.conv_bn_block_25(x24)
        x26 = x24
        x27 = self.conv_bn_block_27(x26)
        x28 = self.conv_bn_block_28(x27)
        x29 = self.conv_bn_block_29(x28)
        x30 = x29 +x27
        x31 = self.conv_bn_block_31(x30)
        x32 = self.conv_bn_block_32(x31)
        x33 = x32 +x30
        x34 = self.conv_bn_block_34(x33)
        x35 = self.conv_bn_block_35(x34)
        x36 = x35 +x33
        x37 = self.conv_bn_block_37(x36)
        x38 = self.conv_bn_block_38(x37)
        x39 = x38 +x36
        x40 = self.conv_bn_block_40(x39)
        x41 = self.conv_bn_block_41(x40)
        x42 = x41 +x39
        x43 = self.conv_bn_block_43(x42)
        x44 = self.conv_bn_block_44(x43)
        x45 = x44 +x42
        x46 = self.conv_bn_block_46(x45)
        x47 = self.conv_bn_block_47(x46)
        x48 = x47 +x45
        x49 = self.conv_bn_block_49(x48)
        x50 = self.conv_bn_block_50(x49)
        x51 = x50 +x48
        x52 = self.conv_bn_block_52(x51)
        x53 = F.concat(x52,x25,dim=1)
        x54 = self.conv_bn_block_54(x53)
        x55 = self.conv_bn_block_55(x54)
        x56 = self.conv_bn_block_56(x55)
        x57 = x55
        x58 = self.conv_bn_block_58(x57)
        x59 = self.conv_bn_block_59(x58)
        x60 = self.conv_bn_block_60(x59)
        x61 = x60 +x58
        x62 = self.conv_bn_block_62(x61)
        x63 = self.conv_bn_block_63(x62)
        x64 = x63 +x61
        x65 = self.conv_bn_block_65(x64)
        x66 = self.conv_bn_block_66(x65)
        x67 = x66 +x64
        x68 = self.conv_bn_block_68(x67)
        x69 = self.conv_bn_block_69(x68)
        x70 = x69 +x67
        x71 = self.conv_bn_block_71(x70)
        x72 = self.conv_bn_block_72(x71)
        x73 = x72 +x70
        x74 = self.conv_bn_block_74(x73)
        x75 = self.conv_bn_block_75(x74)
        x76 = x75 +x73
        x77 = self.conv_bn_block_77(x76)
        x78 = self.conv_bn_block_78(x77)
        x79 = x78 +x76
        x80 = self.conv_bn_block_80(x79)
        x81 = self.conv_bn_block_81(x80)
        x82 = x81 +x79
        x83 = self.conv_bn_block_83(x82)
        x84 = F.concat(x83,x56,dim=1)
        x85 = self.conv_bn_block_85(x84)
        x86 = self.conv_bn_block_86(x85)
        x87 = self.conv_bn_block_87(x86)
        x88 = x86
        x89 = self.conv_bn_block_89(x88)
        x90 = self.conv_bn_block_90(x89)
        x91 = self.conv_bn_block_91(x90)
        x92 = x91 +x89
        x93 = self.conv_bn_block_93(x92)
        x94 = self.conv_bn_block_94(x93)
        x95 = x94 +x92
        x96 = self.conv_bn_block_96(x95)
        x97 = self.conv_bn_block_97(x96)
        x98 = x97 +x95
        x99 = self.conv_bn_block_99(x98)
        x100 = self.conv_bn_block_100(x99)
        x101 = x100 +x98
        x102 = self.conv_bn_block_102(x101)
        x103 = F.concat(x102,x87,dim=1)
        x104 = self.conv_bn_block_104(x103)
        x105 = self.conv_bn_block_105(x104)
        x106 = self.conv_bn_block_106(x105)
        x107 = self.conv_bn_block_107(x106)
        x108 = self.max_pool_108(x107)
        x109 = x107
        x110 = self.max_pool_110(x109)
        x111 = x107
        x112 = self.max_pool_112(x111)
        x113 = F.concat(x112,x110,x108,x107,dim=1)
        x114 = self.conv_bn_block_114(x113)
        x115 = self.conv_bn_block_115(x114)
        x116 = self.conv_bn_block_116(x115)
        x117 = self.conv_bn_block_117(x116)
        x118 = self.upsample_block_118(x117)
        x119 = x85
        x120 = self.conv_bn_block_120(x119)
        x121 = F.concat(x120,x118,dim=1)
        x122 = self.conv_bn_block_122(x121)
        x123 = self.conv_bn_block_123(x122)
        x124 = self.conv_bn_block_124(x123)
        x125 = self.conv_bn_block_125(x124)
        x126 = self.conv_bn_block_126(x125)
        x127 = self.conv_bn_block_127(x126)
        x128 = self.upsample_block_128(x127)
        x129 = x54
        x130 = self.conv_bn_block_130(x129)
        x131 = F.concat(x130,x128,dim=1)
        x132 = self.conv_bn_block_132(x131)
        x133 = self.conv_bn_block_133(x132)
        x134 = self.conv_bn_block_134(x133)
        x135 = self.conv_bn_block_135(x134)
        x136 = self.conv_bn_block_136(x135)
        x137 = self.conv_bn_block_137(x136)
        x138 = self.conv_bn_block_138(x137)
        x139 = self.transform_0(x138)
        x140 = x136
        x141 = self.conv_bn_block_141(x140)
        x142 = F.concat(x141,x126,dim=1)
        x143 = self.conv_bn_block_143(x142)
        x144 = self.conv_bn_block_144(x143)
        x145 = self.conv_bn_block_145(x144)
        x146 = self.conv_bn_block_146(x145)
        x147 = self.conv_bn_block_147(x146)
        x148 = self.conv_bn_block_148(x147)
        x149 = self.conv_bn_block_149(x148)
        x150 = self.transform_1(x149)
        x151 = x147
        x152 = self.conv_bn_block_152(x151)
        x153 = F.concat(x152,x116,dim=1)
        x154 = self.conv_bn_block_154(x153)
        x155 = self.conv_bn_block_155(x154)
        x156 = self.conv_bn_block_156(x155)
        x157 = self.conv_bn_block_157(x156)
        x158 = self.conv_bn_block_158(x157)
        x159 = self.conv_bn_block_159(x158)
        x160 = self.conv_bn_block_160(x159)
        x161 = self.transform_2(x160)
        # detections = [x139, x150, x161]
        detections = F.concat(x161, x150, x139, dim=1)
        return detections
    def load_weights(self, weightfile, fine_tune=False):
        # open the weights file
        fp = open(weightfile, "rb")
        # the first 5 values are header information
        # 1. Majior version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5 Images seen by the network (during training)
        self.header = nd.array(np.fromfile(fp, dtype=np.int32, count=5))
        self.seen = self.header[3]
        weights = nd.array(np.fromfile(fp, dtype=np.float32))
        ptr = 0
        
        def set_data(model, ptr):
            conv = model[0]
            if len(model) > 1 and "batchnorm" in model[1].name:
                bn = model[1]
                # Get the number of weights of Batch Norm Layer
                num_bn_beta = self.numel(bn.beta.shape)
                # Load the weights
                bn_beta = weights[ptr:ptr+num_bn_beta]
                ptr += num_bn_beta
                bn_gamma = weights[ptr:ptr+num_bn_beta]
                ptr += num_bn_beta
                bn_running_mean = weights[ptr:ptr+num_bn_beta]
                ptr += num_bn_beta
                bn_running_var = weights[ptr:ptr+num_bn_beta]
                ptr += num_bn_beta
                
                # Cast the loaded weights into dims of model weights
                bn_beta = bn_beta.reshape(bn.beta.shape)
                bn_gamma = bn_gamma.reshape(bn.gamma.shape)
                bn_running_mean = bn_running_mean.reshape(bn.running_mean.shape)
                bn_running_var = bn_running_var.reshape(bn.running_var.shape)
                
                bn.gamma.set_data(bn_gamma)
                bn.beta.set_data(bn_beta)
                bn.running_mean.set_data(bn_running_mean)
                bn.running_var.set_data(bn_running_var)
            else:
                num_biases = self.numel(conv.bias.shape)
                conv_biases = weights[ptr:ptr+num_biases]
                ptr = ptr + num_biases
                conv_biases = conv_biases.reshape(conv.bias.shape)
                conv.bias.set_data(conv_biases)
            num_weights = self.numel(conv.weight.shape)
            conv_weights = weights[ptr:ptr +num_weights]
            ptr = ptr +num_weights
            conv_weights = conv_weights.reshape(conv.weight.shape)
            conv.weight.set_data(conv_weights)
            return ptr
        modules = self._children
        for block_name in modules:
            if fine_tune:
                if block_name.find("81") != -1:
                    ptr = 56629087
                    continue
                elif block_name.find("93") != -1:
                    ptr = 60898910
                    continue
                elif block_name.find("105") != -1:
                    continue
            module = modules.get(block_name)
            if isinstance(module, nn.HybridSequential):
                ptr = set_data(module, ptr)
        if ptr != len(weights):
            raise ValueError("convert model fail: num of ptr should be equal to len of weights")

    def numel(self,x):
        if isinstance(x, nd.NDArray):
            x = x.asnumpy()
        return np.prod(x)

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description='input class info and netwrok input size')
    parser.add_argument('--num_of_classes', type=int, default=1, help='num of class info of netwrok')
    parser.add_argument('--input_h', type=int, default=608, help='input height of netwrok')
    parser.add_argument('--input_w', type=int, default=608, help='input width of netwrok')
    parser.add_argument('--load_param_name', type=str, help='convert model weight file')
    args = parser.parse_args()

    return args


def run(num_of_classes: int, input_h: int, input_w: int, load_param_name: str, export_dir: str) -> None:
    if not os.path.isfile(load_param_name):
        return

    net = Darknet(num_of_classes, input_dim=input_h)
    net.initialize()
    X = mx.nd.ones(shape=(1, 3, input_h, input_w)) 
    Y = net(X)
    net.load_weights(load_param_name)
    net.hybridize()
    Y = net(X)
    net.export(os.path.join(export_dir, 'model'), 0)


if __name__ == "__main__":
    args = parse_args()

    run(num_of_classes=args.num_of_classes,
        input_h=args.input_h,
        input_w=args.input_w,
        load_param_name=args.load_param_name,
        export_dir="/out/models")
