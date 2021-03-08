#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OneNet Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from detectron2.modeling.poolers import ROIPooler, cat
from detectron2.structures import Boxes
from .deconv import CenternetDeconv

class Head(nn.Module):

    def __init__(self, cfg, backbone_shape=[2048, 1024, 512, 256]):
        super().__init__()
        
        # Build heads.
        num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        # deconv_channel:[2048, 256, 128, 64]
        d_model = cfg.MODEL.OneNet.DECONV_CHANNEL[-1]
        activation = cfg.MODEL.OneNet.ACTIVATION # relu

        self.deconv = CenternetDeconv(cfg, backbone_shape)
        
        self.num_classes = num_classes
        self.d_model = d_model
        # self.num_classes = num_classes # 重复了
        self.activation = _get_activation_fn(activation)
        # 注意三个都没有去掉偏置值
        self.feat1 = nn.Conv2d(self.d_model, self.d_model, kernel_size=3, stride=1, padding=1)
        self.cls_score = nn.Conv2d(d_model, num_classes, kernel_size=3, stride=1, padding=1)
        self.ltrb_pred = nn.Conv2d(d_model, 4, kernel_size=3, stride=1, padding=1)        
        
        # Init parameters.
        prior_prob = cfg.MODEL.OneNet.PRIOR_PROB
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)
        self._reset_parameters()

    def _reset_parameters(self):
        # init all parameters.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss.
            # 给分类的模块初始化一个偏置值
            if p.shape[-1] == self.num_classes:
                nn.init.constant_(p, self.bias_value)
    
    def forward(self, features_list):
        # features_list[res2, res3, res4, re5] -> new_res2 
        features = self.deconv(features_list)
        # 为什么加了个[None]，作用是在dim=0 多增加一个维度
        # 返回了个原图坐标位置
        # locations 不用计算梯度，计算了每个pixel in fmap 对应原图的位置
        locations = self.locations(features)[None] 
        # locations:[1, 2, h, w]      

        feat = self.activation(self.feat1(features))
    
        class_logits = self.cls_score(feat)
        # use F.relu, so pred_ltrb is a positive number
        # ltrb: left top, right bottom 的offset
        pred_ltrb = F.relu(self.ltrb_pred(feat))
        # 返回调整后的bboxes 位置信息
        pred_bboxes = self.apply_ltrb(locations, pred_ltrb)

        return class_logits, pred_bboxes
    
    def apply_ltrb(self, locations, pred_ltrb): 
        """
        :param locations:  (1, 2, H, W)
        :param pred_ltrb:  (N, 4, H, W) 
        """
        # all info same as pred_ltrb including dtype, device ...
        pred_boxes = torch.zeros_like(pred_ltrb)
        # 为什么左上角坐标是用减，右下角坐标用+
        #  pred_ltrb is positive number because of processing by relu
        # 因为locations 是中心点， 所以预测的pred_ltrb 是对中心点的偏移度
        pred_boxes[:,0,:,:] = locations[:,0,:,:] - pred_ltrb[:,0,:,:]  # x1
        pred_boxes[:,1,:,:] = locations[:,1,:,:] - pred_ltrb[:,1,:,:]  # y1
        pred_boxes[:,2,:,:] = locations[:,0,:,:] + pred_ltrb[:,2,:,:]  # x2
        pred_boxes[:,3,:,:] = locations[:,1,:,:] + pred_ltrb[:,3,:,:]  # y2

        return pred_boxes    
    
    @torch.no_grad()
    def locations(self, features, stride=4):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (2, H, W)
        """

        h, w = features.size()[-2:]
        device = features.device
        # 对应原来的尺寸，res2 是stride = 4
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        # torch.meshgird 以shifts_y 的维度h*为行，shifts_x 的维度w*为列
        # 得到两个矩阵 [h*, w*], 第一个是通过行重复扩展，第二个是通过列重复扩展
        # y:[2,4,6] x:[2,4]
        """
        [[2, 2],
         [4, 4],
         [6, 6]]

        [[2, 4],
         [2, 4],
         [2, 4]]
        这样的到的两个值组合就是对应的坐标,例如 （2,2) 是第一个位置
        """
        # H * W 
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1) # 变为只剩一个维度[H*W]
        shift_y = shift_y.reshape(-1) # 
        # stack 会增多一个维度，dim = 1, 说明增加的维度在第一维
        # [1,2,3] [2,3,4] -> [[1,2],[2,3],[3,4]]
        # 为什么 + stride // 2 呢？# 加一半表示中心点？
        # 答：加 stride 的一半，就对应了视野中心
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2            
        # permute 调整一下维度，[h, w, 2] -> [2, h, w]
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        
        return locations


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
