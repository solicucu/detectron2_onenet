#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from detectron2.layers import ShapeSpec,batched_nms
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, MinCostMatcher
from .head import Head
from .util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

__all__ = ["OneNet"]

# 调用detectron2.modeling.build 里面的META_ARCH_REGISTRAY, 注册新的结构，可以方便的根据名称调用
@META_ARCH_REGISTRY.register() # registry 登记处，register 登记注册
class OneNet(nn.Module):
    """
    Implement OneNet
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        
        self.nms = cfg.MODEL.OneNet.NMS # 是否使用非极大值抑制，默认是false
        # [res2, res3, res4, res5]
        self.in_features = cfg.MODEL.OneNet.IN_FEATURES
        # num_classes:80
        self.num_classes = cfg.MODEL.OneNet.NUM_CLASSES
        # 100 is the limit for coco
        self.num_boxes = cfg.TEST.DETECTIONS_PER_IMAGE

        # Build Backbone.
        # use resnet50 as backbone
        self.backbone = build_backbone(cfg)
        # default is 0
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Head.
        # return a class_logits and pred_boxes
        # backbone_shape: dict['res{k}':ShapeSpec(channel,...,stride)]
        # 描述每个feature map 的channel, cur_stride 的信息
        self.head = Head(cfg=cfg, backbone_shape=self.backbone.output_shape())

        # Loss parameters:
        # 2.0
        class_weight = cfg.MODEL.OneNet.CLASS_WEIGHT
        # 2.0
        giou_weight = cfg.MODEL.OneNet.GIOU_WEIGHT
        # 5.0 distance between center point
        l1_weight = cfg.MODEL.OneNet.L1_WEIGHT

        # Build Criterion.
        matcher = MinCostMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_giou=giou_weight)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      losses=losses)
        # pixel_mean:list[3]
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        # x:C x H x W, normalize the pixel
        self.to(self.device)
     """
        data format for dectection:
        list = [dict]
        dict = {
         {
            "file_name":图片完全路径
            "height": 原始图片高
            "width"：原始图片宽
            "image_id"：图片id
            "image": tensor(N,H,W) # gt_boxes 用Boxes 封装，里面是一个tensor with shape(num,4)
            "instances":Instances(a class with attr, gt_boxes(Boxes),gt_classes(list[int]),.image_size)
             # 此.image_size 是经过transfrom 后的
        }
     """
    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        # images(ImageList), images_whwh(Tensor) with [w,h,w,h] for what?
        images, images_whwh = self.preprocess_image(batched_inputs)
        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        # return a dict = {"res2":fmap,..."res5":}
        src = self.backbone(images.tensor)
        features = list()        
        # self.in_features: ["res2", "res3", "res4", "res5"]
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        # Cls & Reg Prediction.
        # outputs_class: with shape (N, num_class, H/4, W/4)
        # outputs_coord(Tensor): with shape (N, 4, H/4, W/4)
        # coord 是已经调整跟实际
        outputs_class, outputs_coord = self.head(features)
        
        output = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        """
        output:{
            "pred_logits":with shape (N, num_class, H/4, W/4)
            "pred_boxes": with shape (N, 4, H/4, W/4) boxes 已经中心原图化
        }
        """
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            """ targets = [dict]
            每个dict 是一个图片的
            dict = {
                label:(tensor) [num] 类别
                boxes:(tensor) [num, 4] (, cx,cy,w,h) 归一化后的，boxes format
                boxes_xyxy: (tensor), [num, 4] # 原来的boxes
                image_size_xyxy:(tensor) [4] # [w,h,w,h]
                image_size_xyxy_tgt:(tensor) [num,4]  item 同上
                area: (tensor), [num] # 计算每个boxes 的面积
                all is *.to(self.device)
            }
            """
            # 2021.2.26
            # loss_dict:{"loss_ce":, "loss_giou":, "loss_bbox":} loss_bbox:l1_loss(x1,y1,x2,y2) already normalized
            loss_dict = self.criterion(output, targets)
            # 对loss 加权
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            
            return loss_dict

        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            #ret: list[Instances] with shape [batch_size]
            """
            if without nms, shape is [topk] for each attr
            Instances:
                .pred_boxes(Boxes): Boxes.tensor with shape[topk]
                .scores(Tensor): shape[topk]
                .pred_classes(Tensor): prediction of class id
            """
            # images.image_sizes is the size after data tranformation
            results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                # 获取图片h,w 信息, 此处的尺寸是原始图片的尺寸
                # image_size 是经过transform 后的
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                # 对 bbox 缩放对应于原始图片的尺寸
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            
            return processed_results

    def prepare_targets(self, targets):
        # targets: a list of Instances(class) with attr: gt_classes, gt_boxes, _image_size
        new_targets = []
        for targets_per_image in targets:
            target = {}
            h, w = targets_per_image.image_size  #获取图片的大小
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            
            gt_classes = targets_per_image.gt_classes
            # gt_boxes.tensor = [x1,y1,x2,y2], 归一化
            # gt_boxes.tensor 才是boxes, 因为gt_boxes 是一个Boxes 类
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes) # 转化为中心(cx, cy, w, h) 模式
            # num instances
            target["labels"] = gt_classes.to(self.device) # class label [num]
            target["boxes"] = gt_boxes.to(self.device) # [num, 4]
            # orgin boxes
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device) #[num, 4]
            target["image_size_xyxy"] = image_size_xyxy.to(self.device) # [4]
            # repeat(num,1) 重复num 行， 1列
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device) # [num, 4]
            # type(gt_boxes) = Boxes,有area 方法
            target["area"] = targets_per_image.gt_boxes.area().to(self.device) # [num]
            new_targets.append(target) 

        return new_targets

    def inference(self, _box_cls, _box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape   (batch_size, K, H, W).
            box_pred (Tensor): tensors of shape (batch_size, 4, H, W).
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        # -> [batch_size, k, query]
        box_cls = _box_cls.flatten(2) # start dim = 2, 
        box_pred = _box_pred.flatten(2)
        # image_sizes:[batch_size, 2]
        assert len(box_cls) == len(image_sizes)
        results = []
        # [N, num_class, query]
        scores = torch.sigmoid(box_cls)

        for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, box_pred, image_sizes
        )):
            # 创建一个实例， image_size 是transform 之后的尺寸
            result = Instances(image_size)
            
            # refer to https://github.com/FateScript/CenterNet-better
            # k is 100 for onenet torch.topk(input, k, dim=None, largest=True, sorted=True)
            # 如果dim 不指定，就会设置dim=-1
            # scores_per_image: [num_class, query]
            # topk_score_cat[num_class, self.num_boxes] 这个数据没有意义(或者理解为每个类的topk)
            # 因为总的目标是求整个数组的topk, 先求行topk, 再求行topk 中的topk 就是总的topk 这个思路
            topk_score_cat, topk_inds_cat = torch.topk(scores_per_image, k=self.num_boxes)
            # 在行topk 中选真正的topk
            topk_score, topk_inds = torch.topk(topk_score_cat.reshape(-1), k=self.num_boxes)
            # 因为每个类选出了topk(100)个，那么ind // topk(self.num_boxes) 就是它的类别
            topk_clses = topk_inds // self.num_boxes
            # [topk]
            scores_per_image = topk_score
            labels_per_image = topk_clses
            
            # 类别topk
            # box_pred_per_image: [4, query]
            # 因为topk_inds_cat:[num_class, topk], topk in [0, query)
            # topk_box_cat: [4, num_class*topk], 也就是每个类选一次
            topk_box_cat = box_pred_per_image[:, topk_inds_cat.reshape(-1)]
            # 总的topk 
            # topk_box = [4, topk]
            topk_box = topk_box_cat[:, topk_inds]
            # box_pred_per_image:[topk, 4]
            box_pred_per_image = topk_box.transpose(0, 1)
            
            if self.nms:
                # 返回一个保留的index
                keep = batched_nms(box_pred_per_image, 
                                   scores_per_image, 
                                   labels_per_image, 
                                   0.5)
                box_pred_per_image = box_pred_per_image[keep]
                scores_per_image = scores_per_image[keep]
                labels_per_image = labels_per_image[keep]

            result.pred_boxes = Boxes(box_pred_per_image)
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]

#         images = ImageList.from_tensors(images, self.size_divisibility)
        # size_divisibility:32 因为backbone 下采样总步长为32
        # 因为原来每张图片可能大小不一，所以要统一pad,构建一个tensor
        # ImageList.tensor, ImageList.image_sizes 
        images = ImageList.from_tensors(images, 32)

        images_whwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwh.append(torch.tensor([w, h, w, h], dtype=torch.float32, device=self.device))
        images_whwh = torch.stack(images_whwh)

        return images, images_whwh
