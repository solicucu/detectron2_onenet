# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
# 很多对bboxes 的operations, nms, box_are, box_iou，generalized_box_iou ...
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

    #x(Tensor): [[x1,y1,x2,y2]]
def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1) # 根据指定维度获取每列的数据
    b = [(x0 + x1) / 2, (y0 + y1) / 2, #求出中点，
         (x1 - x0), (y1 - y0)]   # 求 w,h
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    # 返回每个boxes 的area
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    # 左上角取最大的 
    # boxes1: [N,4], boxes1[:,None] -> [N,1,4] 多了1维
    # troch.max(shape[N,1,2], shape[M,2]), 形状不一致，怎么比
    # 传入两个张量，就会调用的是
    # torch.maximum(input, other,...)
    # Computes the element-wise maximum of input and other.
    # 对于维度数不一样，会产生一个矩阵。。
    # 相当于先 [1,2] 跟 [M,2] 比较，得到一个 [M,2], 一共有N个[1,2] 所以为[N,M,2]

    # 对于 [[m]], [[m],...,[m]] 这样是可以元素级比较的，即第一个的第一维要么为1(可以通过repeated) 重复比较，要么跟第二个一样，其他不可行
    # 这样一来，boxes1 的每个元素可以和boxes2的每个元素进行计算，正是我们想要的
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # 左下角取最小的
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2] # 求 w,h of intersection
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M] # are of intersection

    union = area1[:, None] + area2 - inter # 并集

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    # boxes1: [query, 4]
    # boxes2: [numi, 4]
    # 判断右下角都大于左上角
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2]) # 取最小的 # [query, numi, 2]
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:]) # 取最大的 # [query, numi, 2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1] # 求正好包住他们的面积

    # Lgiou = 1-iou + (area-u)/area
    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
