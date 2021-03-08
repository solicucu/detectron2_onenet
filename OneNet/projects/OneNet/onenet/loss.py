#
# Modified by Peize Sun
# Contact: sunpeize@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
OneNet model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
from fvcore.nn import sigmoid_focal_loss_jit

from .util import box_ops
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from .util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou

from scipy.optimize import linear_sum_assignment


class SetCriterion(nn.Module):
    """ This class computes the loss for OneNet.
    The process happens in two steps: # 匈牙利分配？？？
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, cfg, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict # {"loss_ce":2, "loss_boxes":5, "loss_giou":2}
        self.losses = losses # ["labels", "boxes"]
        self.focal_loss_alpha = cfg.MODEL.OneNet.ALPHA # 0.25
        self.focal_loss_gamma = cfg.MODEL.OneNet.GAMMA # 2.0

    """
    Args:
        outputs:{
            "pred_logits":with shape (N, num_class, H/4, W/4)
            "pred_boxes": with shape (N, 4, H/4, W/4) boxes 已经中心原图化
        }
        targets: list[dict]
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
        indices:indcies(list[tuple(tensor,tensor)]) 每个tuple 是一个图片的匹配信息(src_ind, gt_ind)
        num_boxes: 平均每块gpu的boxes数 a number
    """
    # indice 最大的作用就是把 tg_class 分配到对应的pred_pos
    def loss_labels(self, outputs, targets, indices, num_boxes, log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        bs, k, h, w = src_logits.shape
        # [N, query, num_class] here query = w * h
        src_logits = src_logits.permute(0, 2, 3, 1).reshape(bs, h*w, k)
        # 就是把所有图片的boxes index 都串起来了
        # batch_idx 存储着batch 图片的index
        # return a batch_idx(Tensor[total_num_boxes]), src_idx(Tensor[total_num_boxes])
        idx = self._get_src_permutation_idx(indices)
        # 正常情况，J不就是 range(len(t['labels'])) 吗，多此一举？
        # 串联的每个box的class id 
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # torch.full 跟full_like 有点像，要指定一个形状，然后填充给定的值
        # shape[N, query] = 80，用80 初始是因为没有class_id 会是80
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # idx = (total_num_boxes, total_num_boxes)
        # idx = (i in [0,N), j in [0, query)
        # target_classes[idx], 返回的就是一个长度为 total_num_boxes 一维tensor
        # target_classes[idx] = data data 长度为total_num_boxes，就会把data里面的值一一赋给指定索引位置
        # 如此一来，每个被选中的pre_box 的位置赋予了真实的类别
        target_classes[idx] = target_classes_o

        # prepare one_hot target.
        # flatten(input, start_dim, end_dim)
        # origin: [N, query, num_class] -> [N*query, num_class]
        src_logits = src_logits.flatten(0, 1)

        # origin: [N, query] -> [N*query]
        target_classes = target_classes.flatten(0, 1)
        
        # 2021.2.27
        #as_tuple=True, 返回每个维度所对应的索引，因为这里target_classes 是1-dim 的，所以，ret只有一个
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        # labels:[N*query, num_class]
        # 转化为真正的0-1 矩阵
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_boxes
        losses = {'loss_ce': class_loss}


        if log:                         #xxx 感觉这个src_logits[idx] 错了，如果src_logits.flatten 没执行就还好
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    """
    Args:
        outputs:{
            "pred_logits":with shape (N, num_class, H/4, W/4)
            "pred_boxes": with shape (N, 4, H/4, W/4) boxes 已经中心原图化
        }
        targets: list[dict]
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
        indices:indcies(list[tuple(tensor,tensor)]) 每个tuple 是一个图片的匹配信息(src_ind, gt_ind)
        num_boxes: 平均每块gpu的boxes数 a number
    """
    # indices 最大的作用就是选出被选中的pred_box 跟tg_box 配对
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # batch_idx 存储着batch 图片的index
        # return a batch_idx(Tensor[total_num_boxes]), src_idx(Tensor[total_num_boxes])
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs['pred_boxes']
        bs, k, h, w = src_boxes.shape
        # -> [N, query, 4]
        src_boxes = src_boxes.permute(0, 2, 3, 1).reshape(bs, h*w, k)
        # 获取被选中的pred_boxes
        # [total_num_boxes, 4]
        src_boxes = src_boxes[idx]
        # 把所有的boxes 串在一起
        # target_boxes:[total_num_boxes, 4]
        target_boxes = torch.cat([t['boxes_xyxy'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        # box_ops.generalized_box_iou: return [total_num_boxes, total_num_boxes]
        # [src_N,tg_N], 只有对角线有用
        # 1 终于出现了
        # 此时计算的用的是原图化的w,h
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(src_boxes, target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # 归一化大小
        image_size = torch.cat([v["image_size_xyxy_tgt"] for v in targets])
        src_boxes_ = src_boxes / image_size
        target_boxes_ = target_boxes / image_size
        # src_boxes_: [toatal_num_boxes, 4]
        loss_bbox = F.l1_loss(src_boxes_, target_boxes_, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        return losses

        # 排列
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # torch.full_like create a tensor with shape as src.size and filled with value i.
        # [0,0,0,1,1,1,2,2,3,...] i 表示第i张图片
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # 把所有张量合并成一个列表了
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    """
        outputs:{
            "pred_logits":with shape (N, num_class, H/4, W/4)
            "pred_boxes": with shape (N, 4, H/4, W/4) boxes 已经中心原图化
        }
        targets = [dict]
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
    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # aux_ouput 是什么？
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # torch.no_grad()
        indices = self.matcher(outputs_without_aux, targets)
        """
        indcies(list[tuple(tensor,tensor)])
        每个tuple 是一个图片的匹配信息(src_ind, gt_ind)
        src_ind[i] and gt_ind[i] 是一个 pred-gt bbox pairs
        """

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 计算当前batch 总的实例数
        num_boxes = sum(len(t["labels"]) for t in targets)
        # 转化为tensor.to(output.device)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        # 小于等于1的取1，大于的不变
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        #["labels", "boxes"]
        for loss in self.losses:
            # labels: {"loss_ce":, maybe has "class_err":}
            # boxes: {"loss_box":}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        return losses



class MinCostMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cfg, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class # 2
        self.cost_bbox = cost_bbox # 5
        self.cost_giou = cost_giou # 
        # focal loss 用来干嘛的？
        # alpha:0.25
        self.focal_loss_alpha = cfg.MODEL.OneNet.ALPHA
        # gamma:2.0 
        self.focal_loss_gamma = cfg.MODEL.OneNet.GAMMA
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
    """
        output:{
            "pred_logits":with shape (N, num_class, H/4, W/4)
            "pred_boxes": with shape (N, 4, H/4, W/4) boxes 已经中心原图化
        }
        targets = [dict]
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
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        
        bs, k, h, w = outputs["pred_logits"].shape

        # We flatten to compute the cost matrices in a batch num_queries = w * h
        # [bs, num_clas, h, w] -> [bs, h, w, num_class] -> [batch_size, num_queries, num_class]
        batch_out_prob = outputs["pred_logits"].permute(0, 2, 3, 1).reshape(bs, h*w, k).sigmoid() # [batch_size, num_queries, num_classes]
        # [bs, 4, h, w] -> ...
        batch_out_bbox = outputs["pred_boxes"].permute(0, 2, 3, 1).reshape(bs, h*w, 4) # [batch_size, num_queries, 4]
        
        indices = []
        
        for i in range(bs):
            # 第i张图片总的目标数
            tgt_ids = targets[i]["labels"]
            # 如果没有目标
            if tgt_ids.shape[0] == 0:
                # .to(tensor)应该是想设置跟tensor 一样的device
                indices.append((torch.as_tensor([]).to(batch_out_prob), torch.as_tensor([]).to(batch_out_prob)))
                continue
            # 不同图片numi 是不一样的
            tgt_bbox = targets[i]["boxes_xyxy"] # 原来的boxes 格式 [numi, 4]
            out_prob = batch_out_prob[i]  # [query, num_class]
            out_bbox = batch_out_bbox[i]  # [query, 4]
            
            # ce:       positive         negtive
            # loss = -y*log(p) - (1-y)*log(1-p)
            # Compute the classification cost.
            alpha = self.focal_loss_alpha # 0.25 # 正样本的权重
            gamma = self.focal_loss_gamma # 2 # 难样本挖关注程度因子
            """
            focal loss: 为了解决正负样本极度不平衡问题，同时挖掘难样本的方法
            # p 为预测为正类的概率 (0,1)
            常规的cross_entropy: 如果正类(y=1) loss = -ylog(p), + 负类(y=0) loss = -(1-y)log(1-p)
            for focal loss:
                y in [1, -1] 令 pt = p if y==1 else 1-p
                那么 focal_loss = -log(pt)
                引入 
                alpha: alpha * pos_loss + (1-alpha) * neg_loss 可以调节正负样本的比重 (alpha in [0,1.])
                gamma: focal_loss = -(1-pt)^gama * log(pt), 显然是引入factor = (1-pt)^gama （gama>0)
                显然对与预测概率高的pt,fator 就会较小，所以权重就会更小，而且gama越大，反差感就越大
                这就是为什么能够挖掘难样本的原因，太容易判别的类别会给予较小的权重，难样本会有相对大点的权重
            """
            # (1-alpha) *   p^2 * -log(1-p + eps)
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            # (alpha) *  (1-p)^2 * -log(p + eps)
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())

            # 为什么用-1？？？？难道是对与正类y=1，对于负类y = -1？
            # [query, numi]
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]
            # [query, numi], 每个query 对当前存在的类别的分类损失

            # Compute the L1 cost between boxes
            # [[w,h,w,h],
            #   .....
            #  [w,h,w,h]
            #  ] #query rows
            # shape:[query, 4]
            image_size_out = targets[i]["image_size_xyxy"].unsqueeze(0).repeat(h*w, 1)
            # 同上，但是只有 #numi rows
            # shape:[numi, 4]
            image_size_tgt = targets[i]["image_size_xyxy_tgt"]
            # 归一化， out_bbox 已经原图(transform 后的图)化，如果要归一应 /image_size， image_size_out 确实是原图(transform 后的图)的大小
            # 其实targets[i]["boxes"] 已经归一化了，但是不用，因为他们格式不一样，
            out_bbox_ = out_bbox / image_size_out 
            tgt_bbox_ = tgt_bbox / image_size_tgt

            # here p norm 距离，所以p=1，为1范式距离
            # ret: [query, numi]
            cost_bbox = torch.cdist(out_bbox_, tgt_bbox_, p=1)

            # Compute the giou cost betwen boxes
            # ret: iou-(area-union)/area
            # for YOLO4: Lgiou = 1-iou + (area-u)/area
            # 所以这里用了-，但是1呢
            # 答：后面setCriterion 里面的giou 有加1，因为这里只是作选择，大家共同+1 或者不加是不影响选择的
            # pytroch 源码有实现
            # [query, numi]
            cost_giou = -generalized_box_iou(out_bbox, tgt_bbox) #  -iou + (area-union)/area ?
            # min cost_giou, iou(expected up) (area-union)/area(expected down) 也是正常的

            # Final cost matrix
            # [query, numi] 总的计算成本
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            # [query, numi] dim=0
            # 这里是二维，可以理解为只有row 和 col, dim=0 表示把行筛选合并，只剩下列的维度了。
            _, src_ind = torch.min(C, dim=0)
            # 相当于为每个numi 筛选一个最合适的query
            # 给target 编号，(0, tgt_ids)
            tgt_ind = torch.arange(len(tgt_ids)).to(src_ind) # to 只改变dtype 和device ...
            indices.append((src_ind, tgt_ind))
        # 把每个图片的src_ind(query_ind) , tgt_ind 转化为tensor
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]



def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # input: [N*query, num_class]
    # targets: [N*query, num_class] binary format
    p = torch.sigmoid(inputs) # 把每个数字转化为0-1区间 # 竟然不是用softmax, 是因为有些query是label 全为0？
    # [-targets[i]*log(inputs[i]) - (1-target[i])*log(1-inputs[i]) for i in range(len(inputs))]
    # ce_loss: [N*query, num_class]
    # 显然对于 ce_loss, 这样正样本就对应了-log(inputs[i]), 负样本就对应了-log(1-input[i])
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # pt = p if y==1 else (1-p)
    # p*targets 保留y=1 对应的p， (1-p) 调整识别为neg 的概率
    p_t = p * targets + (1 - p) * (1 - targets) # 其实两者虽然相加，但是位置都是错开的。
    loss = ce_loss * ((1 - p_t) ** gamma) # focal_loss = (1-pt)^gamma * (ce_loss)

    if alpha >= 0:
        # 保留0-1矩阵的好处，
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

# 采用修饰器 @torch.jit.script 或者下面的方式，都是把指定的方法脚本化
# TorchScript: 
"""
TorchScript是一种从PyTorch代码创建可序列化和可优化模型的方法。
任何TorchScript程序都可以从Python进程中保存，并加载到没有Python依赖的进程中。
通过一些特定的修饰器，torch.jit.script, torch.jit.unused ...
从纯Python程序转换为能够独立于Python运行的TorchScript程序，例如在独立的c++程序中。
这使得使用熟悉的Python工具在PyTorch中训练模型，
然后通过TorchScript将模型导出到生产环境中成为可能，

torch.jit.script能够被作为函数或装饰器使用。参数obj可以是class, function, nn.Module
对于nn.Module, 默认地编译其forward方法，并递归地编译其子模块以及被forward调用的函数。

torch.jit.unused: 指定不用脚本化的方法
https://zhuanlan.zhihu.com/p/135911580
"""
sigmoid_focal_loss_jit = torch.jit.script(
    sigmoid_focal_loss
)  # type: torch.jit.ScriptModule
