# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

from .build import META_ARCH_REGISTRY, build_model  # isort:skip

from .panoptic_fpn import PanopticFPN

# import all the meta_arch, so they will be registered
from .rcnn import GeneralizedRCNN, ProposalNetwork
from .retinanet import RetinaNet
from .semantic_seg import SEM_SEG_HEADS_REGISTRY, SemanticSegmentor, build_sem_seg_head


__all__ = list(globals().keys())

# globals() 返回当前文件的所有全局变量 以key-obj(value) 的形式（字典）返回
# 其中会包括奇奇怪怪的 比如"__name__": ..., "__builtins__":..., '__doc__':..., 等等

# __all__ 的用法：
'''
from xxx import *
若xxx定义了__all__属性，则只有__all__内指定的属性、方法、类可被导入；,注意 只需要字符串形式就行
若xxx没定义__all__，则导入模块内的所有公有属性，方法和类, _, __ 开头的类型就导入不了
'''