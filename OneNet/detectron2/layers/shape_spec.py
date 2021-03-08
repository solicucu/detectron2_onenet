# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from collections import namedtuple

# namedtuple 作为一个类，第一个是类名，第二个是字段名
# 跟常规元组使用方式差不多，但是可以通过classname.channels 的方式访问元素
# 这里是继承nametuple 类，重新定义类
class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.

    Attributes:
        channels:
        height:
        width:
        stride:
    """
    # new 是创建一个实例的最根本方法，它不是依赖于实例
    # 创建一个实例时，先调用__new__ 在调用__init__ 的
    # * 的意思，说明后面的参数要通过key-value 的形式传参
    def __new__(cls, *, channels=None, height=None, width=None, stride=None):
        return super().__new__(cls, channels, height, width, stride)
