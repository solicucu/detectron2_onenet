# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.utils.registry import Registry
# 创建一个注册器，在定义相关类或方法使用@xxx_REGISTRY.register() 修饰后
# 可以通过下面的方法，用xxx_REGISTRY.get(name) 就可以获取对应的对象
META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model.

The registered object will be called with `obj(cfg)`
and expected to return a `nn.Module` object.
"""


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model
