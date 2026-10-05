import torch


def make_causal_mask(length: int, device=None):
    """返回 [length,length] 加性 mask：过去/自身为 0，未来为 -inf。"""
    # TODO
    raise NotImplementedError
