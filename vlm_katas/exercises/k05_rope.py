import torch


def precompute_rope(head_dim: int, max_seq_len: int, base=10000.0):
    """返回形状均为 [T,D/2] 的 cos 和 sin。"""
    # TODO
    raise NotImplementedError


def apply_rope(x, cos, sin):
    """对 [B,T,H,D] 的偶/奇通道两两旋转。"""
    # TODO
    raise NotImplementedError
