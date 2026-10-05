import math
import torch


def sinusoidal_position_encoding(length: int, dim: int):
    if dim % 2:
        raise ValueError("dim 必须是偶数")
    pe = torch.zeros(1, length, dim)
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[0, :, 0::2] = torch.sin(position * div)
    pe[0, :, 1::2] = torch.cos(position * div)
    return pe
