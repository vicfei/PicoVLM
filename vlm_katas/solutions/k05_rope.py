import torch


def precompute_rope(head_dim: int, max_seq_len: int, base=10000.0):
    if head_dim % 2:
        raise ValueError("head_dim 必须是偶数")
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq_len).float(), theta)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    real, imag = x[..., 0::2], x[..., 1::2]
    rotated_real = real * cos - imag * sin
    rotated_imag = real * sin + imag * cos
    return torch.stack((rotated_real, rotated_imag), dim=-1).flatten(-2)
