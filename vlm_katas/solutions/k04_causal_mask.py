import torch


def make_causal_mask(length: int, device=None):
    return torch.full((length, length), float("-inf"), device=device).triu(1)
