import torch
import torch.nn as nn

from .k04_causal_mask import make_causal_mask
from .k05_rope import apply_rope, precompute_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.0):
        super().__init__()
        # TODO: 检查整除关系，创建 Q/K/V/O 四个线性层，预计算 RoPE。
        raise NotImplementedError

    def forward(self, x):
        # TODO: 拆 head、对 Q/K 应用 RoPE、加 causal mask、合并 heads。
        raise NotImplementedError
