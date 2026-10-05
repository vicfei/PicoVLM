import math
import torch
import torch.nn as nn

from .k04_causal_mask import make_causal_mask
from .k05_rope import apply_rope, precompute_rope


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.0):
        super().__init__()
        if d_model % nhead:
            raise ValueError("d_model 必须能被 nhead 整除")
        self.nhead = nhead
        self.head_dim = d_model // nhead
        if self.head_dim % 2:
            raise ValueError("RoPE 要求 head_dim 为偶数")
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        cos, sin = precompute_rope(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        batch, length, width = x.shape
        if length > self.cos.size(0):
            raise ValueError("序列长度超过 max_seq_len")
        shape = (batch, length, self.nhead, self.head_dim)
        q = apply_rope(self.q_proj(x).view(shape), self.cos[:length], self.sin[:length])
        k = apply_rope(self.k_proj(x).view(shape), self.cos[:length], self.sin[:length])
        v = self.v_proj(x).view(shape)
        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        weights = self.dropout(torch.softmax(scores + make_causal_mask(length, x.device), dim=-1))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.out_proj(output)
