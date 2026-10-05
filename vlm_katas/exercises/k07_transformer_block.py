import torch.nn as nn

from .k06_attention import CausalSelfAttention


class DecoderBlock(nn.Module):
    """Pre-Norm decoder block。"""

    def __init__(self, d_model, nhead, max_seq_len, dropout=0.0):
        super().__init__()
        # TODO: self-attention、两个 LayerNorm、4 倍宽 FFN。
        raise NotImplementedError

    def forward(self, x):
        # TODO: 两次 Pre-Norm 残差连接。
        raise NotImplementedError


class TinyGPT(nn.Module):
    """既支持 token ids，也支持预先计算的 inputs_embeds。"""

    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, max_seq_len=64):
        super().__init__()
        # TODO
        raise NotImplementedError

    def forward(self, input_ids=None, inputs_embeds=None):
        # TODO: 两者必须且只能提供一个，最后输出 vocab logits。
        raise NotImplementedError
