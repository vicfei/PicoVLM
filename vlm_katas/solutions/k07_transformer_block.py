import torch.nn as nn

from .k06_attention import CausalSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, nhead, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=4, num_layers=2, max_seq_len=64):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, nhead, max_seq_len) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids=None, inputs_embeds=None):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("input_ids 和 inputs_embeds 必须且只能提供一个")
        x = self.token_embed(input_ids) if inputs_embeds is None else inputs_embeds
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.final_norm(x))
