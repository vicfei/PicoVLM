# -*- coding: utf-8 -*-
"""
迷你 VLM：ViT 视觉编码器 + LLaVA 式投影层 + RoPE-GPT 语言模型
在原始两份教学代码（RoPE-GPT / MNIST-ViT）基础上改进拼装而成。
改进点：
  1. ViT 的位置编码改为 register_buffer（原代码是普通属性，不进 state_dict）
  2. PAD 与 EOS 分离（原 GPT 中二者同为 0，导致模型永远学不会主动停止）
  3. RoPE-GPT 支持 inputs_embeds 输入（拼接图像 embedding 的前提）
  4. 训练采用 LLaVA 式分阶段策略：视觉预训练 -> 投影层对齐 -> 指令微调
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= 配置 =================
D_MODEL = 64          # 语言模型维度
NHEAD = 8
NUM_LAYERS = 2
MAX_SEQ_LEN = 64      # 16 图像 token + 文本，足够
VIT_EMBED = 64        # 视觉编码器维度（与 LLM 同宽，投影层可以更简单）
VIT_HEADS = 4
VIT_LAYERS = 2
IMG_TOKEN_NUM = 16    # 28/7=4 -> 4x4=16 个 patch token
PAD_ID, EOS_ID, UNK_ID = 0, 1, 2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ================= 分词器（字级别） =================
class CharTokenizer:
    """按单字切分的中英文混排分词器，词表由语料自动收集。"""

    def __init__(self, corpus):
        chars = sorted(set("".join(corpus)))
        self.itos = ['<pad>', '<eos>', '<unk>'] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi.get(c, UNK_ID) for c in s]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == EOS_ID:
                break
            if i not in (PAD_ID,):
                out.append(self.itos[i])
        return "".join(out)


# ================= RoPE（沿用文件1实现，去掉冗余的 repeat_interleave） =================
def precompute_rotary_emb(dim, max_seq_len, base=10000.0):
    theta = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, theta)      # [max_len, dim/2]
    return torch.cos(angles), torch.sin(angles)  # 直接返回 [T, D/2]


def apply_rotary_emb(x, cos, sin):
    """x: [B, T, H, D]; cos/sin: [T, D/2]，交错式旋转（GPT-J 布局）"""
    cos = cos.unsqueeze(0).unsqueeze(2)         # [1, T, 1, D/2]
    sin = sin.unsqueeze(0).unsqueeze(2)
    x_real, x_imag = x[..., 0::2], x[..., 1::2]
    r_real = x_real * cos - x_imag * sin
    r_imag = x_real * sin + x_imag * cos
    return torch.stack([r_real, r_imag], dim=-1).flatten(-2)


class RoPEMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, max_seq_len, dropout=0.1):
        super().__init__()
        self.nhead, self.head_dim = nhead, d_model // nhead
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        cos, sin = precompute_rotary_emb(self.head_dim, max_seq_len)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x, mask):
        B, T, D = x.shape
        q = apply_rotary_emb(self.Wq(x).view(B, T, self.nhead, self.head_dim),
                             self.cos[:T], self.sin[:T]).transpose(1, 2)
        k = apply_rotary_emb(self.Wk(x).view(B, T, self.nhead, self.head_dim),
                             self.cos[:T], self.sin[:T]).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + mask                  # 加性因果掩码
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)


class RoPEDecoderLayer(nn.Module):
    """Pre-Norm decoder 层（沿用文件1）"""

    def __init__(self, d_model, nhead, max_seq_len, dropout=0.1):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(d_model, nhead, max_seq_len, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = x + self.self_attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


class RoPEGPT(nn.Module):
    """改进自文件1：支持 inputs_embeds；PAD/EOS 分离"""

    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.layers = nn.ModuleList([
            RoPEDecoderLayer(d_model, nhead, max_seq_len)
            for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        nn.init.normal_(self.token_embed.weight, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)

    def forward(self, input_ids=None, inputs_embeds=None):
        x = self.token_embed(input_ids) if inputs_embeds is None else inputs_embeds
        T = x.size(1)
        mask = torch.full((T, T), float('-inf'), device=x.device).triu(1)  # 上三角 -inf
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.final_norm(x))


# ================= ViT 视觉编码器（改进自文件2） =================
class ViTBlock(nn.Module):
    """沿用文件2的 Post-Norm 结构"""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ff = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.norm2(x + self.dropout(ff))


class ViTEncoder(nn.Module):
    """改进点：pos_encoding 注册为 buffer；encode() 输出 patch token（去掉 CLS）"""

    def __init__(self, img_size=28, patch_size=7, in_channels=1,
                 embed_dim=VIT_EMBED, num_heads=VIT_HEADS, num_layers=VIT_LAYERS,
                 num_classes=10):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # 改进：注册为 buffer，随模型保存/迁移（原代码是普通属性）
        pe = torch.zeros(1, num_patches + 1, embed_dim)
        pos = torch.arange(num_patches + 1, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_dim, 2).float()
                        * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pos_encoding', pe)
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, 128) for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, num_classes)   # 仅阶段0预训练用

    def encode(self, x):
        """返回 [B, N, E] 的 patch token 序列（不含 CLS）"""
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)     # [B, N, E]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_encoding
        for blk in self.blocks:
            x = blk(x)
        return x[:, 1:, :]          # 丢弃 CLS，保留 16 个 patch token

    def classify(self, x):
        """阶段0预训练用：取 CLS 分类（走完整流程但不丢 CLS）"""
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
        h = h + self.pos_encoding
        for blk in self.blocks:
            h = blk(h)
        return self.fc(h[:, 0, :])


# ================= 投影层（LLaVA 式 2 层 MLP） =================
class Projector(nn.Module):
    """把视觉 token 从 ViT 特征空间映射到 LLM 词嵌入空间。
    Linear(d_v -> d_l) -> GELU -> Linear(d_l -> d_l)
    """

    def __init__(self, d_v=VIT_EMBED, d_l=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_v, d_l), nn.GELU(), nn.Linear(d_l, d_l))

    def forward(self, x):
        return self.net(x)


# ================= 迷你 VLM 总装 =================
class MiniVLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vision = ViTEncoder()
        self.projector = Projector()
        self.llm = RoPEGPT(vocab_size, D_MODEL, NHEAD, NUM_LAYERS, MAX_SEQ_LEN)

    def splice(self, images, input_ids):
        """图像 embedding 放最前面，后接文本 token embedding"""
        img_emb = self.projector(self.vision.encode(images))   # [B, 16, D]
        txt_emb = self.llm.token_embed(input_ids)              # [B, T, D]
        return torch.cat([img_emb, txt_emb], dim=1)            # [B, 16+T, D]

    def forward(self, images, input_ids, labels=None):
        x = self.splice(images, input_ids)
        logits = self.llm(inputs_embeds=x)
        loss = None
        if labels is not None:
            # 图像位置与 prompt 位置不计损失：labels 前补 16 个 -100
            ignore = torch.full((labels.size(0), IMG_TOKEN_NUM), -100,
                                dtype=torch.long, device=labels.device)
            full_labels = torch.cat([ignore, labels], dim=1)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                full_labels[:, 1:].reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, images, prompt_ids, max_new_tokens=16):
        self.eval()
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids], device=images.device)
            logits, _ = self.forward(images, inp)
            nxt = logits[0, -1].argmax().item()
            ids.append(nxt)
            if nxt == EOS_ID:
                break
        return ids[len(prompt_ids):]
