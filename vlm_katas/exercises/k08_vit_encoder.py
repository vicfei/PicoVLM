import torch
import torch.nn as nn

from .k02_patch_embedding import PatchEmbedding
from .k03_position_encoding import sinusoidal_position_encoding


class ViTEncoder(nn.Module):
    def __init__(self, image_size=28, patch_size=7, embed_dim=32, nhead=4,
                 num_layers=2, num_classes=10):
        super().__init__()
        # TODO: patch embedding、CLS token、位置编码 buffer、双向 encoder、分类头。
        raise NotImplementedError

    def _tokens(self, images):
        # TODO: 构造包含 CLS 的完整 token 序列。
        raise NotImplementedError

    def encode(self, images):
        # TODO: 返回不含 CLS 的 patch tokens。
        raise NotImplementedError

    def classify(self, images):
        # TODO: 用 CLS token 分类。
        raise NotImplementedError
