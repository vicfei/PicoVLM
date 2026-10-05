import torch.nn as nn


class PatchEmbedding(nn.Module):
    """使用无重叠卷积将 [B,C,H,W] 转换成 [B,N,D]。"""

    def __init__(self, image_size=28, patch_size=7, in_channels=1, embed_dim=64):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        # TODO: 定义 kernel_size=stride=patch_size 的卷积。
        self.proj = None

    def forward(self, images):
        # TODO: 卷积后把空间维展平成 token 维，并得到 [B,N,D]。
        raise NotImplementedError
