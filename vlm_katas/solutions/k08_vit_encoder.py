import torch
import torch.nn as nn

from .k02_patch_embedding import PatchEmbedding
from .k03_position_encoding import sinusoidal_position_encoding


class ViTEncoder(nn.Module):
    def __init__(self, image_size=28, patch_size=7, embed_dim=32, nhead=4,
                 num_layers=2, num_classes=10):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, 1, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        position = sinusoidal_position_encoding(self.patch_embed.num_patches + 1, embed_dim)
        self.register_buffer("position", position)
        layer = nn.TransformerEncoderLayer(
            embed_dim, nhead, dim_feedforward=4 * embed_dim, batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.head = nn.Linear(embed_dim, num_classes)

    def _tokens(self, images):
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(images.size(0), -1, -1)
        return self.encoder(torch.cat((cls, patches), dim=1) + self.position)

    def encode(self, images):
        return self._tokens(images)[:, 1:]

    def classify(self, images):
        return self.head(self._tokens(images)[:, 0])
