import torch.nn as nn


class Projector(nn.Module):
    """视觉空间到语言 embedding 空间的两层 MLP。"""

    def __init__(self, vision_dim, language_dim):
        super().__init__()
        # TODO: Linear(vision_dim, language_dim) -> GELU -> Linear(language_dim, language_dim)
        raise NotImplementedError

    def forward(self, vision_tokens):
        # TODO
        raise NotImplementedError
