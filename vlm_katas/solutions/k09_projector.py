import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, vision_dim, language_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, language_dim),
            nn.GELU(),
            nn.Linear(language_dim, language_dim),
        )

    def forward(self, vision_tokens):
        return self.net(vision_tokens)
