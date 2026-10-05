import pytest
import torch
import torch.nn as nn

from tests._loader import load

m = load("k10_multimodal_prefix")


class FakeVision(nn.Module):
    def __init__(self, width=6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.width = width

    def encode(self, images):
        return torch.ones(images.size(0), 4, self.width) * self.scale


class FakeLLM(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.token_embed = nn.Embedding(20, width)
        self.last_inputs = None

    def forward(self, inputs_embeds):
        self.last_inputs = inputs_embeds
        return inputs_embeds


def test_image_tokens_are_prefixed_before_text_tokens():
    vision, llm = FakeVision(), FakeLLM()
    projector = nn.Linear(6, 8)
    model = m.MiniVLMCore(vision, projector, llm)
    ids = torch.tensor([[2, 3, 4], [5, 6, 7]])
    joined = model.splice(torch.randn(2, 1, 10, 10), ids)
    assert joined.shape == (2, 7, 8)
    assert torch.allclose(joined[:, 4:], llm.token_embed(ids))
    assert torch.equal(model(torch.randn(2, 1, 10, 10), ids), llm.last_inputs)


def test_embedding_width_mismatch_is_rejected():
    model = m.MiniVLMCore(FakeVision(), nn.Identity(), FakeLLM())
    with pytest.raises(ValueError):
        model.splice(torch.randn(1, 1, 10, 10), torch.tensor([[2, 3]]))
