import pytest
import torch

from tests._loader import load

m = load("k06_attention")


def test_attention_shape_and_gradients():
    layer = m.CausalSelfAttention(16, 4, 10)
    x = torch.randn(2, 6, 16, requires_grad=True)
    output = layer(x)
    assert output.shape == x.shape
    output.sum().backward()
    assert x.grad is not None


def test_future_token_cannot_change_past_output():
    torch.manual_seed(1)
    layer = m.CausalSelfAttention(16, 4, 10).eval()
    x1 = torch.randn(1, 5, 16)
    x2 = x1.clone()
    x2[:, 4] += 100
    with torch.no_grad():
        y1, y2 = layer(x1), layer(x2)
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)
    assert not torch.allclose(y1[:, 4], y2[:, 4])


def test_invalid_width_and_excess_length_are_rejected():
    with pytest.raises(ValueError):
        m.CausalSelfAttention(15, 4, 10)
    layer = m.CausalSelfAttention(16, 4, 3)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 4, 16))
