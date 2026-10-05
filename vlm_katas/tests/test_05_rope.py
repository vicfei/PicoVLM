import pytest
import torch

from tests._loader import load

m = load("k05_rope")


def test_rope_shapes_and_position_zero_identity():
    cos, sin = m.precompute_rope(8, 6)
    x = torch.randn(2, 6, 3, 8)
    y = m.apply_rope(x, cos, sin)
    assert cos.shape == sin.shape == (6, 4)
    assert y.shape == x.shape
    assert torch.allclose(y[:, 0], x[:, 0])


def test_rotation_preserves_pairwise_norm():
    cos, sin = m.precompute_rope(8, 5)
    x = torch.randn(2, 5, 2, 8)
    y = m.apply_rope(x, cos, sin)
    before = x.reshape(*x.shape[:-1], 4, 2).square().sum(-1)
    after = y.reshape(*y.shape[:-1], 4, 2).square().sum(-1)
    assert torch.allclose(before, after, atol=1e-5)


def test_odd_head_dimension_is_rejected():
    with pytest.raises(ValueError):
        m.precompute_rope(7, 4)
