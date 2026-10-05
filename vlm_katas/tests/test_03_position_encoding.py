import pytest
import torch

from tests._loader import load

m = load("k03_position_encoding")


def test_position_encoding_shape_origin_and_variation():
    pe = m.sinusoidal_position_encoding(5, 8)
    assert pe.shape == (1, 5, 8)
    assert torch.allclose(pe[0, 0, 0::2], torch.zeros(4))
    assert torch.allclose(pe[0, 0, 1::2], torch.ones(4))
    assert not torch.allclose(pe[:, 1], pe[:, 2])
    assert pe.requires_grad is False


def test_odd_dimension_is_rejected():
    with pytest.raises(ValueError):
        m.sinusoidal_position_encoding(4, 7)
