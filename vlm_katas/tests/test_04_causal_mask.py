import torch

from tests._loader import load

m = load("k04_causal_mask")


def test_causal_mask_values():
    mask = m.make_causal_mask(4)
    assert mask.shape == (4, 4)
    assert torch.equal(mask.diag(), torch.zeros(4))
    assert mask[3, 0] == 0
    assert torch.isneginf(mask[0, 1])
    assert torch.isneginf(mask[1, 3])


def test_mask_prevents_future_attention():
    weights = torch.softmax(torch.zeros(4, 4) + m.make_causal_mask(4), dim=-1)
    assert weights[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert torch.allclose(weights[2, :3], torch.full((3,), 1 / 3))
