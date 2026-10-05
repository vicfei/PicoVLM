import pytest
import torch

from tests._loader import load

m = load("k11_answer_loss")


def test_collate_masks_prompt_padding_and_supervises_eos():
    samples = [
        (torch.zeros(1, 2, 2), [5, 6], [7, 8]),
        (torch.ones(1, 2, 2), [9], [4]),
    ]
    images, ids, labels = m.collate_answer_only(samples)
    assert images.shape == (2, 1, 2, 2)
    assert ids.tolist() == [[5, 6, 7, 8, 1], [9, 4, 1, 0, 0]]
    assert labels.tolist() == [[-100, -100, 7, 8, 1], [-100, 4, 1, -100, -100]]


def test_shifted_loss_uses_only_positions_before_answer_targets():
    labels = torch.tensor([[-100, 3, 1]])
    logits = torch.zeros(1, 5, 6, requires_grad=True)  # 2 image + 3 text
    loss = m.shifted_answer_loss(logits, labels, image_token_count=2)
    loss.backward()
    nonzero_positions = logits.grad.abs().sum(-1).ne(0).nonzero().tolist()
    assert nonzero_positions == [[0, 2], [0, 3]]


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        m.shifted_answer_loss(torch.randn(1, 4, 5), torch.ones(1, 3, dtype=torch.long), 2)
