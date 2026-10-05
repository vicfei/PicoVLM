import pytest
import torch

from tests._loader import load

m = load("k07_transformer_block")


def test_decoder_and_gpt_shapes():
    block = m.DecoderBlock(16, 4, 12)
    assert block(torch.randn(2, 7, 16)).shape == (2, 7, 16)
    gpt = m.TinyGPT(23, d_model=16, nhead=4, max_seq_len=12)
    assert gpt(input_ids=torch.randint(0, 23, (2, 7))).shape == (2, 7, 23)
    assert gpt(inputs_embeds=torch.randn(2, 7, 16)).shape == (2, 7, 23)


def test_exactly_one_input_mode_is_required():
    gpt = m.TinyGPT(10, d_model=16, nhead=4)
    with pytest.raises(ValueError):
        gpt()
    with pytest.raises(ValueError):
        gpt(input_ids=torch.ones(1, 2, dtype=torch.long), inputs_embeds=torch.randn(1, 2, 16))


def test_zero_sublayers_make_block_an_identity():
    block = m.DecoderBlock(16, 4, 8)
    for parameter in block.attn.parameters():
        parameter.data.zero_()
    for parameter in block.ffn.parameters():
        parameter.data.zero_()
    x = torch.randn(2, 5, 16)
    assert torch.equal(block(x), x)
