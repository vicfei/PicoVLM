from types import SimpleNamespace

import torch
import torch.nn as nn

from tests._loader import load

m = load("k14_checkpoint")


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(2)
    source = nn.Linear(3, 2)
    tokenizer = SimpleNamespace(itos=["<pad>", "<eos>", "字"], stoi={"<pad>": 0, "<eos>": 1, "字": 2})
    path = tmp_path / "model.pt"
    m.save_checkpoint(path, source, tokenizer, {"d_model": 3, "layers": 1})

    restored = nn.Linear(3, 2)
    metadata = m.load_checkpoint(path, restored)
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)
    assert metadata["tokenizer"]["itos"] == tokenizer.itos
    assert metadata["config"] == {"d_model": 3, "layers": 1}
