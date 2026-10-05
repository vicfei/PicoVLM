import torch
import torch.nn as nn

from tests._loader import load

m = load("k13_generation")


class ScriptedModel(nn.Module):
    def __init__(self, script):
        super().__init__()
        self.script = script
        self.calls = 0

    def forward(self, images, input_ids):
        next_id = self.script[self.calls]
        self.calls += 1
        logits = torch.zeros(1, input_ids.size(1), 8)
        logits[0, -1, next_id] = 10
        return logits, None


def test_generation_stops_at_eos_and_restores_training_mode():
    model = ScriptedModel([4, 5, 1, 7]).train()
    output = m.greedy_generate(model, torch.zeros(1, 1, 2, 2), [2, 3], eos_id=1)
    assert output == [4, 5, 1]
    assert model.calls == 3
    assert model.training is True


def test_generation_respects_max_tokens():
    model = ScriptedModel([4, 4, 4])
    assert m.greedy_generate(model, torch.zeros(1), [2], max_new_tokens=2) == [4, 4]
