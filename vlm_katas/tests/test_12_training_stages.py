import pytest
import torch.nn as nn

from tests._loader import load

m = load("k12_training_stages")


class ThreePartModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = nn.Linear(2, 3)      # 9
        self.projector = nn.Linear(3, 4)   # 16
        self.llm = nn.Linear(4, 5)         # 25


@pytest.mark.parametrize(
    "stage,training,count",
    [
        ("vision_pretrain", (True, False, False), 9),
        ("language_pretrain", (False, False, True), 25),
        ("align", (False, True, False), 16),
        ("sft", (False, True, True), 41),
    ],
)
def test_stage_controls_gradient_and_module_mode(stage, training, count):
    model = ThreePartModel()
    m.configure_stage(model, stage)
    modules = (model.vision, model.projector, model.llm)
    assert tuple(module.training for module in modules) == training
    assert tuple(next(module.parameters()).requires_grad for module in modules) == training
    assert m.trainable_parameter_count(model) == count


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        m.configure_stage(ThreePartModel(), "mystery")
