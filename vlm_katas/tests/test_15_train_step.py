import pytest
import torch
import torch.nn as nn

from tests._loader import load

m = load("k15_train_step")


class ScalarRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, target):
        return ((x * self.weight - target) ** 2).mean()


def test_repeated_steps_reduce_loss_and_return_float():
    model = ScalarRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    batch = (torch.ones(8), torch.full((8,), 3.0))
    losses = [m.train_step(model, batch, optimizer) for _ in range(12)]
    assert isinstance(losses[0], float)
    assert losses[-1] < losses[0] * 0.1
    assert model.training


def test_non_scalar_loss_is_rejected():
    class BadModel(ScalarRegression):
        def forward(self, x, target):
            return (x * self.weight - target) ** 2

    model = BadModel()
    with pytest.raises(ValueError):
        m.train_step(model, (torch.ones(2), torch.ones(2)), torch.optim.SGD(model.parameters(), lr=0.1))
