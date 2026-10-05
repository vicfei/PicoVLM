import torch
import torch.nn as nn

from tests._loader import load

m = load("k09_projector")


def test_projector_shape_architecture_and_gradient():
    projector = m.Projector(24, 32)
    linear_layers = [layer for layer in projector.modules() if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [(24, 32), (32, 32)]
    x = torch.randn(2, 16, 24, requires_grad=True)
    output = projector(x)
    assert output.shape == (2, 16, 32)
    output.square().mean().backward()
    assert x.grad is not None
