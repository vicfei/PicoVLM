import torch

from tests._loader import load

m = load("k08_vit_encoder")


def test_vit_feature_and_classification_shapes():
    model = m.ViTEncoder(embed_dim=32, num_layers=1)
    images = torch.randn(2, 1, 28, 28)
    assert model.encode(images).shape == (2, 16, 32)
    assert model.classify(images).shape == (2, 10)


def test_position_encoding_is_buffer_and_cls_is_parameter():
    model = m.ViTEncoder(embed_dim=32, num_layers=1)
    buffers = dict(model.named_buffers())
    parameters = dict(model.named_parameters())
    assert "position" in buffers
    assert "cls_token" in parameters
    assert "position" in model.state_dict()
