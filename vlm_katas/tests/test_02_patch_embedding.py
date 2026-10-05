import torch

from tests._loader import load

m = load("k02_patch_embedding")


def test_patch_shape_and_count():
    layer = m.PatchEmbedding(image_size=28, patch_size=7, embed_dim=32)
    output = layer(torch.randn(3, 1, 28, 28))
    assert layer.num_patches == 16
    assert output.shape == (3, 16, 32)


def test_non_overlapping_patch_projection():
    layer = m.PatchEmbedding(image_size=4, patch_size=2, embed_dim=1)
    with torch.no_grad():
        layer.proj.weight.fill_(1.0)
        layer.proj.bias.zero_()
    image = torch.arange(16.0).reshape(1, 1, 4, 4)
    assert layer(image).flatten().tolist() == [10.0, 18.0, 42.0, 50.0]
