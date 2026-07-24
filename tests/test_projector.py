import torch

from src.model.projector import VQAMultiModalProjector


def test_projector_projects_vision_hidden_size_to_text_hidden_size(tiny_config):
    projector = VQAMultiModalProjector(tiny_config)
    x = torch.randn(2, 4, tiny_config.vision_config.hidden_size)

    out = projector(x)

    assert out.shape == (2, 4, tiny_config.text_config.hidden_size)
