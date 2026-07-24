import torch
from PIL import Image

from src.inference.serve_repl import encode_image
from src.model import VQAForConditionalGeneration


class FakeImageProcessor:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, self.image_size, self.image_size)}


def test_encode_image_matches_get_image_features_on_same_pixels(tiny_config, tmp_path):
    model = VQAForConditionalGeneration(tiny_config)
    model.eval()

    image_path = tmp_path / "a.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    image_processor = FakeImageProcessor(tiny_config.vision_config.image_size)

    cached = encode_image(model, image_processor, str(image_path), torch.device("cpu"))
    direct = model.get_image_features(torch.zeros(1, 3, tiny_config.vision_config.image_size, tiny_config.vision_config.image_size))

    assert torch.equal(cached, direct)
    assert cached.shape == (1, tiny_config.num_image_tokens, tiny_config.text_config.hidden_size)
