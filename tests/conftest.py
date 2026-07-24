import sys
from pathlib import Path

import pytest
import torch  # noqa: F401  (must be imported before transformers — see README note below)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# On this Windows dev setup, if transformers imports torch for the first time via its
# own lazy submodule chain (e.g. while pytest is collecting), torch's DLL loader fails
# with "shm.dll" not found. Importing torch explicitly first avoids it.
from transformers import AutoConfig as HFAutoConfig
from transformers import CLIPVisionConfig

from src.model import VQAConfig


@pytest.fixture
def tiny_vision_config():
    return CLIPVisionConfig(
        hidden_size=8,
        image_size=32,
        patch_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=16,
        num_channels=3,
    )


@pytest.fixture
def tiny_text_config():
    return HFAutoConfig.for_model(
        "qwen3",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=100,
        max_position_embeddings=64,
    )


@pytest.fixture
def tiny_config(tiny_vision_config, tiny_text_config):
    return VQAConfig(
        vision_config=tiny_vision_config,
        text_config=tiny_text_config,
        image_token_id=99,
    )
