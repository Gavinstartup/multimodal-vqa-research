import pytest

from src.model import VQAConfig


def test_num_image_tokens_patch_strategy(tiny_config):
    side = tiny_config.vision_config.image_size // tiny_config.vision_config.patch_size
    assert tiny_config.num_image_tokens == side * side


def test_num_image_tokens_full_strategy_keeps_cls_token(tiny_vision_config, tiny_text_config):
    config = VQAConfig(
        vision_config=tiny_vision_config,
        text_config=tiny_text_config,
        image_token_id=99,
        vision_feature_select_strategy="full",
    )
    side = tiny_vision_config.image_size // tiny_vision_config.patch_size
    assert config.num_image_tokens == side * side + 1


def test_invalid_select_strategy_raises(tiny_vision_config, tiny_text_config):
    with pytest.raises(ValueError):
        VQAConfig(
            vision_config=tiny_vision_config,
            text_config=tiny_text_config,
            vision_feature_select_strategy="bogus",
        )


def test_save_and_reload_round_trip(tiny_config, tmp_path):
    """Regression test: text_config used to crash on reload (AutoConfig.for_model got
    model_type both positionally and via **text_config, which still contained that key)."""
    tiny_config.save_pretrained(tmp_path)
    reloaded = VQAConfig.from_pretrained(tmp_path)

    assert reloaded.image_token_id == tiny_config.image_token_id
    assert reloaded.projector_hidden_act == tiny_config.projector_hidden_act
    assert reloaded.text_config.hidden_size == tiny_config.text_config.hidden_size
    assert reloaded.text_config.model_type == "qwen3"
    assert reloaded.vision_config.hidden_size == tiny_config.vision_config.hidden_size
    assert reloaded.num_image_tokens == tiny_config.num_image_tokens
