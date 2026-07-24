import pytest
import torch

from src.model import VQAForConditionalGeneration


def build_inputs(config, batch_size=2, num_text_tokens=3):
    num_image_tokens = config.num_image_tokens
    image_ids = torch.full((batch_size, num_image_tokens), config.image_token_id, dtype=torch.long)

    text_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, num_text_tokens))
    # Never let the random text accidentally contain the <image> id itself.
    clash = text_ids == config.image_token_id
    text_ids[clash] = (config.image_token_id + 1) % config.text_config.vocab_size

    input_ids = torch.cat([image_ids, text_ids], dim=1)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :num_image_tokens] = -100  # only the trailing "answer" tokens count towards loss

    pixel_values = torch.randn(
        batch_size, 3, config.vision_config.image_size, config.vision_config.image_size
    )
    return input_ids, attention_mask, labels, pixel_values


def test_forward_produces_finite_loss(tiny_config):
    model = VQAForConditionalGeneration(tiny_config)
    model.freeze_for_stage1()

    input_ids, attention_mask, labels, pixel_values = build_inputs(tiny_config)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_backward_updates_only_the_projector(tiny_config):
    model = VQAForConditionalGeneration(tiny_config)
    model.freeze_for_stage1()

    input_ids, attention_mask, labels, pixel_values = build_inputs(tiny_config)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
    outputs.loss.backward()

    for name, param in model.multi_modal_projector.named_parameters():
        assert param.grad is not None, f"projector param {name} got no gradient"
        assert torch.isfinite(param.grad).all()

    for name, param in model.vision_tower.named_parameters():
        assert param.grad is None, f"frozen vision_tower param {name} unexpectedly received a gradient"

    for name, param in model.language_model.named_parameters():
        assert param.grad is None, f"frozen language_model param {name} unexpectedly received a gradient"


def test_gradient_checkpointing_still_trains_only_the_projector(tiny_config):
    """Regression test: gradient_checkpointing_enable() only takes effect while a module
    is in .train() mode, so freeze_for_stage1() must not force vision_tower/language_model
    into .eval() — verify checkpointing actually engages and gradients still land only on
    the projector (not silently a no-op, not corrupting the frozen backbone's grad state)."""
    model = VQAForConditionalGeneration(tiny_config)
    model.freeze_for_stage1()
    model.language_model.gradient_checkpointing_enable()
    model.vision_tower.gradient_checkpointing_enable()
    model.train()

    assert model.language_model.is_gradient_checkpointing
    assert model.vision_tower.is_gradient_checkpointing

    input_ids, attention_mask, labels, pixel_values = build_inputs(tiny_config)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels, use_cache=False)
    outputs.loss.backward()

    assert torch.isfinite(outputs.loss)
    for param in model.multi_modal_projector.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()
    for param in model.vision_tower.parameters():
        assert param.grad is None
    for param in model.language_model.parameters():
        assert param.grad is None


def test_image_token_count_mismatch_raises(tiny_config):
    model = VQAForConditionalGeneration(tiny_config)
    input_ids, attention_mask, _, pixel_values = build_inputs(tiny_config)
    input_ids[:, 0] = 0  # one <image> placeholder short of what pixel_values provides

    with pytest.raises(ValueError):
        model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)


def test_image_features_actually_land_in_the_merged_embeddings(tiny_config):
    """Token-level fusion check: the <image> positions' embeddings must come from the
    projected vision features, not from the language model's own embedding table."""
    model = VQAForConditionalGeneration(tiny_config)
    input_ids, attention_mask, _, pixel_values = build_inputs(tiny_config, batch_size=1)
    num_image_tokens = tiny_config.num_image_tokens

    with torch.no_grad():
        text_embeds = model.get_input_embeddings()(input_ids)
        merged = model._merge_input_ids_with_image_features(input_ids, text_embeds.clone(), pixel_values)
        expected_image_features = model.get_image_features(pixel_values)

    assert torch.allclose(merged[:, :num_image_tokens, :], expected_image_features)
    assert not torch.allclose(merged[:, :num_image_tokens, :], text_embeds[:, :num_image_tokens, :])
    # the text tail must be untouched by the scatter
    assert torch.equal(merged[:, num_image_tokens:, :], text_embeds[:, num_image_tokens:, :])


def test_generate_returns_only_new_tokens(tiny_config):
    model = VQAForConditionalGeneration(tiny_config)
    model.eval()
    input_ids, attention_mask, _, pixel_values = build_inputs(tiny_config, batch_size=1)

    generated = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=5,
        min_new_tokens=5,
        do_sample=False,
        pad_token_id=0,
    )

    assert generated.shape == (1, 5)


def test_generate_with_precomputed_image_features_matches_pixel_values(tiny_config):
    """The REPL caches get_image_features(pixel_values) across questions about the same
    image — this locks in that passing it back via image_features= is equivalent to
    passing pixel_values= (and greedy decoding is deterministic, so outputs must match)."""
    model = VQAForConditionalGeneration(tiny_config)
    model.eval()
    input_ids, attention_mask, _, pixel_values = build_inputs(tiny_config, batch_size=1)

    with torch.no_grad():
        cached_features = model.get_image_features(pixel_values)

    kwargs = dict(
        input_ids=input_ids, attention_mask=attention_mask,
        max_new_tokens=5, min_new_tokens=5, do_sample=False, pad_token_id=0,
    )
    via_pixels = model.generate(pixel_values=pixel_values, **kwargs)
    via_cached_features = model.generate(image_features=cached_features, **kwargs)

    assert torch.equal(via_pixels, via_cached_features)
