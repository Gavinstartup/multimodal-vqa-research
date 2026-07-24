from transformers import AutoConfig, CLIPVisionConfig, PretrainedConfig


class VQAConfig(PretrainedConfig):
    """Config for the CLIP ViT + MLP projector + Qwen3 VQA model.

    Composes a frozen vision encoder config and a frozen language model config,
    following the same sub-config pattern HF uses for LLaVA-style models.
    """

    model_type = "qwen3_vqa"
    sub_configs = {"vision_config": CLIPVisionConfig, "text_config": AutoConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        image_token_id=None,
        projector_hidden_act="gelu",
        vision_feature_layer=-2,
        vision_feature_select_strategy="patch",
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            vision_config = CLIPVisionConfig(**vision_config)
        elif vision_config is None:
            # openai/clip-vit-large-patch14-336 defaults
            vision_config = CLIPVisionConfig(
                hidden_size=1024,
                image_size=336,
                patch_size=14,
                num_hidden_layers=24,
                num_attention_heads=16,
                intermediate_size=4096,
            )
        self.vision_config = vision_config

        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_model_type = text_config.pop("model_type", "qwen3")
            text_config = AutoConfig.for_model(text_model_type, **text_config)
        elif text_config is None:
            text_config = AutoConfig.for_model("qwen3")
        self.text_config = text_config

        self.image_token_id = image_token_id
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_layer = vision_feature_layer
        if vision_feature_select_strategy not in ("patch", "full"):
            raise ValueError(
                f"vision_feature_select_strategy must be 'patch' or 'full', got {vision_feature_select_strategy}"
            )
        self.vision_feature_select_strategy = vision_feature_select_strategy

        super().__init__(**kwargs)

    @property
    def num_image_tokens(self):
        """Number of patch tokens one image expands to."""
        side = self.vision_config.image_size // self.vision_config.patch_size
        n = side * side
        if self.vision_feature_select_strategy == "full":
            n += 1  # CLS token kept
        return n


__all__ = ["VQAConfig"]
