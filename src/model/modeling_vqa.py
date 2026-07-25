import torch
from transformers import AutoConfig, AutoModelForCausalLM, CLIPVisionConfig, CLIPVisionModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_vqa import VQAConfig
from .projector import VQAMultiModalProjector


class VQAPreTrainedModel(PreTrainedModel):
    config_class = VQAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["CLIPEncoderLayer", "Qwen3DecoderLayer"]


class VQAForConditionalGeneration(VQAPreTrainedModel):
    """CLIP ViT + MLP projector + Qwen3 causal LM, LLaVA-style.

    Image patch features replace the ``<image>`` placeholder positions in the text
    embedding sequence (vectorized via masked_scatter, one call handles every image in
    the batch regardless of how many patch tokens each expands to).
    """

    def __init__(self, config: VQAConfig, load_vision_tower=True, load_language_model=True):
        super().__init__(config)
        self.multi_modal_projector = VQAMultiModalProjector(config)
        # from_pretrained_components() passes False here: it immediately overwrites both
        # with pretrained weights, so randomly initializing a full Qwen3-8B (and CLIP)
        # first would just waste time/RAM before being thrown away.
        self.vision_tower = CLIPVisionModel(config.vision_config) if load_vision_tower else None
        self.language_model = AutoModelForCausalLM.from_config(config.text_config) if load_language_model else None
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        # Not self.vision_tower.embeddings.patch_embedding.weight.dtype: CLIPVisionModel's
        # internal module tree (flat vs. wrapped in a .vision_model submodule) has changed
        # across transformers versions, so pull dtype from any parameter instead of assuming
        # a specific submodule path.
        vision_dtype = next(self.vision_tower.parameters()).dtype
        pixel_values = pixel_values.to(dtype=vision_dtype)
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        selected = image_outputs.hidden_states[self.config.vision_feature_layer]
        if self.config.vision_feature_select_strategy == "patch":
            selected = selected[:, 1:]  # drop the CLS token, keep patch tokens only
        return self.multi_modal_projector(selected)

    def _scatter_image_features(self, input_ids, inputs_embeds, image_features):
        """Splices already-computed (projected) image features into the <image> positions.

        Split out from _merge_input_ids_with_image_features so callers that already have
        image_features cached (e.g. a REPL asking several questions about one image) can
        skip re-running the vision tower + projector on every call.
        """
        image_features = image_features.to(inputs_embeds.dtype)
        special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

        n_image_tokens = special_image_mask.sum() // inputs_embeds.shape[-1]
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Number of <image> placeholder tokens ({n_image_tokens}) does not match "
                f"number of image patch features ({n_image_features}). Every sample must "
                f"expand its <image> token to exactly config.num_image_tokens placeholders."
            )

        return inputs_embeds.masked_scatter(special_image_mask, image_features)

    def _merge_input_ids_with_image_features(self, input_ids, inputs_embeds, pixel_values):
        image_features = self.get_image_features(pixel_values)
        return self._scatter_image_features(input_ids, inputs_embeds, image_features)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            inputs_embeds = self._merge_input_ids_with_image_features(input_ids, inputs_embeds, pixel_values)

        outputs = self.language_model(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            **kwargs,
        )
        return CausalLMOutputWithPast(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, image_features=None, attention_mask=None, **generate_kwargs):
        """Runs generation by pre-merging image features into the prompt embeddings.

        Delegated to language_model.generate(inputs_embeds=...) rather than a full
        GenerationMixin override with per-step image handling: the image only needs to
        be merged once (into the prompt), every generated token afterwards is text-only,
        so there is no benefit to re-deriving image features at each decoding step.

        Pass a pre-computed ``image_features`` (from get_image_features) instead of
        ``pixel_values`` to skip the vision tower + projector forward pass entirely —
        useful for answering several questions about the same image without re-encoding it.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if image_features is None and pixel_values is not None:
            image_features = self.get_image_features(pixel_values)
        if image_features is not None:
            inputs_embeds = self._scatter_image_features(input_ids, inputs_embeds, image_features)
        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    def freeze_for_stage1(self):
        """Freezes vision tower + language model, leaves only the projector trainable.

        Deliberately does not force vision_tower/language_model into .eval() mode: both
        checkpoints use dropout=0.0, so train()/eval() has no numerical effect on them,
        and gradient_checkpointing_enable() (see train_stage1.py) only takes effect while
        a module is in .train() mode. Train/eval mode is instead driven uniformly by
        model.train()/model.eval() on the whole tree, same as any other model.
        """
        for param in self.vision_tower.parameters():
            param.requires_grad = False
        for param in self.language_model.parameters():
            param.requires_grad = False
        for param in self.multi_modal_projector.parameters():
            param.requires_grad = True

    @classmethod
    def from_pretrained_components(
        cls,
        vision_model_name_or_path,
        text_model_name_or_path,
        image_token_id,
        torch_dtype=None,
        **config_kwargs,
    ):
        """Builds the model from two separately pretrained checkpoints (no merged checkpoint exists yet)."""
        vision_config = CLIPVisionConfig.from_pretrained(vision_model_name_or_path)
        text_config = AutoConfig.from_pretrained(text_model_name_or_path)
        config = VQAConfig(
            vision_config=vision_config,
            text_config=text_config,
            image_token_id=image_token_id,
            **config_kwargs,
        )

        model = cls(config, load_vision_tower=False, load_language_model=False)
        model.vision_tower = CLIPVisionModel.from_pretrained(vision_model_name_or_path, torch_dtype=torch_dtype)
        model.language_model = AutoModelForCausalLM.from_pretrained(text_model_name_or_path, torch_dtype=torch_dtype)
        return model


AutoConfig.register("qwen3_vqa", VQAConfig)

__all__ = ["VQAPreTrainedModel", "VQAForConditionalGeneration"]
