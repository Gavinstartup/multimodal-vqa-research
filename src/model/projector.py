import torch.nn as nn
from transformers.activations import ACT2FN


class VQAMultiModalProjector(nn.Module):
    """2-layer MLP projecting CLIP patch features into the Qwen3 embedding space.

    Simplified from the legacy 5-layer ReLU projector (docs/legacy_reference) to the
    LLaVA-1.5 official design: it was overfitting on small datasets and its loss target
    (pooled text embedding) no longer applies now that the LM computes the loss directly.
    """

    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    def forward(self, image_features):
        # Trained in fp32 (see train_stage1.py) while the vision tower feeding this
        # runs in bf16 — cast up so the matmuls here don't inherit that lower precision.
        image_features = image_features.to(self.linear_1.weight.dtype)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


__all__ = ["VQAMultiModalProjector"]
