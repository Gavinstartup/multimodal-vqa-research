"""Stage-1 projector 对齐训练默认配置。

分类方式参考 docs/legacy_reference/config.py，内容整体替换为 CLIP-ViT-L/14-336 +
Qwen3-8B 的新架构；旧版 AttentionMaskSettings（从未被实际使用的死代码）不再保留。
"""

from pathlib import Path


class ModelSettings:
    VISION_MODEL_NAME = "openai/clip-vit-large-patch14-336"
    TEXT_MODEL_NAME = "Qwen/Qwen3-8B"
    IMAGE_TOKEN = "<image>"
    PROJECTOR_HIDDEN_ACT = "gelu"
    VISION_FEATURE_LAYER = -2
    VISION_FEATURE_SELECT_STRATEGY = "patch"


class TrainingSettings:
    # Qwen3-8B is frozen but backward still has to walk all 36 layers to reach the
    # projector, so activation memory (not weight memory) sets the real VRAM ceiling.
    # batch_size=1 + grad_accum=16 (effective batch 16) is sized to fit a ~24GB GPU with
    # gradient checkpointing on; raise batch_size if you've confirmed more headroom.
    BATCH_SIZE = 1
    EPOCHS = 3
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.0
    MAX_LENGTH = 1024
    GRADIENT_ACCUMULATION_STEPS = 16
    LOG_INTERVAL = 10
    VAL_SPLIT = 0.05
    EARLY_STOPPING_PATIENCE = 5
    LR_PATIENCE = 2
    LR_FACTOR = 0.5
    NUM_WORKERS = 4
    SEED = 42


class PathSettings:
    # 推荐用 LLaVA-Pretrain 的 blip_laion_cc_sbu_558k（或原始 LLaVA-CC3M-595K）
    # 作为 Stage-1 对齐数据集，annotations.json 保持其原生 conversations 格式即可，
    # src/data/dataset.py 会自动识别，见 scripts/prepare_autodl.sh 里的下载步骤。
    DATA_DIR = Path("data")
    IMAGE_DIR = DATA_DIR / "images"
    ANNOTATIONS_JSON = DATA_DIR / "blip_laion_cc_sbu_558k.json"
    OUTPUT_DIR = Path("outputs") / "stage1_projector"


class GenerationSettings:
    MAX_NEW_TOKENS = 256
    TEMPERATURE = 0.7
    TOP_P = 0.9
    DO_SAMPLE = False
