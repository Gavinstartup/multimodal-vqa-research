"""Stage-2 指令微调默认配置。

沿用 stage1_config.py 的 ModelSettings/GenerationSettings（架构和生成参数不变），
新增 LoRA 相关设置，训练/路径设置单独定义（数据集、学习率、epoch 数都和 Stage-1
不同：指令数据规模更小、只需要浅调而不是从零对齐，LR 要低得多）。
"""

from pathlib import Path

from .stage1_config import GenerationSettings, ModelSettings

__all__ = ["ModelSettings", "GenerationSettings", "LoraSettings", "TrainingSettings", "PathSettings"]


class LoraSettings:
    R = 16
    ALPHA = 32
    DROPOUT = 0.05
    # Qwen3 attention + MLP projection names; covering both keeps LoRA capacity on the
    # parts of the layer that actually shift what the model attends to / generates,
    # not just attention (LLaVA-1.5's recipe covers MLP too for the same reason).
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class TrainingSettings:
    # Effective batch size 16, same as Stage-1 — LoRA + a small projector is much
    # lighter on activation memory than the frozen 8B backward pass Stage-1 pays for
    # anyway, so this is set by data-parallelism preference, not a new VRAM ceiling.
    BATCH_SIZE = 2
    EPOCHS = 1
    # LLaVA-1.5's LoRA recipe: a 10x split between the two param groups. The adapters
    # start from scratch (lora_B is zero-init) so they can take a much larger step,
    # while the projector is already aligned by Stage-1 and only wants a nudge —
    # pushing it at the adapter's LR would undo that alignment early in the run.
    LEARNING_RATE = 2e-4
    PROJECTOR_LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.0
    MAX_LENGTH = 2048
    GRADIENT_ACCUMULATION_STEPS = 8
    LOG_INTERVAL = 10
    VAL_SPLIT = 0.02
    # Cosine decay with a short warmup, stepped per optimizer step. Stage-1 could get
    # away with epoch-level ReduceLROnPlateau because it runs 3 epochs; instruction
    # tuning is a single pass over the data, so an epoch-level scheduler would never
    # fire even once and the LR would stay flat the whole run.
    WARMUP_RATIO = 0.03
    # Only reachable with --epochs > EARLY_STOPPING_PATIENCE (same invariant as
    # stage1_config). The default single-epoch run can't trigger it — there best/final
    # are the same checkpoint — it's here for multi-epoch runs on smaller subsets.
    EARLY_STOPPING_PATIENCE = 2
    NUM_WORKERS = 4
    SEED = 42


class PathSettings:
    # LLaVA-1.5 的 llava_v1_5_mix665k.json 中指向 coco/train2017 的那 364,100 条
    # （VQAv2 / OKVQA / A-OKVQA / RefCOCO + 原 LLaVA-Instruct-150K），由
    # scripts/prepare_autodl.sh 的 STAGE2_RECIPE=mix665k_coco 过滤产出——它同时把
    # "coco/train2017/xxx.jpg" 的前缀剥成裸文件名，所以 IMAGE_DIR 仍是扁平的 data/images。
    #
    # 换掉纯 LLaVA-Instruct-150K 的原因：那份数据全是 GPT-4 从 caption 生成的长篇描述，
    # 模型靠语言先验把话说圆就能拿到低 loss，视觉侧收不到纠错信号——实测训完能流利作答但
    # 会把「一只长颈鹿两只犀牛」说成斑马和牛。mix665k 里的短答案和区域指令数据没有这个空子可钻。
    #
    # 和 Stage-1 没有可复用的图片：那一阶段用的是 LLaVA-Pretrain 的 LAION/CC/SBU 图片，
    # 得靠 scripts/prepare_autodl.sh 的 DOWNLOAD_STAGE2_DATA=1 另外下 COCO train2017。
    DATA_DIR = Path("data")
    IMAGE_DIR = DATA_DIR / "images"
    ANNOTATIONS_JSON = DATA_DIR / "llava_v1_5_mix665k_coco.json"
    STAGE1_CHECKPOINT = Path("outputs") / "stage1_projector" / "final_model"
    OUTPUT_DIR = Path("outputs") / "stage2_instruct"
