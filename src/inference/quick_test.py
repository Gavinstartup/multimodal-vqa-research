"""训练过程中，用中途保存的 projector 检查点做定性抽查。

跟 generate.py 的区别：generate.py 要读 final_model/（config + tokenizer + projector.pt
打包好的完整目录），这个目录只有整个训练跑完才会生成。这个脚本直接从裸的
projector_latest.pt / projector_best.pt checkpoint 加载权重，配合单独指定的
tokenizer/vision_model/text_model 来源，让你在训练还在跑的时候（另开一个进程，
不影响正在训练的那个）随时用最新进度测一下效果。

用法（从项目根目录运行）：
    python -m src.inference.quick_test \
        --projector_ckpt outputs/stage1_projector/projector_latest.pt \
        --vision_model "$CLIP_MODEL_PATH" --text_model "$TEXT_MODEL_PATH" \
        --image path/to/image.jpg --question "图中有什么？"
"""

import argparse
import logging

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from configs import stage1_config as cfg
from src.model import VQAForConditionalGeneration
from src.utils import resolve_device, resolve_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Mid-training qualitative check against a raw projector checkpoint")
    parser.add_argument("--projector_ckpt", type=str, required=True, help="projector_latest.pt / projector_best.pt")
    parser.add_argument("--vision_model", type=str, default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--text_model", type=str, default=cfg.ModelSettings.TEXT_MODEL_NAME)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.GenerationSettings.MAX_NEW_TOKENS)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override auto device selection, e.g. 'cpu' — use this to check progress "
             "without competing for GPU memory with a training run still in progress on "
             "the same GPU (loading a second full Qwen3-8B copy will not fit alongside it).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else resolve_device()
    dtype = resolve_dtype(device, args.dtype)

    logger.info("Loading tokenizer + image processor...")
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Must mirror train_stage1.py's tokenizer setup exactly: the checkpoint's saved
    # embedding size assumes the <image> token was added before resize_token_embeddings.
    if cfg.ModelSettings.IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [cfg.ModelSettings.IMAGE_TOKEN]})
    image_token_id = tokenizer.convert_tokens_to_ids(cfg.ModelSettings.IMAGE_TOKEN)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    logger.info("Building model (vision=%s, text=%s) ...", args.vision_model, args.text_model)
    model = VQAForConditionalGeneration.from_pretrained_components(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=args.text_model,
        image_token_id=image_token_id,
        torch_dtype=dtype,
        projector_hidden_act=cfg.ModelSettings.PROJECTOR_HIDDEN_ACT,
        vision_feature_layer=cfg.ModelSettings.VISION_FEATURE_LAYER,
        vision_feature_select_strategy=cfg.ModelSettings.VISION_FEATURE_SELECT_STRATEGY,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.to(device=device, dtype=dtype)
    # Matches train_stage1.py: the projector was trained in fp32, not the backbones' bf16.
    model.multi_modal_projector.to(torch.float32)

    logger.info("Loading projector weights from %s ...", args.projector_ckpt)
    projector_state = torch.load(args.projector_ckpt, map_location="cpu")
    if "projector_state_dict" in projector_state:
        projector_state = projector_state["projector_state_dict"]
    model.multi_modal_projector.load_state_dict(projector_state)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)

    image_token = cfg.ModelSettings.IMAGE_TOKEN
    num_image_tokens = model.config.num_image_tokens
    prompt = f"{image_token * num_image_tokens}\n{args.question.strip()}\n"
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    generated_ids = model.generate(
        input_ids=encoded["input_ids"],
        pixel_values=pixel_values,
        attention_mask=encoded["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
