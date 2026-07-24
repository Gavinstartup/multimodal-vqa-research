"""Stage-1 推理：给定一张图片和一个问题，输出模型的回答。

用法（从项目根目录运行）：
    python -m src.inference.generate \
        --checkpoint outputs/stage1_projector/final_model \
        --image path/to/image.jpg \
        --question "图中有什么？"
"""

import argparse
import logging
import os

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from configs import stage1_config as cfg
from src.model import VQAConfig, VQAForConditionalGeneration

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VQA inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Directory saved by train_stage1.py (contains config.json, projector.pt, tokenizer files)")
    parser.add_argument("--vision_model", type=str, default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.GenerationSettings.MAX_NEW_TOKENS)
    parser.add_argument("--do_sample", action="store_true", default=cfg.GenerationSettings.DO_SAMPLE)
    parser.add_argument("--temperature", type=float, default=cfg.GenerationSettings.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=cfg.GenerationSettings.TOP_P)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype) if device.type == "cuda" else torch.float32

    logger.info("Loading config + tokenizer from %s ...", args.checkpoint)
    config = VQAConfig.from_pretrained(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_model)

    logger.info("Building model (vision=%s, text=%s) ...", args.vision_model, config.text_config.name_or_path)
    model = VQAForConditionalGeneration.from_pretrained_components(
        vision_model_name_or_path=args.vision_model,
        text_model_name_or_path=config.text_config.name_or_path,
        image_token_id=config.image_token_id,
        torch_dtype=dtype,
        projector_hidden_act=config.projector_hidden_act,
        vision_feature_layer=config.vision_feature_layer,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
    )
    model.language_model.resize_token_embeddings(len(tokenizer))

    projector_path = os.path.join(args.checkpoint, "projector.pt")
    logger.info("Loading trained projector weights from %s ...", projector_path)
    projector_state = torch.load(projector_path, map_location="cpu")
    if "projector_state_dict" in projector_state:  # projector_best.pt / projector_final.pt style
        projector_state = projector_state["projector_state_dict"]
    model.multi_modal_projector.load_state_dict(projector_state)

    model.to(device=device, dtype=dtype)
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
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    answer = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(answer.strip())


if __name__ == "__main__":
    main()
