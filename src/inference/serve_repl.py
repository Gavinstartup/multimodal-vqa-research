"""常驻推理 REPL：模型只在启动时加载一次，循环回答问题。

解决的问题：generate.py 是一次性 CLI，每次调用都要重新读一遍 CLIP + Qwen3-8B 的权重
（几十秒冷启动）；这个脚本把模型常驻在一个进程里，问多个问题只加载一次。同一张图连续
问多个问题时，图像特征（vision_tower + projector 的输出）也会被缓存，不用每次重新编码。

用法（从项目根目录运行）：
    python -m src.inference.serve_repl --checkpoint outputs/stage1_projector/final_model

交互命令：
    image: <path>   切换/设置当前图片（会重新编码一次并缓存）
    <任意其他输入>   针对当前图片提问
    quit / exit      退出
"""

import argparse
import logging
import os
import threading

import torch
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor, TextIteratorStreamer

from configs import stage1_config as cfg
from src.model import VQAConfig, VQAForConditionalGeneration
from src.utils import resolve_device, resolve_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VQA persistent inference REPL")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vision_model", type=str, default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--max_new_tokens", type=int, default=cfg.GenerationSettings.MAX_NEW_TOKENS)
    parser.add_argument("--do_sample", action="store_true", default=cfg.GenerationSettings.DO_SAMPLE)
    parser.add_argument("--temperature", type=float, default=cfg.GenerationSettings.TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=cfg.GenerationSettings.TOP_P)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    return parser.parse_args()


def load_model(args, device, dtype):
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
    projector_state = torch.load(projector_path, map_location="cpu")
    if "projector_state_dict" in projector_state:
        projector_state = projector_state["projector_state_dict"]
    model.multi_modal_projector.load_state_dict(projector_state)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tokenizer, image_processor


def encode_image(model, image_processor, image_path, device):
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        return model.get_image_features(pixel_values)


def answer(model, tokenizer, image_features, question, args, device):
    num_image_tokens = model.config.num_image_tokens
    image_token = cfg.ModelSettings.IMAGE_TOKEN
    prompt = f"{image_token * num_image_tokens}\n{question.strip()}\n"
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=encoded["input_ids"],
        image_features=image_features,
        attention_mask=encoded["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        streamer=streamer,
    )
    # generate() blocks until done, so it has to run on its own thread for the
    # TextIteratorStreamer to be readable from the main thread as tokens arrive.
    thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
    thread.start()
    for chunk in streamer:
        print(chunk, end="", flush=True)
    thread.join()
    print()


def main():
    args = parse_args()
    device = resolve_device()
    dtype = resolve_dtype(device, args.dtype)

    model, tokenizer, image_processor = load_model(args, device, dtype)
    logger.info("Model loaded once, ready. Type 'image: <path>' to set/switch image, 'quit' to exit.")

    current_image_features = None
    current_image_path = None

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        if user_input.lower().startswith("image:"):
            path = user_input.split(":", 1)[1].strip()
            if not os.path.exists(path):
                print(f"file not found: {path}")
                continue
            current_image_features = encode_image(model, image_processor, path, device)
            current_image_path = path
            print(f"[image set: {path}]")
            continue

        if current_image_features is None:
            print("no image set yet — type 'image: <path>' first")
            continue

        print(f"[{current_image_path}] ", end="")
        answer(model, tokenizer, current_image_features, user_input, args, device)


if __name__ == "__main__":
    main()
