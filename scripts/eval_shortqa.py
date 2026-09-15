"""在短答案评估集上跑一个 checkpoint，输出可比较的准确率数字。

两个指标：

- **exact match**：归一化（小写、去标点冠词）后完全相同。这是 VQA 文献里的常规口径。
- **containment**：标准答案作为词序列出现在模型输出里。Stage-1 或用长描述数据训出来的
  模型即使「看对了」也会答一整段话，exact match 会给它 0 分；containment 能把
  「答对了但啰嗦」和「答错了」区分开，比较不同数据配方时这个更公平。

模型只加载一次，并且按图片分组、同一张图的多个问题复用一次视觉编码（沿用
src/inference/serve_repl.py 里的 load_model / encode_image）。

跑法（项目根目录）：

    source models_paths.env
    python scripts/eval_shortqa.py \
        --checkpoint outputs/stage2_instruct/final_model \
        --vision_model "$VISION_MODEL_PATH" \
        --eval_set data/eval_shortqa.json --image_dir data/images
"""

import argparse
import json
import logging
import os
import re
import string
from collections import defaultdict

import torch

from configs import stage2_config as cfg
from src.inference.serve_repl import encode_image, load_model
from src.utils import resolve_device, resolve_dtype

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


def parse_args():
    parser = argparse.ArgumentParser(description="Short-answer accuracy for a VQA checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vision_model", default=cfg.ModelSettings.VISION_MODEL_NAME)
    parser.add_argument("--eval_set", default="data/eval_shortqa.json")
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 = 全部）")
    parser.add_argument("--max_new_tokens", type=int, default=32,
                        help="短答案题不需要长输出；截短能显著加快评估")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--dump", default="", help="把逐条结果写成 jsonl，便于人工翻看错例")
    return parser.parse_args()


def normalize(text):
    text = text.lower().strip().translate(_PUNCT)
    tokens = [t for t in re.split(r"\s+", text) if t and t not in _ARTICLES]
    return tokens


def contains(pred_tokens, gold_tokens):
    if not gold_tokens:
        return False
    n = len(gold_tokens)
    return any(pred_tokens[i:i + n] == gold_tokens for i in range(len(pred_tokens) - n + 1))


@torch.no_grad()
def generate(model, tokenizer, image_features, question, max_new_tokens, device):
    num_image_tokens = model.config.num_image_tokens
    image_block = f"{cfg.ModelSettings.IMAGE_TOKEN * num_image_tokens}\n{question.strip()}"
    # 和 VQAInstructDataset 训练时的拼接格式保持一致，否则评的是 prompt 不匹配而不是能力。
    prompt = f"USER: {image_block}\nASSISTANT: "
    encoded = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    generated = model.generate(
        input_ids=encoded["input_ids"],
        image_features=image_features,
        attention_mask=encoded["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,  # 评估要可复现，不采样
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[0][encoded["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = resolve_device()
    dtype = resolve_dtype(device, args.dtype)

    with open(args.eval_set, "r", encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    # 按图片分组，同一张图只过一次 vision tower + projector。
    by_image = defaultdict(list)
    for item in items:
        by_image[item["image"]].append(item)
    logger.info("Eval set: %d conversations over %d images", len(items), len(by_image))

    model, tokenizer, image_processor, is_stage2 = load_model(args, device, dtype)
    if not is_stage2:
        logger.warning(
            "Checkpoint has no LoRA adapter — this is a Stage-1 checkpoint. It was never "
            "trained on the USER:/ASSISTANT: format, so treat the numbers as a floor."
        )

    n = exact = contained = 0
    dump_fp = open(args.dump, "w", encoding="utf-8") if args.dump else None
    try:
        for image_name, group in by_image.items():
            image_path = os.path.join(args.image_dir, image_name)
            if not os.path.exists(image_path):
                continue
            image_features = encode_image(model, image_processor, image_path, device)
            for item in group:
                convs = item["conversations"]
                for i in range(0, len(convs), 2):
                    question = convs[i]["value"].replace(cfg.ModelSettings.IMAGE_TOKEN, "").strip()
                    gold = convs[i + 1]["value"].strip()
                    pred = generate(model, tokenizer, image_features, question,
                                    args.max_new_tokens, device)
                    pred_tokens, gold_tokens = normalize(pred), normalize(gold)
                    is_exact = pred_tokens == gold_tokens
                    is_contained = contains(pred_tokens, gold_tokens)
                    n += 1
                    exact += is_exact
                    contained += is_contained
                    if dump_fp:
                        dump_fp.write(json.dumps({
                            "image": image_name, "question": question,
                            "gold": gold, "pred": pred,
                            "exact": is_exact, "contained": is_contained,
                        }, ensure_ascii=False) + "\n")
                    if n % 100 == 0:
                        logger.info("%d questions: exact %.1f%%  contains %.1f%%",
                                    n, 100 * exact / n, 100 * contained / n)
    finally:
        if dump_fp:
            dump_fp.close()

    if not n:
        logger.error("No questions were evaluated — check --image_dir")
        return
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"questions:  {n}")
    print(f"exact match: {exact}/{n} = {100 * exact / n:.2f}%")
    print(f"containment: {contained}/{n} = {100 * contained / n:.2f}%")


if __name__ == "__main__":
    main()
