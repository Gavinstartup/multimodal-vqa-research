"""Stage-2 指令数据的形态体检。

换数据配方（比如从 LLaVA-Instruct-150K 换到 llava_v1_5_mix665k 的 COCO 子集）时，
VQAInstructDataset 的几条隐含假设未必还成立，而它们失效时都是**静默**的：

- `conversations` 长度必须是偶数 —— 否则 `_has_valid_schema` 直接丢样本，只留一行 warning；
- `<image>` 只出现在第一个 human turn —— 其他位置的标记不会被剥掉，会被当成普通文本 token；
- 样本 token 数超过 max_length 时整轮丢弃 —— 丢得太多等于白下数据。

跑法（项目根目录）：

    source models_paths.env
    python scripts/inspect_stage2_data.py --annotations data/llava_v1_5_mix665k_coco.json

用 ijson 流式读，常驻内存与文件大小无关（AutoDL 无卡模式只有 2GB 配额，
而 mix665k 的 COCO 子集有 800MB+）。token 统计默认抽样，因为给几十万条样本
全量跑一遍 tokenizer 要好几分钟，而分布形状用一万条就足够看清楚。
"""

import argparse
import os
import random
from collections import Counter

import ijson


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Stage-2 instruction data shape")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image_dir", default="data/images")
    parser.add_argument("--tokenizer", default=os.environ.get("TEXT_MODEL_PATH"),
                        help="本地 Qwen3 目录；不传就跳过 token 长度统计")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_image_tokens", type=int, default=576,
                        help="图像占位符的 token 数，计入每条样本的长度预算")
    parser.add_argument("--token_sample", type=int, default=10000,
                        help="参与 token 长度统计的样本数（0 表示全量）")
    parser.add_argument("--image_token", default="<image>")
    return parser.parse_args()


def main():
    args = parse_args()

    total = 0
    odd_turns = 0
    missing_image_key = 0
    marker_in_first = 0
    marker_elsewhere = 0
    no_marker = 0
    turn_counts = Counter()
    missing_files = 0
    checked_files = 0

    random.seed(0)
    reservoir = []

    with open(args.annotations, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            total += 1
            convs = item.get("conversations") or []
            turn_counts[len(convs)] += 1
            if len(convs) % 2 != 0 or len(convs) < 2:
                odd_turns += 1
                continue

            image = item.get("image")
            if not image:
                missing_image_key += 1
            elif checked_files < 2000:
                # 全量 os.path.exists 在 36 万条上要几十秒，抽头 2000 条够发现路径错配了
                checked_files += 1
                if not os.path.exists(os.path.join(args.image_dir, image)):
                    missing_files += 1

            first_has = args.image_token in convs[0].get("value", "")
            rest_has = any(args.image_token in c.get("value", "") for c in convs[1:])
            if rest_has:
                marker_elsewhere += 1
            elif first_has:
                marker_in_first += 1
            else:
                no_marker += 1

            if args.token_sample:
                if len(reservoir) < args.token_sample:
                    reservoir.append(item)
                else:
                    j = random.randrange(total)
                    if j < args.token_sample:
                        reservoir[j] = item
            else:
                reservoir.append(item)

    print(f"总条数: {total}")
    print(f"  轮数为奇数/不足两轮（会被静默丢弃）: {odd_turns}")
    print(f"  缺 image 字段: {missing_image_key}")
    print(f"  图片文件缺失: {missing_files}/{checked_files} (抽查)")
    print(f"  <image> 只在第一轮: {marker_in_first}")
    print(f"  <image> 出现在后续轮次（现有代码不会剥离）: {marker_elsewhere}")
    print(f"  完全没有 <image> 标记: {no_marker}")
    print("  轮数分布 (turns -> count):",
          dict(sorted(turn_counts.items())[:12]), "..." if len(turn_counts) > 12 else "")

    if not args.tokenizer:
        print("\n未提供 --tokenizer，跳过 token 长度统计")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    print(f"\n用 {len(reservoir)} 条样本估计 token 长度（max_length={args.max_length}，"
          f"其中 {args.num_image_tokens} 个是图像占位符）")

    lengths = []
    fully_dropped = 0
    partially_dropped = 0
    for item in reservoir:
        convs = item["conversations"]
        # 复刻 VQAInstructDataset.__getitem__ 的预算计算，但不真的建 tensor
        used = args.num_image_tokens
        kept_turns = 0
        overflowed = False
        for i in range(0, len(convs), 2):
            human = convs[i]["value"].replace(args.image_token, "").strip()
            gpt = convs[i + 1]["value"].strip()
            prefix = f"USER: {human}\nASSISTANT: " if i else f"\n{human}\nASSISTANT: "
            turn_len = (len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
                        + len(tokenizer(gpt, add_special_tokens=False)["input_ids"]) + 1)
            if kept_turns and used + turn_len > args.max_length:
                overflowed = True
                break
            used += turn_len
            kept_turns += 1
        lengths.append(used)
        if kept_turns == 0:
            fully_dropped += 1
        elif overflowed:
            partially_dropped += 1

    lengths.sort()
    n = len(lengths)
    def pct(p):
        return lengths[min(n - 1, int(n * p))]
    print(f"  token 长度 p50={pct(0.5)} p90={pct(0.9)} p99={pct(0.99)} max={lengths[-1]}")
    print(f"  有轮次被丢弃的样本: {partially_dropped}/{n} ({100 * partially_dropped / n:.1f}%)")
    print(f"  一轮都放不下的样本: {fully_dropped}/{n}")


if __name__ == "__main__":
    main()
