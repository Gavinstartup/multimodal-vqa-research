"""从 Stage-2 训练集里切出一份短答案评估集，并写出排除了它的训练集。

为什么需要它：换数据配方之后，判断「视觉 grounding 有没有变好」不能只看 val_loss ——
不同数据集的 loss 不可比，而定性抽查几十条只能看出趋势、看不出退步。短答案题（VQAv2 /
OKVQA 那类「图里有几只猫」）的标准答案只有一两个词，可以直接算准确率，是这份数据里唯一
能给出数字的部分。

切出来的样本会从训练集里剔除，否则测的是记忆而不是泛化。

跑法（项目根目录）：

    python scripts/make_shortqa_eval.py \
        --annotations data/llava_v1_5_mix665k_coco.json \
        --train_out data/llava_v1_5_mix665k_coco_train.json \
        --eval_out data/eval_shortqa.json

两趟流式扫描（ijson），常驻内存只和 --num_eval 成正比，与文件大小无关。
"""

import argparse
import json
import random

import ijson


def parse_args():
    parser = argparse.ArgumentParser(description="Carve a short-answer eval set out of the training data")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--eval_out", required=True)
    parser.add_argument("--num_eval", type=int, default=2000)
    parser.add_argument("--max_answer_words", type=int, default=3,
                        help="所有 gpt 轮都不超过这个词数才算短答案题")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_short_answer(item, max_words):
    convs = item.get("conversations") or []
    if len(convs) < 2 or len(convs) % 2:
        return False
    answers = [convs[i]["value"].strip() for i in range(1, len(convs), 2)]
    return all(answers) and all(len(a.split()) <= max_words for a in answers)


def key_of(item):
    # image + 第一个 human turn 足以唯一标识一条对话：同一张图会被多条不同任务的样本引用。
    return (item.get("image"), item["conversations"][0]["value"])


def main():
    args = parse_args()
    random.seed(args.seed)

    # 第一趟：在短答案样本里做蓄水池抽样。
    reservoir = []
    seen_short = 0
    total = 0
    with open(args.annotations, "rb") as f:
        for item in ijson.items(f, "item", use_float=True):
            total += 1
            if not is_short_answer(item, args.max_answer_words):
                continue
            seen_short += 1
            if len(reservoir) < args.num_eval:
                reservoir.append(item)
            else:
                j = random.randrange(seen_short)
                if j < args.num_eval:
                    reservoir[j] = item

    with open(args.eval_out, "w", encoding="utf-8") as f:
        json.dump(reservoir, f, ensure_ascii=False)
    held_out = {key_of(item) for item in reservoir}
    print(f"短答案样本 {seen_short}/{total}，抽出 {len(reservoir)} 条写入 {args.eval_out}")
    del reservoir

    # 第二趟：把剩下的写成训练集。同样是增量写，不把 36 万条同时放在内存里。
    kept = 0
    with open(args.annotations, "rb") as src, open(args.train_out, "w", encoding="utf-8") as out:
        out.write("[")
        for item in ijson.items(src, "item", use_float=True):
            if key_of(item) in held_out:
                continue
            if kept:
                out.write(",")
            json.dump(item, out, ensure_ascii=False)
            kept += 1
        out.write("]")
    print(f"训练集 {kept} 条写入 {args.train_out}（剔除了 {total - kept} 条）")


if __name__ == "__main__":
    main()
