# Multimodal VQA Research

LLaVA 风格多模态视觉问答系统重构版：CLIP ViT + MLP Projector + Qwen3-8B，
按 HuggingFace 多模态模型规范封装，端到端语言模型 loss 训练 projector。

## 架构

- **Vision Tower**: CLIP ViT-L/14-336（patch-level 特征，非 pooled 全局向量）
- **Projector**: MLP，将图像 patch 特征投影到 Qwen3-8B 的 embedding 空间
- **Language Model**: Qwen3-8B（训练时冻结，只更新 projector）

## 训练策略

- **Stage-1（已完成）**：projector 对齐预训练。图像 → vision tower → projector → 替换
  文本序列中 `<image>` 占位符 → 过 Qwen3-8B → 对 answer 部分 token 计算 next-token
  cross-entropy → 只反传更新 projector，vision tower / language model 全程冻结。
- **Stage-2（进行中）**：指令微调。在 Stage-1 权重基础上，用 LLaVA-Instruct-150K 风格的
  多轮指令数据（`src/data/VQAInstructDataset`，`USER: .../ASSISTANT: ...` 多轮拼接，只对
  ASSISTANT 段计算 loss）继续训练，projector 全量更新 + language model 用 LoRA（`peft`）
  微调，让模型学会针对具体问题作答，而不只是生成通用图片描述。

  ```
  # 环境准备时顺带下载指令数据（annotations 走魔搭 ~218MB，COCO train2017 图片走官方源 ~18GB）
  DOWNLOAD_STAGE2_DATA=1 bash scripts/prepare_autodl.sh

  python -m src.train.train_stage2 \
      --image_dir data/images --annotations data/llava_instruct_150k.json \
      --stage1_checkpoint outputs/stage1_projector/final_model
  ```

  产出的 `final_model/`（config + tokenizer + `projector.pt` + LoRA adapter）可以直接传给
  `generate.py`/`serve_repl.py` 的 `--checkpoint`：两个脚本都会自动探测目录里是否有
  `adapter_config.json`，据此决定加载 LoRA 并切换到 Stage-2 训练时用的 prompt 格式。

## 模型权重

Stage-1 训练产物（`config.json` + tokenizer + `projector.pt`）已发布在 HuggingFace：
[GavinTSFMR/multimodal-vqa-stage1](https://huggingface.co/GavinTSFMR/multimodal-vqa-stage1)。
仓库里只有权重，不含代码——用之前先 clone 本仓库，再把下载下来的目录传给
`src/inference/generate.py` 的 `--checkpoint` 参数。

## 目录结构

```
src/
  model/       模型架构定义（config + modeling）
  data/        数据集加载与预处理
  train/       训练脚本
  inference/   推理脚本
scripts/       AutoDL 环境准备等辅助脚本
configs/       训练/模型配置文件
docs/          设计笔记
```

## 状态

Stage-1 projector 对齐训练已完成并验证（`src/inference/serve_repl.py` 常驻推理 REPL
定性抽查：模型能看图生成语义相关的描述），权重已发布到 HuggingFace。Stage-2
LoRA 指令微调的代码（`src/train/train_stage2.py`、`VQAInstructDataset`、
`VQAForConditionalGeneration.prepare_for_stage2`）已就绪，训练尚未在 AutoDL 上跑起来。
