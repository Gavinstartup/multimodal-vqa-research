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
- **Stage-2（规划中）**：指令微调。在 Stage-1 权重基础上，用指令型问答数据（如
  LLaVA-Instruct-150K）继续训练，projector 全量更新 + language model 用 LoRA
  微调，让模型学会针对具体问题作答，而不只是生成通用图片描述。

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
定性抽查：模型能看图生成语义相关的描述），权重已发布到 HuggingFace。当前推进
Stage-2 指令微调，让模型学会针对具体问题作答。
