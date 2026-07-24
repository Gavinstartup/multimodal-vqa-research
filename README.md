# Multimodal VQA Research

LLaVA 风格多模态视觉问答系统重构版：CLIP ViT + MLP Projector + Qwen3-8B，
按 HuggingFace 多模态模型规范封装，端到端语言模型 loss 训练 projector。

## 架构

- **Vision Tower**: CLIP ViT-L/14-336（patch-level 特征，非 pooled 全局向量）
- **Projector**: MLP，将图像 patch 特征投影到 Qwen3-8B 的 embedding 空间
- **Language Model**: Qwen3-8B（训练时冻结，只更新 projector）

## 训练策略

Stage-1 对齐预训练：图像 → vision tower → projector → 替换文本序列中 `<image>` 占位符
→ 过 Qwen3-8B → 对 answer 部分 token 计算 next-token cross-entropy → 只反传更新 projector。

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

项目重构中，从旧版手写 LLaMA+CLIP pipeline 迁移到规范化的 Qwen3-8B 架构。
