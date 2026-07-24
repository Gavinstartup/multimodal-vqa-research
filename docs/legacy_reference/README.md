# 旧版代码参考（LLaMA + open_clip 手写版）

这些文件是从旧项目（`Research Project/123/Script`）复制过来的原样代码，**不是新架构的一部分**，
只作为重写时的工程参考，避免把有价值的工程细节（参数配置项、数据集处理框架、训练循环控制逻辑）丢掉。

## 各文件的可复用程度

- **training_projector.py** — 复用价值最高。数据集加载/校验框架（`ImageTextDataset`）、
  早停、`ReduceLROnPlateau` 调度器、梯度累积、checkpoint 保存策略这些工程部分可以照搬。
  **但 loss 计算部分（cosine + contrastive 对齐 pooled 文本 embedding）必须重写**为端到端
  LM cross-entropy loss，这是新版训练脚本的核心改动。

- **config.py** — 参数项分类方式（ModelSettings/TrainingSettings/GenerationSettings等）值得参考，
  但内容要整体替换为 Qwen3-8B + CLIP-ViT-L/14-336 的新配置，`AttentionMaskSettings` 那部分
  自定义掩码设计整个废弃（详见下方说明）。

- **process_text_input.py** — `<image>` 占位符定位逻辑（`process_text_input_llava` 里查找
  image_token 位置的部分）思路可复用，但要改成：(1) 支持把单个占位符展开成 N 个 patch token
  位置 (2) 构造 `labels`（question 部分设为 `-100`，只在 answer 部分算 loss）。

- **llava_fusion.py** — token 拼接的思路参考（把图像特征替换进文本 embedding 的指定位置），
  但要从 Python for 循环改成向量化的 `masked_scatter`，并支持每张图多个 patch token。

- **encoder_ViT.py** — **不复用**，改用 `transformers.CLIPVisionModel.from_pretrained(...)`
  直接拿 patch-level hidden_states，不用 `open_clip` 手写封装。

- **mlp_projector.py** — **不直接复用**，新版按 LLaVA 官方做法简化为 2 层 MLP + GELU
  （旧版是 5 层 ReLU，容易在小数据集上过拟合）。

## 已知问题（不要带进新版）

1. 训练目标错误：projector 对齐的是文本 token embedding 的均值，而不是对生成结果有监督的信号。
2. 只用 CLIP 的 pooled 全局向量（1个token），丢失空间细节，新版改用 patch-level 多 token。
3. `AttentionMaskSettings` 里的自定义图像/文本双向可见性配置从未被实际使用，是死代码。
4. `llava_fusion.py` 里融合后的 attention_mask 被硬编码覆盖为全 1，丢弃了原始 padding 信息。
5. 模型加载在多处未做全局缓存，每次推理/生成都重新读一次权重文件，是主要的高延迟来源。
