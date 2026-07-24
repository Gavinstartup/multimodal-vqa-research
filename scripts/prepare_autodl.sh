#!/usr/bin/env bash
# AutoDL GPU 训练环境准备：安装依赖 + 预下载模型权重。
# torch 使用镜像自带的 CUDA 版本，不要被 requirements.txt 里的默认索引覆盖安装。
set -euo pipefail

# 国内网络访问 huggingface.co 较慢/不稳定，CLIP 体积小仍走 HF 镜像端点
export HF_ENDPOINT="https://hf-mirror.com"

pip install -r requirements.txt

# Qwen3-8B 体积大（bf16 ~16GB），国内网络下默认从魔搭（ModelScope）下载，比 HF 快很多；
# 想强制走 HF 的话设 MODEL_SOURCE=hf 再跑本脚本。
if [ "${MODEL_SOURCE:-modelscope}" = "modelscope" ]; then
    pip install modelscope
    python - <<'PY'
from modelscope import snapshot_download

qwen_path = snapshot_download("Qwen/Qwen3-8B")
print(f"Qwen3-8B downloaded to: {qwen_path}")
with open("models_paths.env", "w", encoding="utf-8") as f:
    f.write(f'export TEXT_MODEL_PATH="{qwen_path}"\n')
PY
    echo "Qwen3-8B 已通过 ModelScope 下载。训练/推理时用本地路径而不是仓库名："
    echo "  source models_paths.env"
    echo '  python -m src.train.train_stage1 --text_model "$TEXT_MODEL_PATH" ...'
else
    python - <<'PY'
from huggingface_hub import snapshot_download

print("downloading Qwen/Qwen3-8B from HuggingFace ...")
snapshot_download("Qwen/Qwen3-8B")
PY
fi

python - <<'PY'
from huggingface_hub import snapshot_download

print("downloading openai/clip-vit-large-patch14-336 ...")
snapshot_download("openai/clip-vit-large-patch14-336")
PY

# Stage-1 对齐数据集（LLaVA-Pretrain, blip_laion_cc_sbu_558k）体积较大（图片 zip ~20GB），
# 默认不下载；需要时设置 DOWNLOAD_STAGE1_DATA=1 再跑本脚本。
if [ "${DOWNLOAD_STAGE1_DATA:-0}" = "1" ]; then
    python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in ("blip_laion_cc_sbu_558k.json", "images.zip"):
    print(f"downloading LLaVA-Pretrain/{filename} ...")
    hf_hub_download(repo_id="liuhaotian/LLaVA-Pretrain", filename=filename, repo_type="dataset", local_dir="data")
PY
    unzip -q -o data/images.zip -d data/images
fi

echo "AutoDL environment ready."
