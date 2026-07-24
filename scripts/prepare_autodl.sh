#!/usr/bin/env bash
# AutoDL GPU 训练环境准备：安装依赖 + 预下载模型权重。
# torch 使用镜像自带的 CUDA 版本，不要被 requirements.txt 里的默认索引覆盖安装。
set -euo pipefail

# 国内网络访问 huggingface.co 较慢/不稳定，走镜像端点
export HF_ENDPOINT="https://hf-mirror.com"

pip install -r requirements.txt

python - <<'PY'
from huggingface_hub import snapshot_download

for repo_id in ("openai/clip-vit-large-patch14-336", "Qwen/Qwen3-8B"):
    print(f"downloading {repo_id} ...")
    snapshot_download(repo_id)
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
