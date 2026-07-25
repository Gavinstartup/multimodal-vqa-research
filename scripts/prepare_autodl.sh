#!/usr/bin/env bash
# AutoDL GPU 训练环境准备：安装依赖 + 预下载模型权重。
# torch 使用镜像自带的 CUDA 版本，不要被 requirements.txt 里的默认索引覆盖安装。
set -euo pipefail

# AutoDL 系统盘（/）通常只有 30GB 左右，Qwen3-8B(~17GB) + CLIP(~1.7GB) 加起来就快占满，
# 必须全部落在数据盘上，不能用默认的 ~/.cache（否则会悄悄写满系统盘）。DOWNLOAD_STAGE1_DATA=1
# 时还会临时下载 images.zip(魔搭源 ~27.4GB / HF 源 ~20GB，只解压 NUM_SAMPLES 张后立刻删掉)，
# 峰值在 ~50GB 左右（50GB 数据盘几乎没有余量，建议数据盘至少留 60GB+）。DATA_DISK 默认指向
# AutoDL 的数据盘路径，其他平台按需覆盖。
# 前提：整个项目目录本身也要 clone/上传到数据盘下（例如 /root/autodl-tmp/VQA），这样
# configs/stage1_config.py 里默认的相对路径 data/ 才会落在数据盘，而不是系统盘。
DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
mkdir -p "$DATA_DISK"
export HF_HOME="$DATA_DISK/hf_cache"
export MODELSCOPE_CACHE="$DATA_DISK/modelscope_cache"
mkdir -p data

echo "缓存目录: $DATA_DISK  当前工作目录: $(pwd)"
df -h "$DATA_DISK"

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

echo "模型下载完成后的磁盘占用："
df -h "$DATA_DISK"

# Stage-1 对齐数据集（LLaVA-Pretrain, blip_laion_cc_sbu_558k）默认不下载，需要时设置
# DOWNLOAD_STAGE1_DATA=1 再跑本脚本。这个仓库只发布了完整 558K 版本，没有现成的小子集，
# 所以还是要下载完整的 annotations json + images.zip（zip 本身无法只下载一部分），下载后
# 从 558K 里随机采样 NUM_SAMPLES 条（默认 15 万），只解压这些样本用到的图片（不是全部
# 558K 张），大幅降低解压后的磁盘占用；解压用完的 zip 立刻删掉。
# 国内网络下默认从魔搭的 AI-ModelScope/LLaVA-Pretrain 下载（文件结构跟 HF 上的
# liuhaotian/LLaVA-Pretrain 一致，验证过 blip_laion_cc_sbu_558k.json + images.zip 同名同构）；
# 想强制走 HF 的话设 DATA_SOURCE=hf 再跑本脚本。
if [ "${DOWNLOAD_STAGE1_DATA:-0}" = "1" ]; then
    if [ "${DATA_SOURCE:-modelscope}" = "modelscope" ]; then
        pip install modelscope  # 若 MODEL_SOURCE=hf 时没装过，这里兜底装一下；已装则秒过
        python - <<'PY'
from modelscope import dataset_snapshot_download

print("downloading AI-ModelScope/LLaVA-Pretrain ...")
dataset_snapshot_download("AI-ModelScope/LLaVA-Pretrain", local_dir="data")
PY
    else
        python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in ("blip_laion_cc_sbu_558k.json", "images.zip"):
    print(f"downloading LLaVA-Pretrain/{filename} ...")
    hf_hub_download(repo_id="liuhaotian/LLaVA-Pretrain", filename=filename, repo_type="dataset", local_dir="data")
PY
    fi

    NUM_SAMPLES="${NUM_SAMPLES:-150000}"
    python - <<PY
import json
import random
import zipfile

random.seed(42)
with open("data/blip_laion_cc_sbu_558k.json", encoding="utf-8") as f:
    full = json.load(f)

n = min($NUM_SAMPLES, len(full))
subset = random.sample(full, n)
with open("data/blip_laion_cc_sbu_558k.json", "w", encoding="utf-8") as f:
    json.dump(subset, f, ensure_ascii=False)

image_names = {item["image"] for item in subset}
print(f"sampled {len(subset)} / {len(full)} annotations, extracting {len(image_names)} images ...")
with zipfile.ZipFile("data/images.zip") as zf:
    for name in image_names:
        zf.extract(name, "data/images")
print("subset extraction done")
PY

    rm -f data/images.zip  # 只留子集用到的图片，zip 本体（含另外 ~76% 用不到的图片）删掉省空间
    echo "数据集就绪（随机采样 $NUM_SAMPLES 条），annotations: data/blip_laion_cc_sbu_558k.json  images: data/images"
fi

echo "最终磁盘占用："
df -h "$DATA_DISK"
echo "AutoDL environment ready."
