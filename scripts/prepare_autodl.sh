#!/usr/bin/env bash
# AutoDL GPU 训练环境准备：安装依赖 + 预下载模型权重。
# torch 使用镜像自带的 CUDA 版本，不要被 requirements.txt 里的默认索引覆盖安装。
set -euo pipefail

# AutoDL 系统盘（/）通常只有 30GB 左右，Qwen3-8B(~17GB) + CLIP(~1.7GB) 加起来就快占满，
# 必须全部落在数据盘上，不能用默认的 ~/.cache（否则会悄悄写满系统盘）。DOWNLOAD_STAGE1_DATA=1
# 时还会临时下载 images.zip(魔搭源 ~27.4GB / HF 源 ~20GB，只解压 NUM_SAMPLES 张后立刻删掉)，
# 峰值在 ~50GB 左右（50GB 数据盘几乎没有余量，建议数据盘至少留 60GB+）。DOWNLOAD_STAGE2_DATA=1
# 会再临时下载 COCO train2017.zip(~18GB，同样只解压用到的图片后删掉)，两段都跑的话峰值还要
# 再高一截，建议数据盘留 80GB+。DATA_DISK 默认指向 AutoDL 的数据盘路径，其他平台按需覆盖。
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

# 这几个变量只在本次脚本进程里生效，训练/推理是在新开的 shell 里跑的，不 source 一下的话
# CLIPImageProcessor.from_pretrained 等调用会直接去连真正的 huggingface.co、连不到本地缓存。
# 写进 models_paths.env，后面新开 shell 跑训练/推理前先 `source models_paths.env` 即可。
cat > models_paths.env <<ENVEOF
export HF_HOME="$HF_HOME"
export HF_ENDPOINT="$HF_ENDPOINT"
export MODELSCOPE_CACHE="$MODELSCOPE_CACHE"
ENVEOF

pip install -r requirements.txt

# Qwen3-8B 体积大（bf16 ~16GB），国内网络下默认从魔搭（ModelScope）下载，比 HF 快很多；
# 想强制走 HF 的话设 MODEL_SOURCE=hf 再跑本脚本。
if [ "${MODEL_SOURCE:-modelscope}" = "modelscope" ]; then
    pip install modelscope
    python - <<'PY'
from modelscope import snapshot_download

qwen_path = snapshot_download("Qwen/Qwen3-8B")
print(f"Qwen3-8B downloaded to: {qwen_path}")
with open("models_paths.env", "a", encoding="utf-8") as f:
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

# 这个仓库没发布 safetensors，权重是 pytorch_model.bin(~1.7GB) 和 tf_model.h5(~1.7GB) 两份，
# 后者 CLIPVisionModel 根本不会读，只下需要的那份。
#
# HF_HUB_DISABLE_XET=1 是重点：huggingface_hub 新版默认用 Xet 后端，它在内存里缓冲分块，下
# 1.7GB 的权重时峰值会顶爆 AutoDL 无卡模式的 2GB cgroup 配额，被 OOM killer 干掉——表现为一
# 句没头没尾的 "Killed"，配合 set -e 直接让整个脚本中止在这里。退回普通 HTTP 流式下载后内存
# 占用和文件大小无关。并发也收到 1，进一步压低峰值。
HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download

print("downloading openai/clip-vit-large-patch14-336 ...")
snapshot_download(
    "openai/clip-vit-large-patch14-336",
    ignore_patterns=["tf_model.h5", "*.msgpack", "README.md"],
    max_workers=1,
)
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

# Stage-2 指令微调数据集（LLaVA-Instruct-150K）默认不下载，需要时设置
# DOWNLOAD_STAGE2_DATA=1 再跑本脚本。
#
# annotations 和图片来源不同。annotations 默认走魔搭的 AI-ModelScope/LLaVA-Instruct-150K
# （和 HF 上的 liuhaotian/LLaVA-Instruct-150K 同名同构，验证过文件列表一致），国内快很多，
# 想强制走 HF 设 DATA_SOURCE=hf。只取 llava_instruct_150k.json（~218MB）这一个文件，同仓库
# 里还有个 982MB 的 llava_v1_5_mix665k.json，整仓拖下来纯属浪费。
#
# 图片两边都没有：这两个仓库发布的都只是 json。用的是 COCO train2017 —— json 的 "image"
# 字段是裸的 12 位文件名（如 "000000033471.jpg"），正好是 train2017 的命名约定；train2014
# 里同一张图叫 COCO_train2014_000000033471.jpg，名字对不上。train2017 是 train2014 的超集，
# 150K 用到的 id 都覆盖得到。所以图片只能从 images.cocodataset.org 官方地址下（~18GB，国内
# 网络下这是整个流程最慢的一步，慢的话可以手动下好 train2017.zip 放到 data/ 下再重跑本段）。
# 同样是下完整 zip、只解压用到的图片子集、用完立刻删 zip，做法跟 Stage-1 的 images.zip 一致。
#
# 注意 Stage-2 的 COCO 图片和 Stage-1 的 LAION/CC/SBU 图片共用 data/images/。文件名不冲突
# （Stage-1 的带 00453/ 这样的子目录前缀），但磁盘占用是叠加的。
if [ "${DOWNLOAD_STAGE2_DATA:-0}" = "1" ]; then
    if [ "${DATA_SOURCE:-modelscope}" = "modelscope" ]; then
        pip install modelscope  # 若 MODEL_SOURCE=hf 时没装过，这里兜底装一下；已装则秒过
        python - <<'PY'
from modelscope import dataset_snapshot_download

print("downloading AI-ModelScope/LLaVA-Instruct-150K/llava_instruct_150k.json ...")
dataset_snapshot_download(
    "AI-ModelScope/LLaVA-Instruct-150K", local_dir="data",
    allow_file_pattern="llava_instruct_150k.json",
)
PY
    else
        python - <<'PY'
from huggingface_hub import hf_hub_download

print("downloading LLaVA-Instruct-150K/llava_instruct_150k.json ...")
hf_hub_download(
    repo_id="liuhaotian/LLaVA-Instruct-150K", filename="llava_instruct_150k.json",
    repo_type="dataset", local_dir="data",
)
PY
    fi

    # 全量 annotations 单独留一份：采样结果原地覆写的话，之后想换更大的采样数就只能重新下。
    if [ ! -f data/llava_instruct_150k_full.json ]; then
        mv data/llava_instruct_150k.json data/llava_instruct_150k_full.json
    fi

    # 先下到 .part 再改名：18GB 在国内断一次很常见，直接下成最终文件名的话，残留的半个 zip
    # 会让上面这种 `[ ! -f ... ]` 判断误以为已经下好，跳过下载、到解压那步才报一个莫名其妙的
    # 错。curl -C - 支持断点续传，重跑本段会接着上次的进度下。
    if [ ! -f data/train2017.zip ]; then
        echo "downloading COCO train2017 images (~18GB) ..."
        curl -L -C - -o data/train2017.zip.part http://images.cocodataset.org/zips/train2017.zip
        # zip 的中央目录在文件末尾，能读出目录就说明下全了
        python -c "import zipfile, sys; zipfile.ZipFile('data/train2017.zip.part').namelist(); print('zip ok')"
        mv data/train2017.zip.part data/train2017.zip
    fi

    # Stage-1 那段用的也叫 NUM_SAMPLES，这里单独开一个变量，两段才能在同一次运行里各取各的值。
    STAGE2_NUM_SAMPLES="${STAGE2_NUM_SAMPLES:-50000}"
    pip install ijson  # 流式 json 解析，见下面的注释；已装则秒过
    python - <<PY
import json
import os
import random
import shutil
import zipfile

import ijson

# json.load() 这个 218MB 的数组会爆内存：Python 的 dict/str 对象开销通常是文件体积的 3~5 倍，
# 而 AutoDL 无卡模式的 cgroup 配额只有 2GB。改成 ijson 边解析边做蓄水池抽样，常驻内存只和
# STAGE2_NUM_SAMPLES 成正比，和文件多大无关。use_float 避免 ijson 把数字解析成 Decimal
# （json.dump 不认识 Decimal）。
random.seed(42)
n_target = $STAGE2_NUM_SAMPLES

reservoir = []
total = 0
with open("data/llava_instruct_150k_full.json", "rb") as f:
    for item in ijson.items(f, "item", use_float=True):
        total += 1
        if len(reservoir) < n_target:
            reservoir.append(item)
        else:
            j = random.randrange(total)
            if j < n_target:
                reservoir[j] = item

with open("data/llava_instruct_150k.json", "w", encoding="utf-8") as f:
    json.dump(reservoir, f, ensure_ascii=False)

image_names = {item["image"] for item in reservoir}
del reservoir  # 抽样结果已经落盘，解压阶段只需要文件名集合
print(f"sampled {len(image_names)} unique images from {total} conversations, extracting ...")

os.makedirs("data/images", exist_ok=True)
missing = 0
with zipfile.ZipFile("data/train2017.zip") as zf:
    for name in image_names:
        dest = os.path.join("data/images", name)
        if os.path.exists(dest):
            continue  # a previous run may already have extracted this file
        try:
            with zf.open(f"train2017/{name}") as src, open(dest, "wb") as out:
                shutil.copyfileobj(src, out)
        except KeyError:
            missing += 1
if missing:
    print(f"warning: {missing} image(s) absent from train2017.zip; VQAInstructDataset will skip those samples")
print("subset extraction done")
PY

    rm -f data/train2017.zip
    echo "数据集就绪（随机采样 $STAGE2_NUM_SAMPLES 条），annotations: data/llava_instruct_150k.json  images: data/images"
fi

echo "最终磁盘占用："
df -h "$DATA_DISK"
echo "AutoDL environment ready."
