#!/usr/bin/env bash
# AutoDL GPU 训练环境准备：安装依赖 + 预下载模型权重。
# torch 使用镜像自带的 CUDA 版本，不要被 requirements.txt 里的默认索引覆盖安装。
set -euo pipefail

# AutoDL 系统盘（/）通常只有 30GB 左右，Qwen3-8B(~17GB) + CLIP(~1.7GB) 加起来就快占满，
# 必须全部落在数据盘上，不能用默认的 ~/.cache（否则会悄悄写满系统盘）。DOWNLOAD_STAGE1_DATA=1
# 时还会临时下载 images.zip(魔搭源 ~27.4GB / HF 源 ~20GB，只解压 NUM_SAMPLES 张后立刻删掉)，
# 峰值在 ~50GB 左右（50GB 数据盘几乎没有余量，建议数据盘至少留 60GB+）。DOWNLOAD_STAGE2_DATA=1
# 会再临时下载 COCO train2017.zip(~19.3GB，同样只解压用到的图片后删掉)，两段都跑的话峰值还要
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
    echo '  python -m src.train.train_stage1 --text_model "$TEXT_MODEL_PATH" --vision_model "$VISION_MODEL_PATH" ...'
else
    python - <<'PY'
from huggingface_hub import snapshot_download

print("downloading Qwen/Qwen3-8B from HuggingFace ...")
snapshot_download("Qwen/Qwen3-8B")
PY
fi

# CLIP ViT-L/14-336 的权重只有 pytorch_model.bin(~1.7GB) 和 tf_model.h5(~1.7GB) 两份，没发布
# safetensors；后者 CLIPVisionModel 根本不会读，两条路径都排除掉，省一半流量。
# 和 Qwen3-8B 一样默认走魔搭（AI-ModelScope/clip-vit-large-patch14-336 与 HF 上的
# openai/clip-vit-large-patch14-336 同名同构，pytorch_model.bin 字节数一致），MODEL_SOURCE=hf
# 切回 HF。两边都把最终路径写进 models_paths.env，训练/推理统一用 --vision_model 传。
if [ "${MODEL_SOURCE:-modelscope}" = "modelscope" ]; then
    python - <<'PY'
from modelscope import snapshot_download

print("downloading AI-ModelScope/clip-vit-large-patch14-336 ...")
clip_path = snapshot_download(
    "AI-ModelScope/clip-vit-large-patch14-336",
    ignore_file_pattern=["tf_model.h5"],
)
print(f"CLIP downloaded to: {clip_path}")
with open("models_paths.env", "a", encoding="utf-8") as f:
    f.write(f'export VISION_MODEL_PATH="{clip_path}"\n')
PY
else
    # HF_HUB_DISABLE_XET=1 是重点：huggingface_hub 新版默认用 Xet 后端，它在内存里缓冲分块，
    # 下 1.7GB 权重时峰值会顶爆 AutoDL 无卡模式的 2GB cgroup 配额，被 OOM killer 干掉——表现
    # 为一句没头没尾的 "Killed"，配合 set -e 直接让整个脚本中止在这里。退回普通 HTTP 流式下
    # 载后内存占用和文件大小无关。并发也收到 1，进一步压低峰值。
    HF_HUB_DISABLE_XET=1 python - <<'PY'
from huggingface_hub import snapshot_download

print("downloading openai/clip-vit-large-patch14-336 ...")
snapshot_download(
    "openai/clip-vit-large-patch14-336",
    ignore_patterns=["tf_model.h5", "*.msgpack", "README.md"],
    max_workers=1,
)
# 仓库名本身就能被 from_pretrained 解析（已在 HF 缓存里），写进去让两条路径的用法一致。
with open("models_paths.env", "a", encoding="utf-8") as f:
    f.write('export VISION_MODEL_PATH="openai/clip-vit-large-patch14-336"\n')
PY
fi

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
# 图片这两个仓库都没有（发布的都只是 json），得单独下 COCO train2017：json 的 "image" 字段是
# 裸的 12 位文件名（如 "000000033471.jpg"），正好是 train2017 的命名约定；train2014 里同一张
# 图叫 COCO_train2014_000000033471.jpg，名字对不上。train2017 是 train2014 的超集，150K 用到
# 的 id 都覆盖得到。具体下载源见下面那段（默认魔搭，官方源在国内基本不可用）。
# 同样是下完整 zip、只解压用到的图片子集、用完立刻删 zip，做法跟 Stage-1 的 images.zip 一致。
#
# 注意 Stage-2 的 COCO 图片和 Stage-1 的 LAION/CC/SBU 图片共用 data/images/。文件名不冲突
# （Stage-1 的带 00453/ 这样的子目录前缀），但磁盘占用是叠加的。
if [ "${DOWNLOAD_STAGE2_DATA:-0}" = "1" ]; then
    # 两套 Stage-2 数据配方：
    #   instruct150k   —— LLaVA-Instruct-150K，全是 GPT-4 从 caption 生成的长篇描述。
    #   mix665k_coco   —— LLaVA-1.5 的 llava_v1_5_mix665k.json 中指向 coco/train2017 的那
    #                     364,100 条（VQAv2 / OKVQA / A-OKVQA / RefCOCO + 原 150K）。
    # 换配方的原因见 docs/：只用 instruct150k 训出来的模型答得流利但视觉 grounding 很差——
    # 长篇描述里靠语言先验把话说圆就能拿到低 loss，视觉侧收不到纠错信号；mix665k 的短答案和
    # 区域指令数据传不出图像信息就压不下 loss。两条配方共用同一批 COCO train2017 图片。
    STAGE2_RECIPE="${STAGE2_RECIPE:-instruct150k}"

    if [ "$STAGE2_RECIPE" = "mix665k_coco" ]; then
        if [ -f data/llava_v1_5_mix665k.json ]; then
            echo "annotations 已存在（data/llava_v1_5_mix665k.json），跳过下载"
        elif [ "${DATA_SOURCE:-modelscope}" = "modelscope" ]; then
            pip install modelscope  # 若 MODEL_SOURCE=hf 时没装过，这里兜底装一下；已装则秒过
            python - <<'PY'
from modelscope import dataset_snapshot_download

# 和 llava_instruct_150k.json 同一个仓库，不必新增数据源。
print("downloading AI-ModelScope/LLaVA-Instruct-150K/llava_v1_5_mix665k.json (~1.0GB) ...")
dataset_snapshot_download(
    "AI-ModelScope/LLaVA-Instruct-150K", local_dir="data",
    allow_file_pattern="llava_v1_5_mix665k.json",
)
PY
        else
            python - <<'PY'
from huggingface_hub import hf_hub_download

print("downloading LLaVA-Instruct-150K/llava_v1_5_mix665k.json (~1.0GB) ...")
hf_hub_download(
    repo_id="liuhaotian/LLaVA-Instruct-150K", filename="llava_v1_5_mix665k.json",
    repo_type="dataset", local_dir="data",
)
PY
        fi
    # 下面会把 annotations 改名成 _full.json 留底，所以原名文件必然不存在，下载器每次重跑都会
    # 认为要重新拉一遍这 229MB。有留底就直接跳过下载段。
    elif [ -f data/llava_instruct_150k_full.json ]; then
        echo "annotations 已存在（data/llava_instruct_150k_full.json），跳过下载"
    elif [ "${DATA_SOURCE:-modelscope}" = "modelscope" ]; then
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
    # mix665k_coco 走的是过滤而非原地覆写（输出是另一个文件名），不需要这层留底。
    if [ "$STAGE2_RECIPE" != "mix665k_coco" ] && [ ! -f data/llava_instruct_150k_full.json ]; then
        mv data/llava_instruct_150k.json data/llava_instruct_150k_full.json
    fi

    # 图片默认也走魔搭：images.cocodataset.org 在 AutoDL 上实测只有 ~0.06MB/s，19GB 要下三个
    # 月，等于不可用；魔搭的 PAI/COCO2017 是同一个 train2017.zip（19.34GB），实测 ~8.4MB/s，
    # 约 40 分钟。dataset_snapshot_download 自带断点续传和 sha256 校验，不必再手搓 .part。
    # DATA_SOURCE=hf 时回退到官方源（境外网络下官方源更直接），那条路仍然先下 .part 再改名：
    # 直接下成最终文件名的话，断点残留的半个 zip 会让 `[ ! -f ... ]` 误判成已下好，跳过下载、
    # 到解压那步才报一个莫名其妙的错。
    if [ ! -f data/train2017.zip ]; then
        if [ "${DATA_SOURCE:-modelscope}" = "modelscope" ]; then
            echo "downloading COCO train2017 images (~19GB) from ModelScope ..."
            python - <<'PY'
from modelscope import dataset_snapshot_download

dataset_snapshot_download(
    "PAI/COCO2017", local_dir="data", allow_file_pattern="train2017.zip",
)
PY
        else
            echo "downloading COCO train2017 images (~19GB) from images.cocodataset.org ..."
            curl -L -C - -o data/train2017.zip.part http://images.cocodataset.org/zips/train2017.zip
            # zip 的中央目录在文件末尾，能读出目录就说明下全了
            python -c "import zipfile, sys; zipfile.ZipFile('data/train2017.zip.part').namelist(); print('zip ok')"
            mv data/train2017.zip.part data/train2017.zip
        fi
    fi

    pip install ijson  # 流式 json 解析，见下面的注释；已装则秒过

    if [ "$STAGE2_RECIPE" = "mix665k_coco" ]; then
        python - <<'PY'
import json
import os
import shutil
import zipfile

import ijson

# 和下面 instruct150k 那段同样的理由用 ijson：这个 json 有 1.03GB，json.load() 出来的
# dict/str 对象通常是文件体积的 3~5 倍，而 AutoDL 无卡模式的 cgroup 配额只有 2GB。
# 这里连蓄水池都不需要——要的是全部 coco 条目，所以边解析边写盘，常驻内存只有那个
# 文件名集合（~11 万个字符串，十几 MB）。use_float 避免 ijson 产出 Decimal（json.dump 不认）。
PREFIX = "coco/train2017/"
src = "data/llava_v1_5_mix665k.json"
dst = "data/llava_v1_5_mix665k_coco.json"

# json 里的路径形如 "coco/train2017/000000033471.jpg"，剥掉前缀存成裸文件名，就能沿用
# 现有的扁平 data/images/ 布局和 configs 里的 IMAGE_DIR 默认值，不用动训练侧任何代码。
image_names = set()
total = kept = 0
with open(src, "rb") as f, open(dst, "w", encoding="utf-8") as out:
    out.write("[")
    for item in ijson.items(f, "item", use_float=True):
        total += 1
        image = item.get("image")
        if not image or not image.startswith(PREFIX):
            continue  # 纯文本的 ShareGPT 条目，以及 vg/gqa/ocr_vqa/textvqa 的图片
        name = image[len(PREFIX):]
        item["image"] = name
        image_names.add(name)
        if kept:
            out.write(",")
        json.dump(item, out, ensure_ascii=False)
        kept += 1
    out.write("]")

print(f"kept {kept}/{total} conversations ({len(image_names)} unique images), extracting ...")

os.makedirs("data/images", exist_ok=True)
missing = 0
with zipfile.ZipFile("data/train2017.zip") as zf:
    for name in image_names:
        dest = os.path.join("data/images", name)
        if os.path.exists(dest):
            continue  # 上一轮配方已经解压过的那几万张直接复用
        try:
            with zf.open(f"train2017/{name}") as src_fp, open(dest, "wb") as out_fp:
                shutil.copyfileobj(src_fp, out_fp)
        except KeyError:
            missing += 1
if missing:
    print(f"warning: {missing} image(s) absent from train2017.zip; VQAInstructDataset will skip those samples")
print("extraction done")
PY
        echo "数据集就绪（mix665k 的 COCO 子集），annotations: data/llava_v1_5_mix665k_coco.json  images: data/images"
    else
    # Stage-1 那段用的也叫 NUM_SAMPLES，这里单独开一个变量，两段才能在同一次运行里各取各的值。
    STAGE2_NUM_SAMPLES="${STAGE2_NUM_SAMPLES:-50000}"
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
        echo "数据集就绪（随机采样 $STAGE2_NUM_SAMPLES 条），annotations: data/llava_instruct_150k.json  images: data/images"
    fi

    rm -f data/train2017.zip
fi

echo "最终磁盘占用："
df -h "$DATA_DISK"
echo "AutoDL environment ready."
