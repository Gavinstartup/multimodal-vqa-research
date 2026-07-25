"""设备探测：优先 CUDA，其次昇腾 NPU（需要 torch_npu），否则 CPU。

昇腾卡（如 910B）不走 CUDA 驱动，PyTorch 需要额外装 torch_npu 插件后才能
识别 torch.device("npu")；未安装时 torch.npu 属性本身就不存在，所以这里用
try/import 探测而不是假设它一定可用。
"""

import torch


def _npu_available():
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch.npu.is_available()


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _npu_available():
        return torch.device("npu")
    return torch.device("cpu")


def resolve_dtype(device, dtype_name):
    if device.type in ("cuda", "npu"):
        return getattr(torch, dtype_name)
    return torch.float32
