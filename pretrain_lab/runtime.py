# -*- coding: utf-8 -*-
"""runtime.py：训练、评估和环境检查共用的设备与精度选择。"""
import platform
import torch
import transformers


def add_runtime_args(parser):
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto 优先 CUDA，否则 CPU；macOS 暂用 CPU")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float32"], default="auto",
                        help="auto 在支持 BF16 的 CUDA 上用 BF16，其余用 FP32")


def resolve_runtime(device="auto", dtype="auto"):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA，但当前 PyTorch/驱动无法使用 GPU；运行 check_env.py 或显式 --device cpu")
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    if dtype == "auto":
        dtype = "bfloat16" if bf16 else "float32"
    if dtype == "bfloat16" and not bf16:
        raise RuntimeError("本实验室仅在支持 BF16 的 CUDA 上启用 BF16；请改用 --dtype float32")
    return device, getattr(torch, dtype)


def environment_info(device, dtype):
    info = {"python": platform.python_version(), "os": platform.system(),
            "architecture": platform.machine(), "torch": torch.__version__,
            "transformers": transformers.__version__, "cuda_runtime": torch.version.cuda,
            "device": device, "dtype": str(dtype).removeprefix("torch.")}
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, vram_mib=props.total_memory // 1024**2)
    return info
