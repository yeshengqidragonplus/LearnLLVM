# -*- coding: utf-8 -*-
"""setup_env.py：跨平台建立实验环境，PyTorch 后端与通用依赖分开安装。"""
import argparse
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="创建实验环境（建议 Python 3.11）")
    parser.add_argument("--backend", choices=["auto", "cuda", "cpu"], default="auto", help="auto 根据 nvidia-smi 探测 NVIDIA")
    parser.add_argument("--cuda", default="cu126", help="官方 PyTorch CUDA wheel 通道，如 cu126；不是本机 Toolkit 版本")
    parser.add_argument("--torch-version", default="2.14.0", help="PyTorch 版本，默认与本项目已验证版本一致")
    parser.add_argument("--venv", default=".venv", help="虚拟环境路径，相对实验室目录")
    parser.add_argument("--dry-run", action="store_true", help="只打印安装计划，不创建或更改环境")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("至少需要 Python 3.10，建议使用 3.11")
    if not re.fullmatch(r"cu\d+", args.cuda):
        parser.error("CUDA 通道必须形如 cu126")
    backend = args.backend
    if backend == "auto":
        has_gpu = False
        if platform.system() != "Darwin" and shutil.which("nvidia-smi"):
            try:
                check = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                       capture_output=True, text=True, timeout=10)
                has_gpu = check.returncode == 0 and bool(check.stdout.strip())
            except (OSError, subprocess.TimeoutExpired):
                pass
        backend = "cuda" if has_gpu else "cpu"
    if backend == "cuda" and platform.system() == "Darwin":
        parser.error("macOS 不使用 CUDA；本项目当前支持 CPU 路径")
    env = (ROOT / args.venv).resolve()
    python = env / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    commands = []
    if not python.exists():
        commands.append([sys.executable, "-m", "venv", str(env)])
    pip = [str(python), "-m", "pip", "--isolated", "install"]
    torch_cmd = pip + [f"torch=={args.torch_version}"]
    if platform.system() != "Darwin":
        channel = args.cuda if backend == "cuda" else "cpu"
        torch_cmd += ["--index-url", f"https://download.pytorch.org/whl/{channel}"]
    commands += [torch_cmd, pip + ["-r", str(ROOT / "requirements.txt")],
                 [str(python), str(ROOT / "check_env.py"), "--device", backend]]
    print(f"安装目标：{backend} / {env}；macOS GPU、AMD/Intel GPU 尚未验证。", flush=True)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    # 不把已有 CUDA wheel 静默改成 CPU，反之亦然；可指定另一个虚拟环境。
    if python.exists():
        probe = subprocess.run([str(python), "-c", "import torch,json; print(json.dumps(torch.version.cuda))"],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            installed_cuda = json.loads(probe.stdout.strip())
            expected_cuda = args.cuda[2:]
            installed_channel = installed_cuda.replace(".", "") if installed_cuda else None
            if (backend == "cpu" and installed_cuda) or (backend == "cuda" and installed_channel != expected_cuda):
                raise SystemExit("已有环境的 PyTorch 后端不同，请用 --venv 指定新目录；不会覆盖现有环境。")
    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)
    print("环境安装和前后向验证完成。", flush=True)


if __name__ == "__main__":
    main()
