# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## 概览
从零预训练大模型的学习实验。全部代码在 `pretrain_lab/`（Python 3.10+ / PyTorch CUDA / transformers Qwen2 架构缩小版）；无构建系统、无测试框架、无 lint——"跑单个测试" = 跑单个脚本。根目录《大模型预训练学习与实验架构方案.html》是权威实验记录（13.x 章含各版本结果与根因分析）；`pretrain_lab/README.md` 已过时（只描述最早的 toy 实验与已不存在的 `output/`）。

## 命令（必须在 pretrain_lab/ 内运行，脚本内 data/ tokenizer/ logs/ 均为相对路径）
```powershell
.\.venv\Scripts\python 01j_make_math_v72.py                 # 生成数据 -> data/math_v72.jsonl
.\.venv\Scripts\python 03_train_pretrain.py --data data/math_v72.jsonl --out output_math_v72 --no-pack
.\.venv\Scripts\python 04h_eval_math_v7.py --model output_math_v72/final_model   # 系统性评估（准确率）
.\.venv\Scripts\python 05_interact.py --model output_math_v72/final_model        # 交互式续写
```

## 关键约定（非显而易见）
- 版本迭代**新建文件而非改旧文件**：01h(v7)→01i(v7.1)→01j(v7.2)，旧脚本保留作对照；`_*` 前缀脚本维护根 HTML 文档（硬编码绝对路径 `D:\Doc\Leran LLVM\`），`check_*_tmp.py` 是一次性调试脚本。
- **BPE 陷阱**：Qwen2 tokenizer 把 `=-` 合并成单 token（v7.1 失败根因），负结果必须写 `= -X`（等号后空格）；负操作数必须括号 `(-3)+2`；比较句 `a小于b。` 必须无空格。评估 prompt 必须与训练格式逐字一致——`op_num()`/`eq_res()` 在数据生成与评估脚本中**重复定义**，改格式须同步所有副本。
- 训练样本包裹为 `[eos]+tokens+[eos]`；v4+ 数学模型推理输入必须手动加 eos 前缀（`[eos]+encode(prompt)`），否则输入分布与训练不一致。
- `VOCAB_SIZE` 必须取 `max(len(tokenizer), eos_id+1, pad_id+1)`——Qwen2 的 eos=151643 等于 vocab_size，直接用会 embedding 越界。
- 数学实验必须 `--no-pack`（512 切块会切断算式）；该模式自动改 batch=64 并启用 padding + attention mask（labels 置 -100）。
- 每个脚本开头 `sys.stdout.reconfigure(encoding="utf-8")` 防 Windows 控制台中文乱码；`02_load_tokenizer.py` 须在 import transformers **之前**设 `HF_ENDPOINT=https://hf-mirror.com`。
- 代码风格：文件头 `# -*- coding: utf-8 -*-` + docstring 写明文件名与目的；中文注释；argparse 参数用中文 help；超参常量全大写置顶；模型统一 `.to(torch.bfloat16).to(device)`。
