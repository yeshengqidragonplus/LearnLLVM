# Git 提交策略（本仓库，2026-09-29 全量提交时确立）

远端 `git@github.com:yeshengqidragonplus/LearnLLVM.git`，分支 `master`。
根 `.gitignore` 已配置排除项，**提交前不需要再手工筛选**，直接 `git add -A` 即可。

## 排除项（不入库）
- `pretrain_lab/.venv/`（约 4GB，含 torch/cuda 二进制）
- 模型权重：`*.safetensors` / `*.pt` / `*.pth` / `*.bin` / `*.ckpt` / `*.onnx` / `*.gguf`
  —— 每个 `output_*/final_model/model.safetensors` 约 135MB，**超 GitHub 单文件 100MB 硬限制**，
  push 必被拒。权重可由 `03_train_pretrain.py` 重训复现，故不入库、也不用 git-lfs。
- `pretrain_lab/output_*/checkpoints/`（optimizer state）
- `__pycache__/`、`.roo/command-trust.json`（Roo 本地命令信任状态，含个人命令历史）

## 保留入库
代码/文档/HTML、`data/*.jsonl`（最大 `math_v5.jsonl` 14MB）、`logs/*.log`、
`tokenizer/`、以及各 `output_*/final_model/` 下的**小配置文件**
（`config.json` / `generation_config.json` / `tokenizer*.json` / `chat_template.jinja`）——
它们记录各版本实验的模型结构与超参，是实验记录的一部分。

## Windows/cmd 操作坑（实测）
- `powershell -Command "..."` 经 cmd 传参时引号会被破坏（`Sort-Object` 报"不是内部或外部命令"）。
  复杂 PowerShell 逻辑要**写成 .ps1 文件**再 `powershell -File xxx.ps1` 执行。
- cmd 无 `awk`/`wc`/`head`；统计行数用 `find /c /v ""`，过滤用 `findstr`。
- 中文提交信息用 `git commit -F 文件` 而非 `-m`，避免 cmd 编码乱码。
- 核对暂存体积：`git ls-files -s` 取 blob hash → `git cat-file --batch-check='%(objectsize)'`。
