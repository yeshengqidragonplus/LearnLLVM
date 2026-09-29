# 大模型从零预训练 · 学习与实验

> 对应「大模型训练流程学习」讨论定稿的实验方案。目标：**完整跑通从零预训练链路**，不追求产出可用模型。
> 详细方案见项目根目录《大模型预训练学习与实验架构方案.html》。

## 实验速览

| 项 | 值 |
|---|---|
| 数据集 | 200 中文 + 200 英文短句（400 条纯文本，≈几千 token） |
| 任务 | Causal Language Model（因果自回归 next-token 预测） |
| 架构 | Dense Decoder-Only Transformer（Qwen2 架构缩小版） |
| 模型配置 | hidden=256 / 3 层 / 4 头 / MHA / SwiGLU / RoPE / RMSNorm(Pre-LN) / tie_embeddings |
| 词表 | 复用现成 Qwen2 tokenizer（vocab_size = 151936） |
| 参数量 | ≈ 41M（Embedding/LM-Head 共享占大头） |
| 训练 | 全量 BF16 · AdamW · lr=3e-4 · warmup 5% · cosine · grad_clip 1.0 · batch=2 |
| 显存 | 估算 < 1GB（8G 显卡无压力，无需量化/LoRA） |
| 预期 | train_loss 下降；eval_loss 先降后升（过拟合，预期现象） |

## 环境要求

- Windows / Linux / macOS，NVIDIA 显卡（驱动支持 CUDA 12）
- Python 3.10+
- 显存 ≥ 4GB（本项目实际使用 < 1GB）

## 安装依赖（虚拟环境隔离）

```powershell
# 进入项目目录
cd "D:\Doc\Leran LLVM\pretrain_lab"

# 创建虚拟环境
python -m venv .venv

# 安装 CUDA 版 PyTorch（RTX 40 系）
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# 安装其余依赖
.\.venv\Scripts\python -m pip install transformers sentencepiece
```

> 国内下载慢可换镜像：`pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple`（PyPI 版 torch 在 Windows 自带 CUDA 运行时）。

## 运行流程（四步，对应完整链路）

```powershell
# ① 生成玩具数据集（400 条：中英对照词 + 中文数学 + 中英短句）
.\.venv\Scripts\python 01_make_dataset.py

# ② 下载复用现成 Qwen2 tokenizer（需联网，已走国内镜像）
.\.venv\Scripts\python 02_load_tokenizer.py

# ③ 从零预训练（随机初始化模型 + 全量 BF16 训练，约几分钟）
.\.venv\Scripts\python 03_train_pretrain.py

# ④ 推理验证（对比训练前后输出）
.\.venv\Scripts\python 04_infer.py          # 训练后
.\.venv\Scripts\python 04_infer.py --before # 训练前（随机初始化）
```

## 目录结构与产物

```
pretrain_lab/
├── 01_make_dataset.py      # ① 数据生成
├── 02_load_tokenizer.py    # ② tokenizer 加载（复用 Qwen2 现成）
├── 03_train_pretrain.py    # ③ 预训练主脚本
├── 04_infer.py             # ④ 推理验证
├── data/
│   ├── toy_pretrain.jsonl  # 原始数据集（纯文本，一行一条）
│   └── corpus.txt          # 语料转存
├── tokenizer/              # Qwen2 tokenizer 文件
├── output/
│   ├── checkpoints/epochXX # 每 epoch 完整权重
│   └── final_model/        # 最终完整模型（可脱离脚本直接加载）
└── logs/
    ├── train.log           # 训练日志（loss 曲线数据）
    └── pip_torch.log       # pip 安装日志
```

## 训练中要观察的现象（学习要点）

1. **train_loss 持续下降** —— 模型在拟合文本分布；
2. **eval_loss 先降后升** —— 极小数据集必然过拟合，这是**预期现象**；eval_loss 最低点对应的 checkpoint 泛化最好；
3. 训练后推理：能复现训练文本里的模式（如 `一加一等于 → 二`、`苹果 → apple`）；**没见过的句子依然乱码**，完全正常；
4. 若 loss 剧烈震荡：检查 lr / warmup / 梯度裁剪（本项目已按定稿配置好）。

## 后续路线（讨论定稿）

Dense 完整链路跑通 → 切 GQA 贴近真实模型 → MoE 对照实验（只替换 FFN）→ SFT → DPO。
