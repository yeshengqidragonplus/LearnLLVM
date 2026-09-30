# -*- coding: utf-8 -*-
"""
03_train_pretrain.py
从零预训练主脚本：随机初始化一个 Dense Decoder-Only Transformer（Qwen2 架构小模型），
使用 JSONL 数学数据做因果自回归预训练，按设备选择 BF16 或 FP32。

流程对应（讨论纪要第 6 章）：
  1) 加载数据集（data/math_v5.jsonl）
  2) 加载 tokenizer（tokenizer/，复用现成 Qwen2）
  3) 随机初始化模型（不加载任何预训练权重！）
  4) 数据预处理：tokenize + packing 成 512 块，train/val 划分
  5) 训练循环：labels=input_ids，全部位置算 loss（无 mask）
  6) 超参：AdamW / lr=3e-4 / warmup 5% / cosine / grad_clip 1.0 / batch=2
  7) 每 epoch 保存 checkpoint，最终保存完整模型
"""
import json
import math
import os
import random
import sys
import time
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from runtime import add_runtime_args, resolve_runtime
from runtime import environment_info
from torch.utils.data import DataLoader, TensorDataset
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

# Windows 控制台输出 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ================= 0. 超参（讨论定稿） =================
SEED = 42
BATCH_SIZE = 2
MAX_SEQ_LEN = 512
NUM_EPOCHS = 30
LR = 3e-4
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0
VAL_RATIO = 0.2


# 命令行参数：数学实验完整参数见 README；默认架构仍是小型通用配置。
import argparse
_parser = argparse.ArgumentParser()
_parser.add_argument("--data", default="data/math_v5.jsonl", help="训练数据 jsonl 路径")
_parser.add_argument("--out", default="output", help="输出目录（checkpoints/ 与 final_model/ 的子目录）")
_parser.add_argument("--hidden", type=int, default=256, help="hidden_size")
_parser.add_argument("--layers", type=int, default=3, help="num_hidden_layers")
_parser.add_argument("--heads", type=int, default=4, help="num_attention_heads / kv heads(MHA)")
_parser.add_argument("--no-pack", action="store_true",
                     help="每条样本独立训练（不 packing，杜绝 512 切块把样本切断；batch 内 padding + mask）")
_parser.add_argument("--batch", type=int, default=None,
                     help="no-pack 模式的 batch size（默认 64；样本很长时 OOM 可降，"
                          "8GB 显卡建议显式指定 8 或 16，并按真实样本长度调整）")
_parser.add_argument("--bucket", action="store_true",
                     help="no-pack 模式按长度分桶（长度相近的样本凑一个 batch，"
                          "减少 padding 浪费；CoT 长短混合数据集提速明显）")
_parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
_parser.add_argument("--seed", type=int, default=42, help="数据划分与模型初始化随机种子")
_parser.add_argument("--no-checkpoints", action="store_true",
                     help="不保存每 epoch checkpoint（只存 final_model，省磁盘；每 epoch 约 130MB）")
_parser.add_argument("--init-from", default=None,
                     help="从已有模型权重续训（课程学习/回放场景）。"
                          "架构参数必须与该模型一致（hidden/layers/heads），否则报错。"
                          "不指定则随机初始化（从零预训练）。")
_parser.add_argument("--lr", type=float, default=None,
                     help="学习率（默认 3e-4；续训建议 5e-5~1e-4，避免冲毁已有能力）")
_parser.add_argument("--loss-chunk", type=int, default=0,
                     help="沿序列分块算 fp32 loss 的块大小（0=HF 原生；"
                          "64 可把 (B,L,151646)fp32 峰值显存降约 5 倍，解决长样本 OOM，"
                          "数值与 HF mean 等价）")
_parser.add_argument("--verify-loss", action="store_true",
                     help="训练前对拍：分块 loss 与 HF 原生 loss 必须几乎一致，否则拒绝训练")
_parser.add_argument("--nave", action="store_true",
                     help="启用 NAVE 数字值嵌入（根目录实验总册中的 NAVE v5 设计）："
                          "给每个数字 token 注入因果安全的'前缀值'表示，"
                          "让数在潜空间成为整体（绕开复制通道定宽墙）。默认关。")
_parser.add_argument("--nave-theta", type=int, default=8,
                     help="NAVE 连续位权编码的角度数（2*n 维输出），默认 8")
add_runtime_args(_parser)
_args = _parser.parse_args()
DEVICE, DTYPE = resolve_runtime(_args.device, _args.dtype)
SEED = _args.seed

if _args.no_pack:
    BATCH_SIZE = _args.batch if _args.batch else 64   # 独立短样本，batch 加大提速

NUM_EPOCHS = _args.epochs

DATA_PATH = _args.data
TOKENIZER_DIR = "tokenizer"
CKPT_DIR = f"{_args.out}/checkpoints"
FINAL_DIR = f"{_args.out}/final_model"
LOG_PATH = f"{_args.out}/train.log"
RUN_PATH = Path(_args.out) / "run.json"
if RUN_PATH.exists() or Path(FINAL_DIR).exists():
    raise SystemExit("输出目录已有实验记录或模型，请换一个 --out，避免覆盖另一次实验。")

random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)


def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ================= 1. 加载数据集 =================
texts = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
log(f"[1/7] 数据集加载完成：{len(texts)} 条样本 <- {DATA_PATH}")

# ================= 2. 加载 tokenizer（复用现成 Qwen2） =================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
# 词表大小必须覆盖特殊 token id（Qwen2 的 eos=151643 == vocab_size，直接用它做 embedding 行数会越界）
VOCAB_SIZE = max(len(tokenizer), tokenizer.eos_token_id + 1, tokenizer.pad_token_id + 1)
log(f"[2/7] Tokenizer 加载完成：vocab_size = {VOCAB_SIZE}（eos={tokenizer.eos_token_id}）")

# 每次实验自带可公开分享的元数据；不保存本机用户名、主机名或环境变量。
def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


try:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True, timeout=5).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True, timeout=5).strip())
except (OSError, subprocess.SubprocessError):
    revision, dirty = None, None
run_record = {
    "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
    "environment": environment_info(DEVICE, DTYPE), "arguments": vars(_args),
    "git_commit": revision, "git_dirty": dirty,
    "script_sha256": {p.name: sha256_file(p) for p in Path(__file__).parent.glob("*.py")},
    "data": {"samples": len(texts), "sha256": sha256_file(DATA_PATH)},
    "tokenizer_sha256": {p.name: sha256_file(p) for p in Path(TOKENIZER_DIR).iterdir() if p.is_file()},
    "effective_batch_size": BATCH_SIZE, "epochs": [],
}


def save_run_record():
    temporary = RUN_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(run_record, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(RUN_PATH)


save_run_record()

# ================= 3. 初始化模型（随机 或 从已有权重续训） =================
config = Qwen2Config(
    vocab_size=VOCAB_SIZE,          # 跟随现成 tokenizer 的词表
    hidden_size=_args.hidden,
    num_hidden_layers=_args.layers,
    num_attention_heads=_args.heads,
    num_key_value_heads=_args.heads,   # MHA（=heads），后续可切 GQA
    intermediate_size=int(_args.hidden * 8 / 3),   # SwiGLU 惯例
    max_position_embeddings=MAX_SEQ_LEN,
    hidden_act="silu",              # SwiGLU
    rms_norm_eps=1e-6,
    rope_theta=1000000.0,
    tie_word_embeddings=True,       # Embedding 与 LM-Head 权重共享
    use_cache=False,                # 训练时不缓存 KV
)
if _args.init_from:
    # 课程学习/续训：加载已有权重。架构一致性必须校验——
    # hidden/layers/heads/vocab 任一不符就拒绝训练（防止静默错配烧卡）。
    model = Qwen2ForCausalLM.from_pretrained(_args.init_from)
    cfg = model.config
    mismatches = []
    if cfg.hidden_size != _args.hidden:
        mismatches.append(f"hidden {cfg.hidden_size} != {_args.hidden}")
    if cfg.num_hidden_layers != _args.layers:
        mismatches.append(f"layers {cfg.num_hidden_layers} != {_args.layers}")
    if cfg.num_attention_heads != _args.heads:
        mismatches.append(f"heads {cfg.num_attention_heads} != {_args.heads}")
    if cfg.vocab_size != VOCAB_SIZE:
        mismatches.append(f"vocab {cfg.vocab_size} != {VOCAB_SIZE}")
    if mismatches:
        raise SystemExit(f"[错误] --init-from 模型架构与参数不符：{'; '.join(mismatches)}。"
                         f"续训必须用与原模型完全相同的 --hidden/--layers/--heads。")
    model = model.to(DTYPE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[3/7] 从 {_args.init_from} 加载权重续训：总参数量 = {n_params/1e6:.2f}M，"
        f"设备 = {DEVICE}，精度 = {DTYPE}（课程学习模式）")
else:
    model = Qwen2ForCausalLM(config)    # 只有结构配置，权重全部随机初始化！
    model = model.to(DTYPE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[3/7] 模型随机初始化完成：总参数量 = {n_params/1e6:.2f}M，设备 = {DEVICE}，精度 = {DTYPE}")

# ================= 3b. NAVE 数字值嵌入（--nave，根目录实验总册中的 NAVE v5 设计） =================
# 关键：把编码器挂为 model 的**子模块**（model.nave），使：
#   ① 进入 model.parameters() → 优化器自动覆盖
#   ② save_pretrained/from_pretrained 自动存取（state_dict 含子模块）
#   ③ 关闭 --nave 时完全不创建 → 与 v4-A 基线严格一致（可复现对照）
NAVE = None
if _args.nave:
    from nave import NaveWrapper
    # 查 '0'..'9' 的 token id（必须按 0..9 顺序！nave.prefix_values 用下标当数字值）
    _digit_ids = []
    for _d in "0123456789":
        _ids = tokenizer.encode(_d, add_special_tokens=False)
        if len(_ids) != 1:
            raise SystemExit(f"[错误] 数字 {_d!r} 不是单 token（{_ids}）——NAVE 要求逐位单 token")
        _digit_ids.append(_ids[0])
    NAVE = NaveWrapper(model, _args.hidden, _digit_ids, n_theta=_args.nave_theta)
    # ⚠️ 必须在 model.to(DEVICE) 之后创建 → 需手动搬到同设备/同精度
    NAVE.enc = NAVE.enc.to(DTYPE).to(DEVICE)
    model.nave = NAVE.enc                      # 挂子模块 → 进 parameters/save
    _nave_params = sum(p.numel() for p in NAVE.enc.parameters())
    log(f"[3b] NAVE 已启用：digit_ids={_digit_ids}，theta={_args.nave_theta}，"
        f"编码器参数 {_nave_params/1e3:.1f}K（零初始化 proj → 起点等价于无 NAVE）")


def embed_inputs(input_ids):
    """统一嵌入入口：有 NAVE 则 基础嵌入 + 值嵌入；否则走原路径"""
    if NAVE is None:
        return model.get_input_embeddings()(input_ids)
    return NAVE.embed(model, input_ids)

# 学习率：续训默认小 lr（5e-5，防冲毁已有能力）；从零训练保持 3e-4
if _args.lr is not None:
    LR = _args.lr
elif _args.init_from:
    LR = 5e-5
log(f"    学习率 = {LR:g}" + ("（续训小 lr）" if _args.init_from else ""))


# ================= 4. 数据预处理：tokenize + packing =================
def build_blocks(sample_texts, max_len):
    """把多条文本拼接成连续 token 流，切成 max_len 的块。

    v2 管线修复：每条样本【前后】都插 <eos>，保证"eos 开头 + 算式"成为训练状态，
    与测试时"孤立前缀 + eos 前缀"的输入分布一致（否则模型只学会完整式子，
    孤立前缀测试时会退化按高频 token 瞎猜）。
    """
    stream = []
    eos = tokenizer.eos_token_id
    for t in sample_texts:
        stream += [eos] + tokenizer.encode(t) + [eos]
    blocks = []
    for i in range(0, len(stream) - max_len + 1, max_len):
        blocks.append(stream[i:i + max_len])
    return blocks


random.shuffle(texts)
n_val = int(len(texts) * VAL_RATIO)
val_texts, train_texts = texts[:n_val], texts[n_val:]

if _args.no_pack:
    # 独立样本模式：每条 = [eos] + tokens + [eos]，batch 内 padding + attention mask
    eos = tokenizer.eos_token_id
    train_seqs = [[eos] + tokenizer.encode(t) + [eos] for t in train_texts]
    val_seqs = [[eos] + tokenizer.encode(t) + [eos] for t in val_texts]

    def collate(batch_seqs):
        max_len = max(len(s) for s in batch_seqs)
        ids = torch.full((len(batch_seqs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch_seqs), max_len), dtype=torch.long)
        for i, s in enumerate(batch_seqs):
            ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, :len(s)] = 1
        return ids, mask

    if _args.bucket:
        # 按长度分桶：长度相近的凑一个 batch，短样本不被长 CoT 拖累。
        # v8.2 数据长度 8~90 token 混杂，不分桶时 padding 浪费约 3 倍算力。
        train_sorted = sorted(range(len(train_seqs)), key=lambda i: len(train_seqs[i]))
        val_sorted = sorted(range(len(val_seqs)), key=lambda i: len(val_seqs[i]))

        # 用生成器包装，保持与 DataLoader 相同的迭代接口
        class _BucketLoader:
            def __init__(self, idxs, seqs, bs):
                self.idxs, self.seqs, self.bs = list(idxs), seqs, bs

            def __iter__(self):
                idxs = self.idxs[:]
                random.shuffle(idxs)
                batches = [idxs[i:i + self.bs] for i in range(0, len(idxs), self.bs)]
                random.shuffle(batches)
                for b in batches:
                    yield collate([self.seqs[i] for i in b])

            def __len__(self):
                return (len(self.idxs) + self.bs - 1) // self.bs

        train_loader = _BucketLoader(train_sorted, train_seqs, BATCH_SIZE)
        val_loader = _BucketLoader(val_sorted, val_seqs, BATCH_SIZE)
        log(f"[4/7] 独立样本模式（长度分桶）：train {len(train_texts)} / val {len(val_texts)} 条，"
            f"batch={BATCH_SIZE}")
    else:
        train_loader = DataLoader(train_seqs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_seqs, batch_size=BATCH_SIZE, collate_fn=collate)
        log(f"[4/7] 独立样本模式：train {len(train_texts)} 条 / val {len(val_texts)} 条"
            f"（batch 内 pad，样本内转移全部完整，无切块破坏）")
    log(f"[4/7] 独立样本模式：train {len(train_texts)} 条 / val {len(val_texts)} 条"
        f"（batch 内 pad，样本内转移全部完整，无切块破坏）")
else:
    train_blocks = build_blocks(train_texts, MAX_SEQ_LEN)
    val_blocks = build_blocks(val_texts, MAX_SEQ_LEN)
    log(f"[4/7] 预处理完成：train {len(train_texts)} 条 -> {len(train_blocks)} 个块；"
        f"val {len(val_texts)} 条 -> {len(val_blocks)} 个块（packing，max_len={MAX_SEQ_LEN}）")
    train_ds = TensorDataset(torch.tensor(train_blocks, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_blocks, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ================= 5. 优化器 + 学习率调度（warmup + cosine） =================
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = max(1, int(total_steps * WARMUP_RATIO))


def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * p))


scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
log(f"[5/7] 优化器 AdamW(lr={LR})，总步数 ≈ {total_steps}，warmup {warmup_steps} 步，cosine 退火")


# ================= 6. 训练循环 =================
# ---------- 分块 fp32 loss（大词表长样本 OOM 解法）----------
# 背景：HF 的 CausalLM 内部把 logits 升 fp32 算 cross_entropy。
#   词表 151646 × 序列 305 × batch 16 → logits fp32 峰值 2.96GB（8G 卡 OOM）。
# 原理：交叉熵 = 逐 token 求和，可分块累加，数学上完全等价（非近似！）。
#   峰值从 (B, L, V) 降到 (B, chunk, V)，chunk=64 时省约 5 倍。
# 关键：loss 归约方式必须与 HF 一致——HF 用 mean over 非 -100 token，
#   故分块先累计 sum 与有效 token 数，最后相除。
import torch.nn.functional as F


def chunked_loss(batch, chunk):
    """返回分块 fp32 loss（与 HF mean 等价）"""
    if _args.no_pack:
        ids, mask = batch
        input_ids = ids.to(DEVICE)
        attn = mask.to(DEVICE)
        labels = input_ids.clone()
        labels[mask == 0] = -100
    else:
        input_ids = batch[0].to(DEVICE)
        attn = None
        labels = input_ids
    # 前向只取 hidden（不建 logits）
    # NAVE：把 input_ids 换成 inputs_embeds（含值嵌入）
    if NAVE is not None:
        embeds = embed_inputs(input_ids)
        hidden = model.model(inputs_embeds=embeds,
                             attention_mask=attn).last_hidden_state
    else:
        hidden = model.model(input_ids=input_ids,
                             attention_mask=attn).last_hidden_state  # (B, L, H)
    # shift：预测下一个 token
    shift_hidden = hidden[:, :-1, :]
    shift_labels = labels[:, 1:]
    L = shift_hidden.shape[1]
    total_sum = 0.0
    total_cnt = 0
    for i in range(0, L, chunk):
        h_c = shift_hidden[:, i:i + chunk, :]
        y_c = shift_labels[:, i:i + chunk]
        logits_c = model.lm_head(h_c).float()          # 仅本块升 fp32
        s = F.cross_entropy(logits_c.reshape(-1, logits_c.shape[-1]),
                            y_c.reshape(-1), reduction="sum", ignore_index=-100)
        total_sum = total_sum + s
        total_cnt += (y_c != -100).sum()
    return total_sum / total_cnt.clamp(min=1)


@torch.no_grad()
def evaluate(loader):
    model.eval()
    losses = []
    for batch in loader:
        if _args.loss_chunk:
            losses.append(float(chunked_loss(batch, _args.loss_chunk)))
            continue
        if _args.no_pack:
            ids, mask = batch
            input_ids = ids.to(DEVICE)
            labels = input_ids.clone()
            labels[mask == 0] = -100          # padding 位置不参与 loss
            if NAVE is not None:
                out = model(inputs_embeds=embed_inputs(input_ids),
                            attention_mask=mask.to(DEVICE), labels=labels)
            else:
                out = model(input_ids=input_ids, attention_mask=mask.to(DEVICE), labels=labels)
        else:
            input_ids = batch[0].to(DEVICE)
            if NAVE is not None:
                out = model(inputs_embeds=embed_inputs(input_ids), labels=input_ids)
            else:
                out = model(input_ids=input_ids, labels=input_ids)   # 预训练：labels = input_ids
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / len(losses)


# ---------- 训练前对拍（--verify-loss）：分块 loss 必须与 HF 原生几乎一致 ----------
if _args.verify_loss:
    if not _args.loss_chunk:
        raise SystemExit("[错误] --verify-loss 需配合 --loss-chunk 使用")
    log("[对拍] 分块 loss vs HF 原生 loss（必须近似一致，否则拒绝训练）")
    _probe = next(iter(train_loader))
    # HF 原生
    if _args.no_pack:
        p_ids, p_mask = _probe
        p_in = p_ids.to(DEVICE)
        p_lab = p_in.clone()
        p_lab[p_mask == 0] = -100
        with torch.no_grad():
            if NAVE is not None:
                ref = model(inputs_embeds=embed_inputs(p_in),
                            attention_mask=p_mask.to(DEVICE), labels=p_lab).loss.item()
            else:
                ref = model(input_ids=p_in, attention_mask=p_mask.to(DEVICE), labels=p_lab).loss.item()
    else:
        p_in = _probe[0].to(DEVICE)
        with torch.no_grad():
            if NAVE is not None:
                ref = model(inputs_embeds=embed_inputs(p_in), labels=p_in).loss.item()
            else:
                ref = model(input_ids=p_in, labels=p_in).loss.item()
    with torch.no_grad():
        got = float(chunked_loss(_probe, _args.loss_chunk))
    diff = abs(ref - got)
    rel = diff / (abs(ref) + 1e-12)
    log(f"       HF 原生 = {ref:.6f} | 分块 = {got:.6f} | 绝对差 = {diff:.6f} | 相对差 = {rel:.4%}")
    if rel > 1e-3:
        raise SystemExit(f"[错误] 分块 loss 与原生偏差过大（{rel:.4%} > 0.1%），"
                         f"归约方式可能不一致，拒绝训练")
    log("       对拍通过（相对差 < 0.1%）")

log("[6/7] 开始训练 ...")
t0 = time.time()
step = 0
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        if _args.loss_chunk:
            loss = chunked_loss(batch, _args.loss_chunk)      # 分块 fp32（省显存）
        else:
            if _args.no_pack:
                ids, mask = batch
                input_ids = ids.to(DEVICE)
                labels = input_ids.clone()
                labels[mask == 0] = -100
                if NAVE is not None:
                    out = model(inputs_embeds=embed_inputs(input_ids),
                                attention_mask=mask.to(DEVICE), labels=labels)
                else:
                    out = model(input_ids=input_ids, attention_mask=mask.to(DEVICE), labels=labels)
            else:
                input_ids = batch[0].to(DEVICE)
                if NAVE is not None:
                    out = model(inputs_embeds=embed_inputs(input_ids), labels=input_ids)
                else:
                    out = model(input_ids=input_ids, labels=input_ids)   # 全部 token 参与 loss，无 mask
            loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        epoch_loss += loss.item()
        step += 1

    train_loss = epoch_loss / len(train_loader)
    eval_loss = evaluate(val_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    elapsed = time.time() - t0
    log(f"  epoch {epoch:02d}/{NUM_EPOCHS} | train_loss {train_loss:.4f} | "
        f"eval_loss {eval_loss:.4f} | lr {cur_lr:.2e} | 已用 {elapsed:.0f}s")
    run_record["epochs"].append({"epoch": epoch, "train_loss": train_loss,
                                "eval_loss": eval_loss, "lr": cur_lr, "elapsed_seconds": elapsed})
    save_run_record()

    # 每 epoch 保存一个 checkpoint（完整权重，不是 LoRA）；--no-checkpoints 时跳过（省磁盘）
    if not _args.no_checkpoints:
        model.save_pretrained(f"{CKPT_DIR}/epoch{epoch:02d}")

log(f"[7/7] 训练结束，总耗时 {time.time() - t0:.0f}s")

# ================= 7. 保存最终完整模型 =================
model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
# NAVE 编码器单独存：from_pretrained 不会加载非 config 声明的子模块（nave.*），
# 评估时必须手动 load 这个文件（见 04e_eval_v5.py）
if NAVE is not None:
    torch.save(NAVE.enc.state_dict(), f"{FINAL_DIR}/nave.pt")
    log(f"NAVE 编码器已保存到 {FINAL_DIR}/nave.pt")
log(f"最终模型已保存到 {FINAL_DIR}/（完整权重 + config + tokenizer）")
run_record["status"] = "completed"
run_record["finished_utc"] = datetime.now(timezone.utc).isoformat()
save_run_record()
