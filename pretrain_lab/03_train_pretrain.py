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
from transformers import AddedToken, Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

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
_parser.add_argument("--resize-embeddings", action="store_true",
                     help="续训时允许扩词表：旧 embedding 行原样保留，只为新增的 "
                          "Package 标记随机初始化新行（并同步 tied LM-Head）。"
                          "不加此开关时词表不一致仍直接报错。只允许扩容，不允许缩小。")
_parser.add_argument("--init-nave", default=None,
                     help="与 --init-from 配套：加载基座同目录的 nave.pt 编码器权重。"
                          "不指定则 NAVE 编码器按零初始化重新开始（会削弱基座数学能力）。")
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
_parser.add_argument("--package-code-token", action="store_true",
                     help="注册单一特殊 token <|PKG_CODE|>；仅用于代码 Package 实验，扩展词表并随模型保存")
_parser.add_argument("--package-nested-tokens", action="store_true",
                     help="注册嵌套原子包 token <|PKG_NUM|>/<|PKG_NAME|>（方案 A）；"
                          "用于 C/C++ 声明双向数据，数值与标识符各占一个专用 token")
_parser.add_argument("--target-mask", action="store_true",
                     help="只对 target 计 loss（分阶段设计的必需开关）：数据每行必须同时有 "
                          "prompt 与 target 字段，prompt 与 eos 前缀位置一律置 -100。"
                          "需配合 --no-pack；--verify-loss 会对拍掩码前后的一致性。")
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
rows = []
with open(DATA_PATH, encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))
texts = [r["text"] for r in rows]
TARGET_MASK = _args.target_mask
if TARGET_MASK and not _args.no_pack:
    raise SystemExit("[错误] --target-mask 需要配合 --no-pack：packing 模式下样本边界被切块，"
                     "无法稳定对齐 prompt/target 掩码。")
log(f"[1/7] 数据集加载完成：{len(rows)} 条样本 <- {DATA_PATH}"
    + ("（target-only loss 已启用）" if TARGET_MASK else ""))

# ================= 2. 加载 tokenizer（复用现成 Qwen2） =================
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
added_tokens = []

def _register_package_token(token_str: str) -> dict:
    """注册一个 Package 专用 token，返回审计记录；失败则退出。"""
    previous = tokenizer.encode(token_str, add_special_tokens=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [
        AddedToken(token_str, special=True, normalized=False)
    ]})
    ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(ids) != 1:
        raise SystemExit(f"[错误] {token_str} 没有成为单 token：{ids}")
    return {"token": token_str, "id": ids[0], "previous_ids": previous}

if _args.package_code_token:
    added_tokens.append(_register_package_token("<|PKG_CODE|>"))
if _args.package_nested_tokens:
    added_tokens.append(_register_package_token("<|PKG_NUM|>"))
    added_tokens.append(_register_package_token("<|PKG_NAME|>"))
# 词表大小必须覆盖特殊 token id（Qwen2 的 eos=151643 == vocab_size，直接用它做 embedding 行数会越界）
VOCAB_SIZE = max(len(tokenizer), tokenizer.eos_token_id + 1, tokenizer.pad_token_id + 1)
log(f"[2/7] Tokenizer 加载完成：vocab_size = {VOCAB_SIZE}（eos={tokenizer.eos_token_id}，新增 token={added_tokens}）")

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
    "added_tokens": added_tokens,
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
    #
    # 兼容两种目录：① 标准的 save_pretrained 产物（有 config.json）；
    # ② 只剩权重的历史目录（如 output_v5c1/final_model 只有 model.safetensors）。
    # 后者没有 config.json，必须用命令行给出的结构参数现造 config 才能加载。
    _init_dir = Path(_args.init_from)
    if (_init_dir / "config.json").exists():
        model = Qwen2ForCausalLM.from_pretrained(_args.init_from)
    else:
        _safetensors = sorted(_init_dir.glob("*.safetensors"))
        if not _safetensors:
            raise SystemExit(f"[错误] {_args.init_from} 既没有 config.json，"
                             f"也没有 .safetensors 权重，无法续训。")
        from safetensors.torch import load_file
        # 用命令行参数重建结构（因为目录里没有 config.json 可供核对）
        _rebuilt = Qwen2Config(
            vocab_size=0,               # 占位，稍后按权重实际行数回填
            hidden_size=_args.hidden,
            num_hidden_layers=_args.layers,
            num_attention_heads=_args.heads,
            num_key_value_heads=_args.heads,
            intermediate_size=int(_args.hidden * 8 / 3),
            max_position_embeddings=MAX_SEQ_LEN,
            hidden_act="silu",
            rms_norm_eps=1e-6,
            rope_theta=1000000.0,
            tie_word_embeddings=True,
            use_cache=False,
        )
        _state = load_file(str(_safetensors[0]))
        _embed_key = "model.embed_tokens.weight"
        if _embed_key not in _state:
            raise SystemExit(f"[错误] {_safetensors[0]} 中没有 {_embed_key}，"
                             f"不是本仓库 03 训练器保存的权重。")
        _rebuilt.vocab_size = int(_state[_embed_key].shape[0])
        model = Qwen2ForCausalLM(_rebuilt)
        _missing, _unexpected = model.load_state_dict(_state, strict=False)
        # 两类键缺失是正常的，不算结构错配：
        #   · nave.*         —— 编码器单独存 nave.pt，不在这份 safetensors 里；
        #   · lm_head.weight —— tie_word_embeddings=True 时与 embed_tokens 共享，
        #                       save_pretrained 不会重复保存（加载后仍指向同一张表）。
        # 其余键缺一个，就是 --hidden/--layers/--heads 写错了。
        _tied = bool(_rebuilt.tie_word_embeddings)
        _real_missing = [
            k for k in _missing
            if not k.startswith("nave.")
            and not (_tied and k == "lm_head.weight")
        ]
        if _real_missing:
            raise SystemExit(
                f"[错误] 加载 {_args.init_from} 缺少权重：{_real_missing[:5]}…"
                f"（共 {len(_real_missing)} 个）。请核对 --hidden/--layers/--heads 是否与原模型一致。")
        del _state
        log(f"       {_args.init_from} 无 config.json，已按 --hidden/--layers/--heads "
            f"重建结构并从 {_safetensors[0].name} 载入权重")
    cfg = model.config
    mismatches = []
    if cfg.hidden_size != _args.hidden:
        mismatches.append(f"hidden {cfg.hidden_size} != {_args.hidden}")
    if cfg.num_hidden_layers != _args.layers:
        mismatches.append(f"layers {cfg.num_hidden_layers} != {_args.layers}")
    if cfg.num_attention_heads != _args.heads:
        mismatches.append(f"heads {cfg.num_attention_heads} != {_args.heads}")
    need_resize = cfg.vocab_size != VOCAB_SIZE
    if need_resize and not _args.resize_embeddings:
        mismatches.append(
            f"vocab {cfg.vocab_size} != {VOCAB_SIZE}（若确实要给基座新增 Package 标记，"
            f"加 --resize-embeddings：旧行保留、只为新增标记随机初始化新行）")
    elif need_resize and cfg.vocab_size > VOCAB_SIZE:
        # 只允许扩容，不允许缩小：缩小等于丢弃已训练的词向量，一定是配置写错了。
        mismatches.append(f"vocab {cfg.vocab_size} > {VOCAB_SIZE}：不允许缩词表续训")
    if mismatches:
        raise SystemExit(f"[错误] --init-from 模型架构与参数不符：{'; '.join(mismatches)}。"
                         f"续训必须用与原模型完全相同的 --hidden/--layers/--heads。")
    if need_resize:
        old_vocab = cfg.vocab_size
        # resize_token_embeddings 会新分配整张表，因此必须先把旧表逐行拷回。
        # tie_word_embeddings=True 时 LM-Head 与 embedding 共享同一张表，resize 会同步两者。
        old_embed = model.get_input_embeddings().weight.detach().clone()
        # mean_resizing=False：关掉 HF 默认的"按旧表均值/协方差重建新行"，
        # 因为下面会把新增行改成随机初始化（不要让新行带上旧词表的统计先验）。
        model.resize_token_embeddings(VOCAB_SIZE, mean_resizing=False)
        with torch.no_grad():
            new_embed = model.get_input_embeddings().weight
            new_embed[:old_vocab].copy_(old_embed)
        del old_embed
        # 新增行必须随机初始化：若沿用零初始化，"永远输出空"会成为强先验，新标记难学会。
        with torch.no_grad():
            new_embed[old_vocab:].normal_(mean=0.0, std=cfg.initializer_range)
        model.tie_weights()
        log(f"       词表扩容 {old_vocab} -> {VOCAB_SIZE}：旧 {old_vocab} 行逐行保留，"
            f"新增 {VOCAB_SIZE - old_vocab} 行随机初始化（tied 已同步）")
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
    # 续训时默认从基座同目录的 nave.pt 继承已训练好的值编码器。
    # 不继承的话 proj 是零初始化的，基座刚学会的"数值整体表示"会被立刻清空。
    _nave_init = _args.init_nave
    if _nave_init is None and _args.init_from:
        _candidate = Path(_args.init_from) / "nave.pt"
        if _candidate.exists():
            _nave_init = str(_candidate)
    if _nave_init:
        _nave_state = torch.load(_nave_init, map_location="cpu", weights_only=True)
        NAVE.enc.load_state_dict(_nave_state)
        log(f"[3b] NAVE 编码器权重继承自 {_nave_init}（继续保留基座的数值表示）")
    # ⚠️ 必须在 model.to(DEVICE) 之后创建 → 需手动搬到同设备/同精度
    NAVE.enc = NAVE.enc.to(DTYPE).to(DEVICE)
    model.nave = NAVE.enc                      # 挂子模块 → 进 parameters/save
    _nave_params = sum(p.numel() for p in NAVE.enc.parameters())
    log(f"[3b] NAVE 已启用：digit_ids={_digit_ids}，theta={_args.nave_theta}，"
        f"编码器参数 {_nave_params/1e3:.1f}K"
        f"{'' if _nave_init else '（零初始化 proj → 起点等价于无 NAVE）'}")


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


random.shuffle(rows)
n_val = int(len(rows) * VAL_RATIO)
val_rows, train_rows = rows[:n_val], rows[n_val:]
val_texts = [r["text"] for r in val_rows]
train_texts = [r["text"] for r in train_rows]

if _args.no_pack:
    # 独立样本模式：每条 = [eos] + prompt + target + [eos]，batch 内 padding + attention mask。
    # 同时产出 labels：--target-mask 时 prompt 与 eos 前缀位置为 -100（不参与 loss），
    # 否则 labels 与 input_ids 相同（整条计 loss，保持既有语义数据的旧行为）。
    eos = tokenizer.eos_token_id

    def encode_supervised(row):
        """返回 (input_ids, labels)，两者等长；labels=-100 表示该位置不计 loss。"""
        if TARGET_MASK:
            if "prompt" not in row or "target" not in row:
                raise SystemExit(
                    "[错误] --target-mask 要求每行都有 prompt 与 target 字段；"
                    f"当前行缺少，字段为 {list(row)}")
            prompt_ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
            target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
            seq = [eos] + prompt_ids + target_ids + [eos]
            # 前 1+len(prompt) 个位置置 -100：eos 前缀与整段 prompt 都不学；
            # 最后一个 prompt 位置的 logits 仍要预测第一个 target token，故掩码正好到此为止。
            labels = [-100] * (1 + len(prompt_ids)) + target_ids + [eos]
            # 边界自检：prompt 与 target 分开编码必须等于整条 text 编码，
            # 否则 BPE 在拼接处合并了 token，监督位置会整体错位（静默算错 loss）。
            whole_ids = tokenizer.encode(row["text"], add_special_tokens=False)
            if whole_ids != prompt_ids + target_ids:
                raise SystemExit(
                    "[错误] prompt/target 的 token 边界不一致：分开编码 "
                    f"{len(prompt_ids)}+{len(target_ids)} 个 token，整条 text 却是 "
                    f"{len(whole_ids)} 个。target 开头/结尾与 prompt 拼接处被 BPE 合并，"
                    "target-only 掩码会错位。请在数据侧保证 prompt 以换行结尾"
                    "（或改用词表级边界标记）后重新生成数据。")
            return seq, labels
        ids = tokenizer.encode(row["text"])
        seq = [eos] + ids + [eos]
        return seq, seq[:]

    train_pairs = [encode_supervised(r) for r in train_rows]
    val_pairs = [encode_supervised(r) for r in val_rows]

    def collate(batch_pairs):
        max_len = max(len(seq) for seq, _ in batch_pairs)
        ids = torch.full((len(batch_pairs), max_len), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(batch_pairs), max_len), dtype=torch.long)
        labels = torch.full((len(batch_pairs), max_len), -100, dtype=torch.long)
        for i, (seq, lab) in enumerate(batch_pairs):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1
            labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)
        return ids, mask, labels

    if TARGET_MASK:
        supervised = sum(sum(1 for y in lab if y != -100) for _, lab in train_pairs)
        total = sum(len(seq) for seq, _ in train_pairs)
        log(f"        target-only 掩码：监督 token {supervised}/{total} = "
            f"{100.0 * supervised / max(1, total):.1f}%（prompt 与 eos 前缀不计 loss）")

    if _args.bucket:
        # 按长度分桶：长度相近的凑一个 batch，短样本不被长 CoT 拖累。
        # v8.2 数据长度 8~90 token 混杂，不分桶时 padding 浪费约 3 倍算力。
        train_sorted = sorted(range(len(train_pairs)), key=lambda i: len(train_pairs[i][0]))
        val_sorted = sorted(range(len(val_pairs)), key=lambda i: len(val_pairs[i][0]))

        # 用生成器包装，保持与 DataLoader 相同的迭代接口
        class _BucketLoader:
            def __init__(self, idxs, pairs, bs):
                self.idxs, self.pairs, self.bs = list(idxs), pairs, bs

            def __iter__(self):
                idxs = self.idxs[:]
                random.shuffle(idxs)
                batches = [idxs[i:i + self.bs] for i in range(0, len(idxs), self.bs)]
                random.shuffle(batches)
                for b in batches:
                    yield collate([self.pairs[i] for i in b])

            def __len__(self):
                return (len(self.idxs) + self.bs - 1) // self.bs

        train_loader = _BucketLoader(train_sorted, train_pairs, BATCH_SIZE)
        val_loader = _BucketLoader(val_sorted, val_pairs, BATCH_SIZE)
        log(f"[4/7] 独立样本模式（长度分桶）：train {len(train_rows)} / val {len(val_rows)} 条，"
            f"batch={BATCH_SIZE}")
    else:
        train_loader = DataLoader(train_pairs, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_pairs, batch_size=BATCH_SIZE, collate_fn=collate)
    log(f"[4/7] 独立样本模式：train {len(train_rows)} 条 / val {len(val_rows)} 条"
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
        ids, mask, labels = batch
        input_ids = ids.to(DEVICE)
        attn = mask.to(DEVICE)
        labels = labels.to(DEVICE)      # padding 与 prompt 位置已是 -100
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
            ids, mask, labels = batch
            input_ids = ids.to(DEVICE)
            labels = labels.to(DEVICE)        # padding / prompt 位置已置 -100
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
        p_ids, p_mask, p_lab = _probe
        p_in = p_ids.to(DEVICE)
        p_lab = p_lab.to(DEVICE)
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
                ids, mask, labels = batch
                input_ids = ids.to(DEVICE)
                labels = labels.to(DEVICE)    # padding / prompt 位置已置 -100
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
run_record["trained_tokenizer_sha256"] = {
    p.name: sha256_file(p) for p in Path(FINAL_DIR).iterdir()
    if p.is_file() and ("tokenizer" in p.name or p.name == "special_tokens_map.json")
}
# NAVE 编码器单独存：from_pretrained 不会加载非 config 声明的子模块（nave.*），
# 评估时必须手动 load 这个文件（见 04e_eval_v5.py）
if NAVE is not None:
    torch.save(NAVE.enc.state_dict(), f"{FINAL_DIR}/nave.pt")
    log(f"NAVE 编码器已保存到 {FINAL_DIR}/nave.pt")
log(f"最终模型已保存到 {FINAL_DIR}/（完整权重 + config + tokenizer）")
run_record["status"] = "completed"
run_record["finished_utc"] = datetime.now(timezone.utc).isoformat()
save_run_record()
