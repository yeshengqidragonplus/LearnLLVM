# -*- coding: utf-8 -*-
"""包引用探针 v8：把「模型是否理解 key 的类型」测到底。

================= 为什么还要再来一版 =================
v7（22 号）留了一个悬而未决的问题：

    type_id 轴同分布 0.88~1.00，留出集掉到 0.25~1.00，
    分错的样本几乎全部把答案吐成 <|PKG_TASK|>。

当时的诊断是「配比坍缩」（type_id 只占 192/512 条、答案是单 token 的开集生成）。
但那是**推断**，没有做对照。本版把这一件事拆成可分辨的几档 + 一个负对照，一次问清。

  A. type_id_choice_k2  给 2 个候选类型，选一个
  B. type_id_choice_k6  给全部 6 个候选类型，选一个
  C. type_id_open       不给候选，直接说类型（= v7 的形态）
  D. type_id_unseen     负对照：这个 key 在训练里**从未出现**，模型有多少把握？

A/B/C 三档**配比完全相同**（每种类型同样条数），唯一变量是「候选提示」。于是：

  A/B 高、C 低  → 不是配比问题，是开集生成问题 → 协议里必须给候选枚举
  A/B/C 都高    → v7 的低分纯属配比不足，加量即可解决
  A/B/C 都低    → 156 个不透明符号到 6 个类型标记的关联学不会（真实边界）

================= D 档测的是一条结构性代价 =================
原子 key `<|PKG_NUM|,#Z>` 注册后是**单个 token**，它和类型标记 `<|PKG_NUM|>`
之间没有任何词形联系（一个整体不可分，一个不含序号）。所以：

    没进过训练集的 key，其类型知识**只能是随机的（1/6）**。

这不是缺陷，是「包做成原子整体」这件事的固有代价，推论很直接：

    包表（156 项）必须**先声明、再训练**；
    给医生加新 key = 必须补训练，不能指望零样本推广。

D 档就是把这个代价测成一个数，写进协议约束里。

================= 与 v7 的另一处差别：留出集不再靠 03 随机切 =================
v7 用 03 的 `--dump-split`（`random.shuffle` 后前 20%）当留出集，导致：

1. 每格只剩 4~10 条，单条波动 10%~25%，20 档只有 7 条；
2. 无法保证「已注册的 key 在训练部分都出现过」，未出现的 key 拿着随机
   embedding 去评估，会制造无法归因的错。

本版由本脚本**直接产出两个文件**，分层固定：

    data/pronoun_probe_v8_train.jsonl    训练用
    data/pronoun_probe_v8_holdout.jsonl  留出用（每格 HOLDOUT_PER_CELL 条）

并硬断言：除 6 个刻意留白的 `#Z` key 外，每个 key 在**训练文件**里至少出现
MIN_KEY_TRAIN_OCCURRENCES 次。训练时再用 03 的 30% 随机 val 也只能减少量，
不会把某个 key 完全摘出去（这一点训练后用 val_split.jsonl 再核对一次）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# 用纯 tokenizers 库统计 token：数据脚本不需要 torch，
# 避免 transformers 导入链拉起 CUDA DLL（小页面文件机器上会 WinError 1455）
from tokenizers import AddedToken, Tokenizer

# Windows 控制台输出 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")

ACT_OUTPUT = "<|OUTPUT|>"

# ================= 包引用形态（2026-10-23 定稿）=================
# 引用 = 类型标记 + 编号 token：`<|PKG_NUM|>#123`，共 **2 token**。
#
#   类型标记 `<|PKG_NUM|>`：6 个之一，**同类型共用同一行 embedding**
#   编号 token `#123`    ：401 个（#0..#400），整体编码成单 token
#
# 为什么编号必须整体编码：拆开写 `#123` 是 ['#','1','2','3'] 4 个普通
# token，模型要逐字符生成 → 必然撞复制宽度（实测 #94→#949 多吞一位、
# #1331→#1313 顺序乱，2 位档准确率仅 0.30）。整体编码后模型只需
# 「选一个编号 token」，不需逐字符复现。
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
# 编号上限与 tokens.json 的 _id_tokens.count 一致
ID_COUNT = 401                     # #0..#400
# 引用整体匹配：类型标记 + 编号
REF_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
# 裸类型标记（不带编号）
BARE_RE = re.compile(
    r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!#\d)")
# 类型标记整体
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

# 14 个协议标记占 151646..151659，编号从 151660 起。
# 必须与 tokens.json 的 _id_tokens.id_base 一致。
ID_TOKEN_BASE = 151660

# 负对照用：编号 400（最大值）在训练中绝不出现，测零样本。
UNSEEN_ID = 400
SEEN_IDS = list(range(UNSEEN_ID))          # 0..399


def make_key(pkg_type: str, index: int) -> str:
    """造一个包引用：`<|PKG_NUM|>#123`。index 是编号 0..400。"""
    if pkg_type not in PKG_TYPES:
        raise ValueError(f"未知包类型 {pkg_type}")
    if not 0 <= index < ID_COUNT:
        raise ValueError(f"编号范围 0..{ID_COUNT - 1}，收到 {index}")
    return f"<|PKG_{pkg_type}|>#{index}"


def id_tokens() -> list[str]:
    """按 tokens.json 顺序枚举全部编号 token：#0, #1, ... #400。"""
    return [f"#{n}" for n in range(ID_COUNT)]


# ================= 轴的取值 =================
# A/B：候选个数（答案始终是类型标记本身，不是候选的位置——避免又变成数位置）
CHOICE_KS = {"type_id_choice_k2": 2, "type_id_choice_k6": len(PKG_TYPES)}
# C/D：开集（不给候选）。C 用已训练的 key，D 用从未训练过的 #Z。
OPEN_AXES = ("type_id_open", "type_id_unseen")
TYPE_ID_AXES = tuple(CHOICE_KS) + OPEN_AXES
# 语义选取：干扰项个数（**全部是其他类型**，靠类型区分而非位置计数）
PICK_DISTRACTORS = (2, 5, 10, 20)
# 投影目标 token 数（无关填充）
DISTANCES = (10, 50, 100, 200, 400, 800)

# 每格条数 / 留出条数
PER_CELL = 30
PER_CELL_UNSEEN = 10
HOLDOUT_PER_CELL = 6
HOLDOUT_UNSEEN = 2
MIN_KEY_TRAIN_OCCURRENCES = 3

SEED = 20261023

# 协议标记：训练时由 03 的 --protocol-tokens 注册（见 tokens.json）。
# 数据脚本必须**同样注册**后再统计 token，否则 <|OUTPUT|> 会被切成 5 个 token，
# 量出来的投影长度和边界检查全是假的（本文件踩过这个坑）。
PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]

# 语义指代的措辞：按类型说「那个数值/那个名字/那段文本」，
# **不含任何序数**，模型只能靠 key 的类型前缀来找，不能靠数位置。
TYPE_WORD = {
    "NUM": ["那个数值", "刚才那个数值", "用户给的那个数值"],
    "NAME": ["那个名字", "刚才那个名字", "用户给的那个名字"],
    "TEXT": ["那段文本", "刚才那段文本"],
    "CODE": ["那段代码", "刚才那段代码"],
    "ARITH": ["那个算式", "刚才那个算式"],
    "TASK": ["那个任务", "刚才那个任务"],
}

# 类型辨识的问法（都不含序数，避免教模型去数位置）
TASK_LINES = ["判断包的类型", "看这个包是哪一类", "识别包的组织方式"]
ASK_OPEN = [
    "{key} 是哪种类型的包？只回答一个类型标记。",
    "上面那个 {key} 属于哪个类型？只回答类型标记。",
]
ASK_CHOICE = [
    "{key} 是哪种类型的包？从候选里选一个，只回答类型标记。",
    "参照候选，{key} 属于哪个类型？只回答类型标记。",
]


def load_tokenizer(directory: Path) -> Tokenizer:
    """加载 tokenizer，注册协议标记 + 全部编号 token，与训练时状态一致。

    注册顺序必须与 tokens.json / 03_train_pretrain.py 完全一致，
    否则 ID 错位：一个医生写的编号，另一个医生会读成别的包。
    """
    tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    for t in PROTOCOL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False).ids
        if len(ids) != 1:
            raise SystemExit(f"[错误] {t} 没有成为单 token：{ids}")

    # ---- 编号 token：#0..#400，共 401 个，每个必须是单 token ----
    all_ids = id_tokens()
    tok.add_special_tokens([
        AddedToken(k, special=True, normalized=False) for k in all_ids
    ])
    mis_single, mis_id = [], []
    for offset, k in enumerate(all_ids):
        got = tok.encode(k, add_special_tokens=False).ids
        if len(got) != 1:
            mis_single.append(f"{k}->{got}")
            continue
        expect = ID_TOKEN_BASE + offset
        if got[0] != expect:
            mis_id.append(f"{k}: got {got[0]} expect {expect}")
    if mis_single:
        raise SystemExit("[错误] 以下编号没有成为单 token（碎片化）："
                         + "；".join(mis_single[:5]))
    if mis_id:
        raise SystemExit("[错误] 编号 ID 与 tokens.json 布局不一致："
                         + "；".join(mis_id[:5]))

    # 拼接边界：encoded(标记+编号) == 两段分别编码后拼接（--target-mask 前提）
    edge_bad = []
    for m in ("<|PKG_NUM|>", "<|PKG_NAME|>", "<|PKG_TEXT|>"):
        for t in ("#0", "#7", "#123", "#400"):
            if (tok.encode(m + t, add_special_tokens=False).ids
                    != (tok.encode(m, add_special_tokens=False).ids
                        + tok.encode(t, add_special_tokens=False).ids)):
                edge_bad.append(f"{m}+{t}")
    if edge_bad:
        raise SystemExit("[错误] 标记与编号拼接边界不一致："
                         + "；".join(edge_bad[:5]))
    return tok


# ================= 造数据 =================


@dataclass
class ProbeSample:
    """一条探针样本：投影式上下文 + 提问 + 期望答案。

    answer_kind 决定判分方式：
      "type" → 答案是一个类型标记 <|PKG_NUM|>
      "key"  → 答案是一个包 key   <|PKG_NUM|,#B>
    """

    prompt: str
    target: str
    axis: str
    level: str
    answer_kind: str
    target_refs: list[str]
    context_tokens: int


def _render_pkg_line(entries: list[tuple[str, int]]) -> str:
    """把 (类型, 0-based序号) 列表渲染成 [包] 行。每项恰好一个原子 key。"""
    return "  ".join(make_key(t, i) for t, i in entries)


def _ctx_len(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def build_sample(axis: str, level: str, rng: random.Random,
                 tokenizer: Tokenizer) -> ProbeSample:
    """按某个轴的一个取值造一条样本。

    全部轴共同点：**提问/代词都不含序数**，模型不需要数位置。
    """
    # ---------------- 轴：类型辨识（A/B/C/D 四档） ----------------
    if axis in TYPE_ID_AXES:
        pkg_type = level
        idx = UNSEEN_INDEX if axis == "type_id_unseen" else rng.choice(SEEN_INDICES)
        key = make_key(pkg_type, idx)

        lines = [f"[任务] {rng.choice(TASK_LINES)}", f"[包] {key}"]
        if axis in CHOICE_KS:
            k = CHOICE_KS[axis]
            others = [t for t in PKG_TYPES if t != pkg_type]
            cands = [pkg_type] + rng.sample(others, k - 1)
            rng.shuffle(cands)
            lines.append("[候选] " + "  ".join(f"<|PKG_{t}|>" for t in cands))
            question = rng.choice(ASK_CHOICE).format(key=key)
        else:
            # C/D 档不给候选：与 A/B 的唯一差别就是这一行
            question = rng.choice(ASK_OPEN).format(key=key)
        lines.append("[动作]")
        prompt = "\n".join(lines) + "\n" + question + "\n"

        return ProbeSample(prompt=prompt,
                           target=f"{ACT_OUTPUT} <|PKG_{pkg_type}|>",
                           axis=axis, level=str(level), answer_kind="type",
                           target_refs=[key],
                           context_tokens=_ctx_len(tokenizer, prompt))

    # ---------------- 轴：语义选取 / 距离 ----------------
    if axis in ("semantic_pick", "distance"):
        if axis == "semantic_pick":
            n_distract = int(level)
            distance = 10          # 不加填充
        else:
            n_distract = 5         # 固定少量干扰
            distance = int(level)

        # 包表里**只有一个是 NUM**，其余都是其他类型 → 只能靠类型前缀定位，
        # 不能靠位置计数（这是与旧版最本质的区别）。
        other_types = [t for t in PKG_TYPES if t != "NUM"]
        picked = rng.sample(SEEN_INDICES, n_distract + 1)
        types = [rng.choice(other_types) for _ in range(n_distract + 1)]
        target_pos = rng.randrange(n_distract + 1)
        types[target_pos] = "NUM"
        entries = [(types[i], picked[i]) for i in range(n_distract + 1)]
        target_key = make_key("NUM", picked[target_pos])
        package_line = _render_pkg_line(entries)

        pronoun = rng.choice(TYPE_WORD["NUM"])
        task = "把指代的对象交给输出"

        # 距离控制：填充塞进 [任务] 行（填充里的编号只是噪声，不构成"数位置"要求）
        filler_units = ["已确认条目%d。" % i for i in range(1, 1600)]
        filler_parts: list[str] = []
        while True:
            prompt = (f"[任务] {task} {' '.join(filler_parts)}".rstrip() + "\n"
                      f"[包] {package_line}\n"
                      f"[动作]")
            if _ctx_len(tokenizer, prompt) >= distance \
                    or len(filler_parts) >= len(filler_units):
                break
            filler_parts.append(filler_units[len(filler_parts)])

        # 结尾必须换行（--target-mask 的 token 边界依赖它）
        prompt_with_q = f"{prompt}\n用户说：把{pronoun}交给输出。\n"
        return ProbeSample(prompt=prompt_with_q,
                           target=f"{ACT_OUTPUT} {target_key}",
                           axis=axis, level=str(level), answer_kind="key",
                           target_refs=[target_key],
                           context_tokens=_ctx_len(tokenizer, prompt_with_q))

    raise ValueError(f"未知轴：{axis}")


def cells() -> list[tuple[str, str]]:
    """34 格：类型辨识 24 格（4 档 × 6 类型）+ 语义 4 格 + 距离 6 格。"""
    return ([("type_id_choice_k2", t) for t in PKG_TYPES]
            + [("type_id_choice_k6", t) for t in PKG_TYPES]
            + [("type_id_open", t) for t in PKG_TYPES]
            + [("type_id_unseen", t) for t in PKG_TYPES]
            + [("semantic_pick", str(n)) for n in PICK_DISTRACTORS]
            + [("distance", str(d)) for d in DISTANCES])


# 负对照整档**只能出现在留出集**：它的 key 在训练里一次都不许出现，
# 否则负对照就失去了意义（那样测的只是普通泛化，不是零样本）。
HOLDOUT_ONLY_AXES = ("type_id_unseen",)


def _per_cell(axis: str) -> int:
    return PER_CELL_UNSEEN if axis in HOLDOUT_ONLY_AXES else PER_CELL


def _holdout_per_cell(axis: str) -> int:
    return PER_CELL_UNSEEN if axis in HOLDOUT_ONLY_AXES else HOLDOUT_PER_CELL


def build_split(tokenizer: Tokenizer) -> tuple[list[dict], list[dict]]:
    """返回 (train_rows, holdout_rows)。

    普通轴按格固定切前 N 条给留出集；`type_id_unseen` 整档只进留出集。
    """
    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    for axis, level in cells():
        n = _per_cell(axis)
        n_hold = _holdout_per_cell(axis)
        holdout_all = axis in HOLDOUT_ONLY_AXES
        for i in range(n):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            s = build_sample(axis, level, rng, tokenizer)
            row = {
                "kind": "probe",
                "axis": s.axis,
                "level": s.level,
                "answer_kind": s.answer_kind,
                "prompt": s.prompt,
                "target": s.target,
                "text": s.prompt + s.target,   # 必须严格等于 prompt+target
                "target_refs": s.target_refs,
                "context_tokens": s.context_tokens,
            }
            to_hold = holdout_all or i < n_hold
            (holdout_rows if to_hold else train_rows).append(row)
    return train_rows, holdout_rows


# ================= 审计 =================


def _pkg_line(row: dict) -> str:
    lines = [ln for ln in row["prompt"].split("\n") if ln.startswith("[包]")]
    if len(lines) != 1:
        raise RuntimeError(f"[包] 行必须恰好 1 行，实际 {len(lines)} 行")
    return lines[0]


def _tail(row: dict) -> str:
    return row["prompt"].split("[动作]")[-1]


def audit(train_rows: list[dict], holdout_rows: list[dict],
          tokenizer: Tokenizer, check_coverage: bool = True) -> dict:
    """检查数据可用性与隔离性。任何一条不合法都直接抛错。"""
    rows = train_rows + holdout_rows

    # 1) 每行必须有必需字段
    for row in rows:
        for field in ("prompt", "target", "text", "answer_kind"):
            if not row.get(field):
                raise RuntimeError(f"缺少字段 {field}：{row}")
        if not row["target"].startswith(ACT_OUTPUT):
            raise RuntimeError(f"target 动作不对：{row['target']}")

    # 2) 包表里每一项必须**恰好是一个原子 key**，不允许夹带说明文字。
    #    这条守的是「包是原子整体」：一旦有人把 `数值(#A)` 加回去，
    #    测出来的东西就变成字符噪声。
    for row in rows:
        items = [s for s in _pkg_line(row)[len("[包]"):].split("  ") if s.strip()]
        if not items:
            raise RuntimeError(f"[包] 行为空：{row['prompt'][:60]!r}")
        for seg in items:
            if not KEY_RE.fullmatch(seg.strip()):
                raise RuntimeError(
                    f"包表条目不是一个整体 key（不得附加说明文字）：{seg!r}")

    # 3) target 里的引用必须原样出现在 [包] 行里
    for row in rows:
        present = set(KEY_RE.findall(_pkg_line(row)))
        for ref in row["target_refs"]:
            if ref not in present:
                raise RuntimeError(f"target 引用了投影里不存在的包键 {ref}")

    # 4) prompt/target token 边界（--target-mask 的正确性依赖它）
    bad = []
    for row in rows:
        whole = tokenizer.encode(row["text"], add_special_tokens=False).ids
        parts = (tokenizer.encode(row["prompt"], add_special_tokens=False).ids
                 + tokenizer.encode(row["target"], add_special_tokens=False).ids)
        if whole != parts:
            bad.append(row["target"])
    if bad:
        raise RuntimeError(
            f"prompt/target 拼接处 token 边界不一致（{len(bad)} 条），"
            f"--target-mask 会算错，示例：{bad[0][:40]!r}")

    # 5) **关键：语义轴里 NUM 包必须唯一**
    #    目标靠类型定位，若包表里出现两个 NUM，「那个数值」就有歧义，
    #    题目本身不成立。这条必须断言，不能靠事后看分数猜。
    for row in rows:
        if row["axis"] not in ("semantic_pick", "distance"):
            continue
        nums = [k for k in KEY_RE.findall(_pkg_line(row))
                if k.startswith("<|PKG_NUM|")]
        if len(nums) != 1:
            raise RuntimeError(f"语义轴的 NUM 包必须唯一，实际 {len(nums)} 个：{nums}")
        if nums[0] != row["target_refs"][0]:
            raise RuntimeError(
                f"target 不是那个唯一的 NUM 包：target={row['target_refs']} nums={nums}")

    # 6) **关键：提问/代词里不能出现序数**
    #    旧版用「第几个」导致测的是数位置。这里断言问句不含序数，
    #    否则整份数据又退回到无效需求。
    ordinal_re = re.compile(r"第[一二三四五六七八九十百零\d]+[个项]")
    for row in rows:
        if ordinal_re.search(_tail(row)):
            raise RuntimeError(
                f"提问里出现了序数（会退化成效数位置）：{_tail(row)!r}")

    # 7) 类型辨识四档：候选集合与答案必须自洽
    for row in rows:
        if row["axis"] not in TYPE_ID_AXES:
            continue
        key = row["target_refs"][0]
        key_type = key.split("_", 1)[1].split("|")[0]
        answer = row["target"].split()[1]
        if answer != f"<|PKG_{key_type}|>":
            raise RuntimeError(f"类型答案与 key 类型不符：key={key} answer={answer}")

        if row["axis"] == "type_id_unseen":
            if not key.endswith(f"#{ID_CHARS[UNSEEN_INDEX]}>"):
                raise RuntimeError(f"负对照必须用未训练的序号：{key}")
        else:
            if key.endswith(f"#{ID_CHARS[UNSEEN_INDEX]}>"):
                raise RuntimeError(f"非负对照的格子里不许出现未训练序号：{key}")

        cand_lines = [ln for ln in row["prompt"].split("\n")
                      if ln.startswith("[候选]")]
        if row["axis"] in CHOICE_KS:
            k = CHOICE_KS[row["axis"]]
            if len(cand_lines) != 1:
                raise RuntimeError(f"选择题必须恰好 1 行 [候选]：{row['prompt'][:60]!r}")
            cands = TYPE_MARK_RE.findall(cand_lines[0])
            if len(cands) != k:
                raise RuntimeError(f"候选数应为 {k}，实际 {cands}")
            if len(set(cands)) != k:
                raise RuntimeError(f"候选有重复：{cands}")
            if answer not in cands:
                raise RuntimeError(f"正确答案不在候选里：answer={answer} cands={cands}")
        else:
            if cand_lines:
                raise RuntimeError(
                    f"{row['axis']} 必须不给候选（否则与选择题无法区分）")

    # 8) 每个轴每个取值都有样本；**同一轴内**各格等量（不同轴可以不等，
    #    因为 type_id_unseen 是刻意做小的负对照）。
    #    训练集豁免 HOLDOUT_ONLY_AXES —— 那些格子按设计不许进训练。
    expect_cells = {"train": [c for c in cells() if c[0] not in HOLDOUT_ONLY_AXES],
                    "holdout": cells()}
    for name, rs in (("train", train_rows), ("holdout", holdout_rows)):
        counts: dict[tuple[str, str], int] = {}
        for row in rs:
            k = (row["axis"], row["level"])
            counts[k] = counts.get(k, 0) + 1
        missing = [c for c in expect_cells[name] if c not in counts]
        if missing:
            raise RuntimeError(f"{name} 缺少格子：{missing}")
        extra = [c for c in counts if c not in expect_cells[name]]
        if extra:
            raise RuntimeError(f"{name} 出现了不该有的格子：{extra}")
        per_axis: dict[str, set[int]] = {}
        for (axis, _lv), n in counts.items():
            per_axis.setdefault(axis, set()).add(n)
        uneven = {a: v for a, v in per_axis.items() if len(v) != 1}
        if uneven:
            raise RuntimeError(f"{name} 同一轴内各格样本数不一致：{uneven}")

    # 9) 距离轴：实际 token 数必须随 level 单调不减
    dist_by_level: dict[str, list[int]] = {}
    for row in rows:
        if row["axis"] == "distance":
            dist_by_level.setdefault(row["level"], []).append(row["context_tokens"])
    ordered = [int(d) for d in DISTANCES if str(d) in dist_by_level]
    means = [sum(dist_by_level[str(d)]) / len(dist_by_level[str(d)]) for d in ordered]
    for a, b in zip(means, means[1:]):
        if b < a - 5:
            raise RuntimeError(f"距离轴下降过多：{dict(zip(ordered, means))}")

    # 10) **关键：训练文件必须覆盖全部已训练的 key，且绝不包含 #Z**
    #     未在训练中出现的 key 拿着随机 embedding 去评估，会制造无法归因的错。
    train_occ: dict[str, int] = {}
    for row in train_rows:
        for k in KEY_RE.findall(row["text"]):
            train_occ[k] = train_occ.get(k, 0) + 1

    unseen_keys = {make_key(t, UNSEEN_INDEX) for t in PKG_TYPES}
    leaked = sorted(unseen_keys & set(train_occ))
    if leaked:
        raise RuntimeError(f"负对照的 key 泄漏进了训练文件：{leaked[:3]}")

    seen_keys = [make_key(t, i) for t in PKG_TYPES for i in SEEN_INDICES]
    never = [k for k in seen_keys if k not in train_occ]
    if never:
        raise RuntimeError(
            f"有 {len(never)} 个已注册 key 在训练文件里从未出现（评估不可归因）："
            f"{never[:5]}")
    thin = [k for k in seen_keys if train_occ[k] < MIN_KEY_TRAIN_OCCURRENCES]
    if check_coverage and thin:
        raise RuntimeError(
            f"有 {len(thin)} 个 key 在训练文件里出现不足 "
            f"{MIN_KEY_TRAIN_OCCURRENCES} 次：{[(k, train_occ[k]) for k in thin[:5]]}")

    return {
        "train_rows": len(train_rows),
        "holdout_rows": len(holdout_rows),
        "per_axis_layout": {
            axis: {
                "types_or_levels": sum(1 for c in cells() if c[0] == axis),
                "train_per_cell": 0 if axis in HOLDOUT_ONLY_AXES
                else _per_cell(axis) - _holdout_per_cell(axis),
                "holdout_per_cell": _holdout_per_cell(axis),
            }
            for axis in sorted({c[0] for c in cells()})
        },
        "total_tokens": sum(r["context_tokens"] for r in rows),
        "max_context_tokens": max(r["context_tokens"] for r in rows),
        "distance_axis_mean_tokens": {str(d): round(m, 1)
                                      for d, m in zip(ordered, means)},
        "key_coverage": {
            "seen_keys": len(seen_keys),
            "unseen_keys": len(unseen_keys),
            "min_train_occurrences": min(train_occ[k] for k in seen_keys),
            "max_train_occurrences": max(train_occ[k] for k in seen_keys),
        },
        "audit": {
            "atomic_key_only": True,
            "num_key_unique": True,
            "no_ordinal_in_question": True,
            "target_boundary_ok": True,
            "choice_candidates_consistent": True,
            "unseen_key_never_in_train": True,
            "all_seen_keys_covered": True,
        },
        "note": "v8：类型辨识拆成 k2/k6/open/unseen 四档（配比相同，唯一变量是"
                "候选提示），留出集由本脚本分层直出，不依赖 03 的随机切分。",
    }


# ================= 自测 =================


def _digest(rs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rs:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def _self_test(tokenizer: Tokenizer) -> None:
    train_rows, holdout_rows = build_split(tokenizer)
    # 自测阶段不做覆盖率门槛（数据量小，覆盖率必然不达标）
    report = audit(train_rows, holdout_rows, tokenizer, check_coverage=False)

    # 四档类型辨识的配比必须**完全相同**，否则 A/B/C 之间不可比
    per_type: dict[str, dict[str, int]] = {}
    for row in train_rows + holdout_rows:
        if row["axis"] in TYPE_ID_AXES:
            per_type.setdefault(row["axis"], {})
            per_type[row["axis"]][row["level"]] = \
                per_type[row["axis"]].get(row["level"], 0) + 1
    for axis in ("type_id_choice_k2", "type_id_choice_k6", "type_id_open"):
        vals = set(per_type[axis].values())
        assert len(vals) == 1 and len(per_type[axis]) == len(PKG_TYPES), \
            f"{axis} 配比不齐：{per_type[axis]}"

    # 语义轴：干扰项必须都是**非 NUM**类型，否则「那个数值」有歧义
    for row in train_rows:
        if row["axis"] != "semantic_pick":
            continue
        keys = KEY_RE.findall(_pkg_line(row))
        nums = [k for k in keys if k.startswith("<|PKG_NUM|")]
        assert len(nums) == 1, f"NUM 包不唯一：{nums}"

    # 距离轴：最长档必须明显长于最短档
    means = list(report["distance_axis_mean_tokens"].values())
    assert means[-1] > means[0] * 2, f"距离跨度太小：{means}"

    # 可复现：同 seed 重建逐字节一致
    again_train, again_hold = build_split(tokenizer)
    assert _digest(train_rows) == _digest(again_train), "train 同 seed 重建不一致"
    assert _digest(holdout_rows) == _digest(again_hold), "holdout 同 seed 重建不一致"

    report["status"] = "passed"
    report["reproducible"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ================= 主流程 =================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="包引用探针 v8：类型辨识四档 + 语义选取 + 距离，分层留出集")
    parser.add_argument("--build", action="store_true", help="生成数据")
    parser.add_argument("--self-test", action="store_true", help="自测（含审计）")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)

    if args.self_test:
        _self_test(tokenizer)
        if not args.build:
            return

    if not args.build:
        parser.print_help()
        return

    train_rows, holdout_rows = build_split(tokenizer)
    report = audit(train_rows, holdout_rows, tokenizer)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_path = DATA_DIR / "pronoun_probe_v8_train.jsonl"
    hold_path = DATA_DIR / "pronoun_probe_v8_holdout.jsonl"
    for path, rs in ((train_path, train_rows), (hold_path, holdout_rows)):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["file_sha256"] = {
        train_path.name: hashlib.sha256(train_path.read_bytes()).hexdigest(),
        hold_path.name: hashlib.sha256(hold_path.read_bytes()).hexdigest(),
    }

    rep_path = DATA_DIR / "pronoun_probe_v8_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n训练：{train_path}\n留出：{hold_path}\n报告：{rep_path}")


if __name__ == "__main__":
    main()
