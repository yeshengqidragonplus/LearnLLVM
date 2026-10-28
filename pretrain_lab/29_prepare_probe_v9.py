# -*- coding: utf-8 -*-
"""包引用探针 v9：定稿形态 `类型标记 + 整体编号 token`。

================= 形态演进（三版试错后才定）=================
| 版本 | 引用形态 | 问题 |
|---|---|---|
| v6/v7/v8 | 156 个 per-key token | 同类型包各占独立 embedding，违背「同类型共用 token」；未训练 key 那行随机，实测 0/60 |
| 26 号 | 编号拆成 4 个普通 token | 逐字符生成 → 复制宽度：94 变 949 多吞一位、1331 变 1313 顺序乱，2 位档仅 0.30 |
| 28 号 | 编号定长补零（仍拆开） | 缓解到 0.90，仍是逐字符复现，且每编号多 2~3 token |
| **本版** | **编号整体编码成单 token** | **无复制问题**：编号是单 token，模型只需「选一个」 |

================= 定稿形态 =================
一个引用 = **2 token**：

    <|PKG_NUM|>  #123
    └─ 类型 ─┘   └编号┘
    6 个之一     401 个之一（#0..#400）
    同类型共用    所有类型共享同一套编号

- 类型标记同类型共用一行 embedding（用户设计的核心）
- 编号 `#0..#400` 整体编码，每个单 token、ID 连续（151660..152060）
- 展开 = 程序按 (类型, 编号) 查缓存替换，**不走大模型**；替换出的内容若仍含
  标记就再展开 → 嵌套循环直到无标记

成本对比（实测）：引用 2 token（拆开 3~5）；包表 22 项 65 token（拆开 114），省 43%。

================= 本版测三条轴 =================
  type_id        给一个引用，说出它的类型（答案 1 token，封闭集合）
  semantic_pick  包表里只有一个 NUM，其余异类型；按「那个数值」选出它
  distance       同上 + 无关填充到 800 token，测投影能写多长

三条轴都**不含序数**：协议里模型不需要数「第几个包」，只需按类型前缀定位。

================= 两条硬断言（防结论不可归因）=================
1. `num_ref_unique`：语义轴里 NUM 引用必须唯一，否则「那个数值」有歧义。
2. `no_ordinal_in_question`：提问不许出现「第几个」，否则退化成数位置。

================= 留出集由本脚本分层直出 =================
不依赖 03 的 `random.shuffle` 切 20%（那样每格只剩几条，且无法保证
已训练编号在训练部分都出现过）。审计断言：除刻意留白的 `#400` 外，
每个已训练编号在训练文件里至少出现 MIN_ID_TRAIN_OCCURRENCES 次。
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

from tokenizers import AddedToken, Tokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
ACT_OUTPUT = "<|OUTPUT|>"

# ================= 包引用形态 =================
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_COUNT = 401                 # #0..#400，与 tokens.json 的 _id_tokens.count 一致
ID_TOKEN_BASE = 151660         # 14 个协议标记占 151646..151659

# 引用整体：类型标记 + #数字
REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
# 裸类型标记（不带编号）
BARE_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!#\d)")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

# 负对照：编号 400 训练中绝不出现，测「未训练编号能否工作」
UNSEEN_ID = 400
SEEN_IDS = list(range(UNSEEN_ID))       # 0..399

# ================= 轴的取值 =================
CHOICE_KS = {"type_id_choice_k2": 2, "type_id_choice_k6": len(PKG_TYPES)}
OPEN_AXES = ("type_id_open", "type_id_unseen")
TYPE_ID_AXES = tuple(CHOICE_KS) + OPEN_AXES
PICK_DISTRACTORS = (2, 5, 10, 20)
DISTANCES = (10, 50, 100, 200, 400, 800)

# 样本量：**只有 target 含编号的轴才训练「选对编号」这个能力**。
# 第一轮教训（PER_CELL 全 30）：400 个编号在 target 里平均只被监督
# 0.60 次（240 次总监督 / 400 编号），结果「类型全对、编号全错」。
# 所以含编号的轴（semantic_pick / distance）必须远大于 type_id 轴。
PER_CELL = 30                 # type_id 四档：target 只含类型标记
PER_CELL_REF = 150            # semantic_pick / distance：target 含编号
PER_CELL_UNSEEN = 10
HOLDOUT_PER_CELL = 6
HOLDOUT_REF_PER_CELL = 30
HOLDOUT_UNSEEN = 2
MIN_ID_TRAIN_OCCURRENCES = 3
SEED = 20261026

# 哪些轴的 target 里含编号（决定编号选择是否被监督）
REF_AXES = ("semantic_pick", "distance")

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]

TYPE_WORD = {
    "NUM": ["那个数值", "刚才那个数值", "用户给的那个数值"],
    "NAME": ["那个名字", "刚才那个名字", "用户给的那个名字"],
    "TEXT": ["那段文本", "刚才那段文本"],
    "CODE": ["那段代码", "刚才那段代码"],
    "ARITH": ["那个算式", "刚才那个算式"],
    "TASK": ["那个任务", "刚才那个任务"],
}

TASK_LINES = ["判断包的类型", "看这个包是哪一类", "识别包的组织方式"]
ASK_OPEN = [
    "{ref} 是哪种类型的包？只回答一个类型标记。",
    "上面那个 {ref} 属于哪个类型？只回答类型标记。",
]
ASK_CHOICE = [
    "{ref} 是哪种类型的包？从候选里选一个，只回答类型标记。",
    "参照候选，{ref} 属于哪个类型？只回答类型标记。",
]


def make_ref(pkg_type: str, n: int) -> str:
    """造一个包引用：`<|PKG_NUM|>#123`（类型标记 + 整体编号 token）。"""
    if pkg_type not in PKG_TYPES:
        raise ValueError(f"未知包类型 {pkg_type}")
    if not 0 <= n < ID_COUNT:
        raise ValueError(f"编号范围 0..{ID_COUNT - 1}，收到 {n}")
    return f"<|PKG_{pkg_type}|>#{n}"


def id_tokens() -> list[str]:
    """按 tokens.json 顺序枚举全部编号 token：#0, #1, ... #400。"""
    return [f"#{n}" for n in range(ID_COUNT)]


def load_tokenizer(directory: Path) -> Tokenizer:
    """注册协议标记 + 全部编号 token，与训练时状态一致。

    注册顺序必须与 tokens.json / 03_train_pretrain.py 完全一致，
    否则 ID 错位：一个医生写的编号，另一个医生会读成别的包。
    """
    tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    bad = [t for t in PROTOCOL_TOKENS
           if len(tok.encode(t, add_special_tokens=False).ids) != 1]
    if bad:
        raise SystemExit(f"[错误] 协议标记未成为单 token：{bad[:3]}")

    all_ids = id_tokens()
    tok.add_special_tokens([
        AddedToken(k, special=True, normalized=False) for k in all_ids
    ])
    mis_single, mis_id = [], []
    for offset, k in enumerate(all_ids):
        got = tok.encode(k, add_special_tokens=False).ids
        if len(got) != 1:
            mis_single.append(f"{k}->{got}")
        elif got[0] != ID_TOKEN_BASE + offset:
            mis_id.append(f"{k}: got {got[0]} expect {ID_TOKEN_BASE + offset}")
    if mis_single:
        raise SystemExit("[错误] 编号未成为单 token（碎片化）："
                         + "；".join(mis_single[:5]))
    if mis_id:
        raise SystemExit("[错误] 编号 ID 与 tokens.json 布局不一致："
                         + "；".join(mis_id[:5]))

    # 拼接边界：encoded(标记+编号) 必须 == 两段分别编码后拼接。
    # --target-mask 的监督位置依赖这一点，错位时不会报错、只会静默算错 loss。
    edge_bad = []
    for m in ("<|PKG_NUM|>", "<|PKG_NAME|>", "<|PKG_TEXT|>", "<|PKG_TASK|>"):
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
class Sample:
    prompt: str
    target: str
    axis: str
    level: str
    answer_kind: str           # "type"（答案类型标记）| "ref"（答案包引用）
    target_refs: list[str]
    ctx_tokens: int


def _pkg_line(entries: list[tuple[str, int]]) -> str:
    """把 (类型, 编号) 列表渲染成 [包] 行。每项恰好一个引用（2 token）。"""
    return "  ".join(make_ref(t, n) for t, n in entries)


def _ctx(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def build_sample(axis: str, level: str, rng: random.Random,
                 tok: Tokenizer, offset: int = 0) -> Sample:
    """按某个轴的一个取值造一条样本。全部轴都不含序数。

    `offset` 是**全局递增**的样本计数器（跨格累加）。编号按它取模轮转，
    而不是纯随机：纯随机会漏掉个别编号（实测漏了 #139），而审计要求
    「每个已训练编号在训练文件里都出现过」——否则未出现的编号 token
    拿着随机 embedding 去评估，会制造无法归因的错。
    """
    # ---------------- 类型辨识（A/B/C/D 四档）----------------
    if axis in TYPE_ID_AXES:
        pkg_type = level
        if axis == "type_id_unseen":
            n = UNSEEN_ID
        else:
            # 编号由全局计数器经 LCG 打散后取模。
            # 直接用 offset % 400 会与「每格条数」产生周期性空档（实测漏 12 个），
            # 用质数乘子打散可确保 400 个样本内均匀覆盖全部编号。
            n = (offset * 151 + 97) % len(SEEN_IDS)
        ref = make_ref(pkg_type, n)

        lines = [f"[任务] {rng.choice(TASK_LINES)}", f"[包] {ref}"]
        if axis in CHOICE_KS:
            k = CHOICE_KS[axis]
            others = [t for t in PKG_TYPES if t != pkg_type]
            cands = [pkg_type] + rng.sample(others, k - 1)
            rng.shuffle(cands)
            lines.append("[候选] " + "  ".join(f"<|PKG_{t}|>" for t in cands))
            question = rng.choice(ASK_CHOICE).format(ref=ref)
        else:
            question = rng.choice(ASK_OPEN).format(ref=ref)
        lines.append("[动作]")
        prompt = "\n".join(lines) + "\n" + question + "\n"

        return Sample(prompt, f"{ACT_OUTPUT} <|PKG_{pkg_type}|>", axis,
                      str(level), "type", [ref], _ctx(tok, prompt))

    # ---------------- 语义选取 / 距离 ----------------
    if axis in ("semantic_pick", "distance"):
        if axis == "semantic_pick":
            n_distract = int(level)
            distance = 10
        else:
            n_distract = 5
            distance = int(level)

        # 包表里**只有一个是 NUM**，其余都是其他类型 → 只能靠类型前缀定位
        other_types = [t for t in PKG_TYPES if t != "NUM"]
        # 编号按全局计数器轮转，保证每个编号都进过训练（见 build_sample 说明）
        picked: list[int] = []
        cur = (offset * 151 + 97) % len(SEEN_IDS)
        while len(picked) < n_distract + 1:
            if cur not in picked:
                picked.append(cur)
            cur = (cur + 1) % len(SEEN_IDS)
        types = [rng.choice(other_types) for _ in range(n_distract + 1)]
        target_pos = rng.randrange(n_distract + 1)
        types[target_pos] = "NUM"
        entries = [(types[i], picked[i]) for i in range(n_distract + 1)]
        target_ref = make_ref("NUM", picked[target_pos])
        package_line = _pkg_line(entries)

        pronoun = rng.choice(TYPE_WORD["NUM"])
        task = "把指代的对象交给输出"

        # 距离控制：填充塞进 [任务] 行（填充里的编号只是噪声，不构成数位置要求）
        filler_units = ["已确认条目%d。" % i for i in range(1, 1600)]
        filler_parts: list[str] = []
        while True:
            prompt = (f"[任务] {task} {' '.join(filler_parts)}".rstrip() + "\n"
                      f"[包] {package_line}\n"
                      f"[动作]")
            if _ctx(tok, prompt) >= distance \
                    or len(filler_parts) >= len(filler_units):
                break
            filler_parts.append(filler_units[len(filler_parts)])

        # 结尾必须换行（--target-mask 的 token 边界依赖它）
        prompt_with_q = f"{prompt}\n用户说：把{pronoun}交给输出。\n"
        return Sample(prompt_with_q, f"{ACT_OUTPUT} {target_ref}", axis,
                      str(level), "ref", [target_ref],
                      _ctx(tok, prompt_with_q))

    raise ValueError(f"未知轴：{axis}")


def cells() -> list[tuple[str, str]]:
    """34 格：类型辨识 24 格 + 语义 4 格 + 距离 6 格。"""
    return ([("type_id_choice_k2", t) for t in PKG_TYPES]
            + [("type_id_choice_k6", t) for t in PKG_TYPES]
            + [("type_id_open", t) for t in PKG_TYPES]
            + [("type_id_unseen", t) for t in PKG_TYPES]
            + [("semantic_pick", str(n)) for n in PICK_DISTRACTORS]
            + [("distance", str(d)) for d in DISTANCES])


HOLDOUT_ONLY_AXES = ("type_id_unseen",)


def _per_cell(axis: str) -> int:
    """含编号的轴给更多样本（只看它们训练「选对编号」）。"""
    if axis in HOLDOUT_ONLY_AXES:
        return PER_CELL_UNSEEN
    return PER_CELL_REF if axis in REF_AXES else PER_CELL


def _holdout_per_cell(axis: str) -> int:
    if axis in HOLDOUT_ONLY_AXES:
        return PER_CELL_UNSEEN
    return HOLDOUT_REF_PER_CELL if axis in REF_AXES else HOLDOUT_PER_CELL


def build_split(tok: Tokenizer) -> tuple[list[dict], list[dict]]:
    """返回 (train_rows, holdout_rows)。

    普通轴按格固定切前 N 条给留出集；`type_id_unseen` 整档只进留出集
    （否则负对照就退化成普通泛化）。
    """
    train: list[dict] = []
    hold: list[dict] = []
    offset = 0          # 全局递增计数器：保证编号覆盖到 #0..#399
    for axis, level in cells():
        n = _per_cell(axis)
        n_hold = _holdout_per_cell(axis)
        to_hold_all = axis in HOLDOUT_ONLY_AXES
        for i in range(n):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            s = build_sample(axis, level, rng, tok, offset=offset)
            offset += 1
            row = {
                "kind": "probe",
                "axis": s.axis, "level": s.level,
                "answer_kind": s.answer_kind,
                "prompt": s.prompt, "target": s.target,
                "text": s.prompt + s.target,   # 必须严格等于 prompt+target
                "target_refs": s.target_refs,
                "context_tokens": s.ctx_tokens,
            }
            (hold if (to_hold_all or i < n_hold) else train).append(row)
    return train, hold


# ================= 审计 =================


def _pkg_line_of(row: dict) -> str:
    lines = [ln for ln in row["prompt"].split("\n") if ln.startswith("[包]")]
    if len(lines) != 1:
        raise RuntimeError(f"[包] 行必须恰好 1 行，实际 {len(lines)} 行")
    return lines[0]


def _tail(row: dict) -> str:
    return row["prompt"].split("[动作]")[-1]


def audit(train: list[dict], hold: list[dict], tok: Tokenizer,
          check_coverage: bool = True) -> dict:
    rows = train + hold

    # 1) 必需字段
    for row in rows:
        for f in ("prompt", "target", "text", "answer_kind"):
            if not row.get(f):
                raise RuntimeError(f"缺少字段 {f}：{row}")
        if not row["target"].startswith(ACT_OUTPUT):
            raise RuntimeError(f"target 动作不对：{row['target']}")

    # 2) [包] 行每一项必须恰好是一个引用（类型标记 + 编号），不得夹带说明文字
    for row in rows:
        items = [s for s in _pkg_line_of(row)[len("[包]"):].split("  ") if s.strip()]
        if not items:
            raise RuntimeError(f"[包] 行为空：{row['prompt'][:60]!r}")
        for seg in items:
            if not REF_RE.fullmatch(seg.strip()):
                raise RuntimeError(
                    f"[包] 条目不是完整引用 `<|PKG_X|>#N`：{seg!r}")

    # 3) target 里的引用必须原样出现在 [包] 行
    for row in rows:
        if row["answer_kind"] != "ref":
            continue
        present = {f"{m}#{n}" for m, n in
                   REF_RE.findall(_pkg_line_of(row))}
        # 用「类型#编号」组合比对，避免把不同编号误配
        for ref in row["target_refs"]:
            mm = REF_RE.fullmatch(ref)
            if not mm or f"{mm.group(1)}#{mm.group(2)}" not in present:
                raise RuntimeError(f"target 引用了投影里不存在的包 {ref}")

    # 4) prompt/target token 边界（--target-mask 正确性依赖它）
    bad = [r["target"] for r in rows
           if tok.encode(r["text"], add_special_tokens=False).ids
           != (tok.encode(r["prompt"], add_special_tokens=False).ids
               + tok.encode(r["target"], add_special_tokens=False).ids)]
    if bad:
        raise RuntimeError(
            f"prompt/target 拼接处 token 边界不一致（{len(bad)} 条），"
            f"--target-mask 会算错，示例：{bad[0][:40]!r}")

    # 5) **关键：语义轴里 NUM 引用必须唯一**
    #    目标靠类型前缀定位，若出现两个 NUM，「那个数值」就有歧义，
    #    题目本身不成立。这条必须断言，不能靠事后看分数猜。
    for row in rows:
        if row["axis"] not in ("semantic_pick", "distance"):
            continue
        nums = [(m, n) for m, n in REF_RE.findall(_pkg_line_of(row))
                if m == "NUM"]
        if len(nums) != 1:
            raise RuntimeError(f"语义轴的 NUM 引用必须唯一，实际 {len(nums)}：{nums}")
        mm = REF_RE.fullmatch(row["target_refs"][0])
        if not mm or (mm.group(1), mm.group(2)) != nums[0]:
            raise RuntimeError(
                f"target 不是那个唯一的 NUM：target={row['target_refs']} nums={nums}")

    # 6) **关键：提问/代词里不能出现序数**
    ordinal_re = re.compile(r"第[一二三四五六七八九十百零\d]+[个项]")
    for row in rows:
        if ordinal_re.search(_tail(row)):
            raise RuntimeError(
                f"提问里出现了序数（会退化成数位置）：{_tail(row)!r}")

    # 7) 类型辨识四档：候选与答案自洽
    for row in rows:
        if row["axis"] not in TYPE_ID_AXES:
            continue
        ref = row["target_refs"][0]
        mm = REF_RE.fullmatch(ref)
        if not mm:
            raise RuntimeError(f"类型轴的引用形态不对：{ref}")
        key_type, key_id = mm.group(1), int(mm.group(2))
        answer = row["target"].split()[1]
        if answer != f"<|PKG_{key_type}|>":
            raise RuntimeError(f"类型答案与引用不符：ref={ref} answer={answer}")

        if row["axis"] == "type_id_unseen":
            if key_id != UNSEEN_ID:
                raise RuntimeError(f"负对照必须用未训练编号 #{UNSEEN_ID}：{ref}")
        elif key_id == UNSEEN_ID:
            raise RuntimeError(f"非负对照不许用未训练编号：{ref}")

        cand_lines = [ln for ln in row["prompt"].split("\n")
                      if ln.startswith("[候选]")]
        if row["axis"] in CHOICE_KS:
            k = CHOICE_KS[row["axis"]]
            if len(cand_lines) != 1:
                raise RuntimeError(f"选择题必须恰好 1 行 [候选]")
            cands = TYPE_MARK_RE.findall(cand_lines[0])
            if len(cands) != k or len(set(cands)) != k:
                raise RuntimeError(f"候选数应为 {k} 且不重复：{cands}")
            if answer not in cands:
                raise RuntimeError(f"正确答案不在候选里：{answer} vs {cands}")
        else:
            if cand_lines:
                raise RuntimeError(f"{row['axis']} 必须不给候选")

    # 8) 每格都有样本；训练集豁免 HOLDOUT_ONLY_AXES；同轴内等量
    expect = {"train": [c for c in cells() if c[0] not in HOLDOUT_ONLY_AXES],
              "holdout": cells()}
    for name, rs in (("train", train), ("holdout", hold)):
        cnt: dict[tuple[str, str], int] = {}
        for r in rs:
            cnt[(r["axis"], r["level"])] = cnt.get((r["axis"], r["level"]), 0) + 1
        missing = [c for c in expect[name] if c not in cnt]
        if missing:
            raise RuntimeError(f"{name} 缺少格子：{missing}")
        extra = [c for c in cnt if c not in expect[name]]
        if extra:
            raise RuntimeError(f"{name} 出现了不该有的格子：{extra}")
        per_axis: dict[str, set[int]] = {}
        for (a, _lv), n in cnt.items():
            per_axis.setdefault(a, set()).add(n)
        uneven = {a: v for a, v in per_axis.items() if len(v) != 1}
        if uneven:
            raise RuntimeError(f"{name} 同轴内各格不等量：{uneven}")

    # 9) 距离轴 token 数单调不减
    dist: dict[str, list[int]] = {}
    for r in rows:
        if r["axis"] == "distance":
            dist.setdefault(r["level"], []).append(r["context_tokens"])
    ordered = [int(d) for d in DISTANCES if str(d) in dist]
    means = [sum(dist[str(d)]) / len(dist[str(d)]) for d in ordered]
    for a, b in zip(means, means[1:]):
        if b < a - 5:
            raise RuntimeError(f"距离轴下降过多：{dict(zip(ordered, means))}")

    # 10) **关键：训练文件必须覆盖全部已训练编号，且绝不包含 #400**
    #     未在训练中出现的编号 token 拿着随机 embedding 评估，会制造无法归因的错。
    train_occ: dict[int, int] = {}
    for r in train:
        for _m, n in REF_RE.findall(r["text"]):
            train_occ[int(n)] = train_occ.get(int(n), 0) + 1

    if UNSEEN_ID in train_occ:
        raise RuntimeError(f"负对照编号 #{UNSEEN_ID} 泄漏进了训练文件")

    never = [n for n in SEEN_IDS if n not in train_occ]
    if never:
        raise RuntimeError(
            f"有 {len(never)} 个编号在训练文件里从未出现（评估不可归因）："
            f"{never[:8]}")
    thin = [n for n in SEEN_IDS
            if train_occ[n] < MIN_ID_TRAIN_OCCURRENCES]
    if check_coverage and thin:
        raise RuntimeError(
            f"有 {len(thin)} 个编号出现不足 {MIN_ID_TRAIN_OCCURRENCES} 次："
            f"{[(n, train_occ[n]) for n in thin[:8]]}")

    return {
        "train_rows": len(train), "holdout_rows": len(hold),
        "per_axis_layout": {
            axis: {"cells": sum(1 for c in cells() if c[0] == axis),
                   "train_per_cell": 0 if axis in HOLDOUT_ONLY_AXES
                   else _per_cell(axis) - _holdout_per_cell(axis),
                   "holdout_per_cell": _holdout_per_cell(axis)}
            for axis in sorted({c[0] for c in cells()})
        },
        "total_tokens": sum(r["context_tokens"] for r in rows),
        "max_context_tokens": max(r["context_tokens"] for r in rows),
        "distance_axis_mean_tokens": {str(d): round(m, 1)
                                      for d, m in zip(ordered, means)},
        "id_coverage": {
            "seen_ids": len(SEEN_IDS), "unseen_ids": 1,
            "min_train_occurrences": min(train_occ[n] for n in SEEN_IDS),
            "max_train_occurrences": max(train_occ[n] for n in SEEN_IDS),
        },
        "package_form": f"<|PKG_<TYPE>|>#<ID>，ID 为整体编码 token #0..#{ID_COUNT - 1}",
        "reference_tokens": 2,
        "audit": {
            "ref_is_complete": True,
            "num_ref_unique": True,
            "no_ordinal_in_question": True,
            "target_boundary_ok": True,
            "choice_candidates_consistent": True,
            "unseen_id_never_in_train": True,
            "all_seen_ids_covered": True,
        },
    }


# ================= 自测 =================


def _digest(rs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rs:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def _self_test(tok: Tokenizer) -> None:
    train, hold = build_split(tok)
    rep = audit(train, hold, tok, check_coverage=False)

    # 四档类型辨识配比必须完全一致（否则 A/B/C 之间不可比）
    per_type: dict[str, dict[str, int]] = {}
    for row in train + hold:
        if row["axis"] in TYPE_ID_AXES:
            per_type.setdefault(row["axis"], {})
            per_type[row["axis"]][row["level"]] = \
                per_type[row["axis"]].get(row["level"], 0) + 1
    for axis in ("type_id_choice_k2", "type_id_choice_k6", "type_id_open"):
        vals = set(per_type[axis].values())
        assert len(vals) == 1 and len(per_type[axis]) == len(PKG_TYPES), \
            f"{axis} 配比不齐：{per_type[axis]}"

    # 引用恒为 2 token（这是本版的核心收益）
    for row in train:
        ref = row["target_refs"][0]
        n = len(tok.encode(ref, add_special_tokens=False).ids)
        if n != 2:
            raise RuntimeError(f"引用不是 2 token（{n}）：{ref}")

    # 距离跨度
    means = list(rep["distance_axis_mean_tokens"].values())
    assert means[-1] > means[0] * 2, f"距离跨度太小：{means}"

    # 可复现
    t2, h2 = build_split(tok)
    assert _digest(train) == _digest(t2), "train 重建不一致"
    assert _digest(hold) == _digest(h2), "holdout 重建不一致"

    rep["status"] = "passed"
    rep["reproducible"] = True
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="包引用探针 v9（定稿形态：类型标记 + 整体编号 token）")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)

    if args.self_test:
        _self_test(tok)
        if not args.build:
            return

    if not args.build:
        ap.print_help()
        return

    train, hold = build_split(tok)
    rep = audit(train, hold, tok)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rs in (("train", train), ("holdout", hold)):
        p = DATA_DIR / f"probe_v9_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(rs)} 条 -> {p}")
    rep["file_sha256"] = paths
    rp = DATA_DIR / "probe_v9_report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n报告：{rp}")


if __name__ == "__main__":
    main()
