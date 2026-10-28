# -*- coding: utf-8 -*-
"""包引用探针 v10：修复 v9b 的「相邻编号捷径」。

================= v9b 失败归因（本版要解决的问题）=================
v9b（output_probe_v9b）留出集 semantic_pick 只有 0.46，错例全部是
「类型对、编号差 ±1~2」。根因在 29 号生成器的编号分配：

    picked: list[int] = []
    cur = (offset * 151 + 97) % len(SEEN_IDS)
    while len(picked) < n_distract + 1:
        ...
        cur = (cur + 1) % len(SEEN_IDS)      ← 干扰编号与 target 相邻！

包表永远是 `#297 #298 #299` 这种**连续块**（holdout 实测确认）。
连续块给了记忆捷径：模型记住「这组指纹 → 答案是块尾」即可拿分，
不必学「读 <|PKG_NUM|> 后面的编号」这条 induction 规则。
证据：train_loss 0.0206 / eval_loss 0.3765，18 倍 gap = 纯记忆；
留出集上 induction 不精确，输出的错编号全是包表内干扰项。

================= v10 的三处修复 =================
1. **编号随机不相邻**：干扰编号与 target 编号差 ≥ MIN_ID_GAP（16），
   且干扰项之间也互不相邻。模型只能靠「读类型标记后面的编号」定位。
2. **target 编号池轮转**：target 编号按全局计数器轮转覆盖 #0..#399，
   每个编号作为答案至少被监督 MIN_TARGET_OCC 次（v9b 平均只有 3 次，
   401 选 1 的绑定学不牢）。
3. **样本量加倍**：semantic_pick / distance 每格 150→300 条训练，
   每编号监督次数从 ~3 提到 ~10+。

================= 沿用 v9 的定稿形态（不变）=================
一个引用 = 2 token：`<|PKG_NUM|>#123`（类型标记 + 整体编号 token）。
类型标记同类型共用一行 embedding；编号 #0..#400 整体编码、所有类型
共享同一套编号；展开 = 程序按 (类型, 编号) 查缓存替换，不走大模型。

================= 三条轴（与 v9 相同，都不含序数）=================
  type_id        给一个引用，说出它的类型（答案 1 token，封闭集合）
  semantic_pick  包表里只有一个 NUM，其余异类型；按「那个数值」选出它
  distance       同上 + 无关填充到 800 token，测投影能写多长

================= 两条硬断言（防结论不可归因）=================
1. `num_ref_unique`：语义轴里 NUM 引用必须唯一，否则「那个数值」有歧义。
2. `no_ordinal_in_question`：提问不许出现「第几个」，否则退化成数位置。

================= 本版新增的硬断言 =================
3. `id_gap_ok`：语义/距离轴里，target 编号与每个干扰编号的差 ≥ MIN_ID_GAP，
   干扰编号之间也互差 ≥ MIN_ID_GAP。审计逐条断言，防止生成器回归。
4. `target_id_coverage`：每个已训练编号作为 target 至少出现
   MIN_TARGET_OCC 次（v9b 只保证「出现在包表里」，没保证「作为答案被监督」）。
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

# ================= 包引用形态（与 v9/tokens.json 一致）=================
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_COUNT = 401                 # #0..#400
ID_TOKEN_BASE = 151660         # 14 个协议标记占 151646..151659

REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
BARE_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!#\d)")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

# 负对照：编号 400 训练中绝不出现
UNSEEN_ID = 400
SEEN_IDS = list(range(UNSEEN_ID))       # 0..399

# ================= v10 新参数 =================
MIN_ID_GAP = 16                # 干扰编号与 target、干扰彼此之间编号差下限
MIN_TARGET_OCC = 6             # 每个已训练编号作为 target 的最少监督次数

# ================= 轴的取值（与 v9 相同）=================
CHOICE_KS = {"type_id_choice_k2": 2, "type_id_choice_k6": len(PKG_TYPES)}
OPEN_AXES = ("type_id_open", "type_id_unseen")
TYPE_ID_AXES = tuple(CHOICE_KS) + OPEN_AXES
PICK_DISTRACTORS = (2, 5, 10, 20)
DISTANCES = (10, 50, 100, 200, 400, 800)

# 样本量：含编号的轴加倍（v9b 每格 150，监督密度不足）
PER_CELL = 30                  # type_id 四档：target 只含类型标记
PER_CELL_REF = 300             # semantic_pick / distance：target 含编号
PER_CELL_UNSEEN = 10
HOLDOUT_PER_CELL = 6
HOLDOUT_REF_PER_CELL = 30
HOLDOUT_UNSEEN = 2
MIN_ID_TRAIN_OCCURRENCES = 3   # 出现在包表里的下限（沿用 v9）
SEED = 20261026

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
    table: str                    # [包] 行内容（不含前缀），便于审计
    target: str
    axis: str
    level: str
    answer_kind: str              # "type" | "ref"
    target_refs: list[str]
    ctx_tokens: int


def _ctx(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def _pick_ids_gap(rng: random.Random, target: int, count: int) -> list[int]:
    """随机选 count 个干扰编号：与 target 差 ≥ MIN_ID_GAP，彼此也差 ≥ MIN_ID_GAP。

    v9b 的教训：干扰编号与 target 相邻（cur+1 连续取），包表呈连续块，
    模型靠「块尾/块头」指纹即可拿分，不必学 induction 规则。

    采样方式：等距网格。随机相位 phase ∈ 0..15，格点
    phase, phase+16, ..., phase+384（25 个，最大 ≤ 399）。
    同一格网上任意两点线性差 ≥ 16，天然满足互不相邻；
    剔除与 target 差 < 16 的至多 2 个格点后仍剩 ≥ 23 个，
    count ≤ 20 必然够用。再从中随机取 count 个。
    """
    phase = rng.randrange(MIN_ID_GAP)
    grid = [phase + k * MIN_ID_GAP
            for k in range(len(SEEN_IDS) // MIN_ID_GAP)]   # 25 个格点
    pool = [g for g in grid if abs(g - target) >= MIN_ID_GAP]
    if len(pool) < count:
        raise RuntimeError(
            f"干扰编号池不足：target={target} pool={len(pool)} count={count}")
    return rng.sample(pool, count)


def build_sample(axis: str, level: str, rng: random.Random,
                 tok: Tokenizer, offset: int = 0,
                 target_id: int | None = None) -> Sample:
    """按某个轴的一个取值造一条样本。全部轴都不含序数。

    `offset` 是全局递增计数器（type_id 轴的包表编号按它轮转，只要求
    「出现过」）。`target_id` 是 ref 轴的答案编号，由 build_split 用
    **独立轮转计数器**分配——train 2700 条按 counter%400 完美均匀，
    每个编号恰好 6~7 次作为答案被监督。v9b 只保证「出现在包表里」，
    没保证「作为 target 被监督」，401 选 1 的绑定学不牢。
    """
    # ---------------- 类型辨识（A/B/C/D 四档）----------------
    if axis in TYPE_ID_AXES:
        pkg_type = level
        if axis == "type_id_unseen":
            n = UNSEEN_ID
        else:
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

        return Sample(prompt, ref, f"{ACT_OUTPUT} <|PKG_{pkg_type}|>", axis,
                      str(level), "type", [ref], _ctx(tok, prompt))

    # ---------------- 语义选取 / 距离 ----------------
    if axis in ("semantic_pick", "distance"):
        if axis == "semantic_pick":
            n_distract = int(level)
            distance = 10
        else:
            n_distract = 5
            distance = int(level)

        if target_id is None:
            raise ValueError("ref 轴必须显式传 target_id（由 build_split 轮转分配）")
        # 干扰编号：随机、与 target 差 ≥ MIN_ID_GAP、彼此差 ≥ MIN_ID_GAP
        distract_ids = _pick_ids_gap(rng, target_id, n_distract)

        # 包表里**只有一个是 NUM**，其余都是其他类型 → 只能靠类型前缀定位
        other_types = [t for t in PKG_TYPES if t != "NUM"]
        types = [rng.choice(other_types) for _ in range(n_distract + 1)]
        target_pos = rng.randrange(n_distract + 1)
        types[target_pos] = "NUM"
        entries = []
        di = 0
        for i, t in enumerate(types):
            if i == target_pos:
                entries.append((t, target_id))
            else:
                entries.append((t, distract_ids[di]))
                di += 1
        target_ref = make_ref("NUM", target_id)
        package_line = "  ".join(make_ref(t, n) for t, n in entries)

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
        return Sample(prompt_with_q, package_line,
                      f"{ACT_OUTPUT} {target_ref}", axis,
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
    if axis in HOLDOUT_ONLY_AXES:
        return PER_CELL_UNSEEN
    return PER_CELL_REF if axis in REF_AXES else PER_CELL


def _holdout_per_cell(axis: str) -> int:
    if axis in HOLDOUT_ONLY_AXES:
        return HOLDOUT_UNSEEN
    return HOLDOUT_REF_PER_CELL if axis in REF_AXES else HOLDOUT_PER_CELL


def build_split(tok: Tokenizer) -> tuple[list[dict], list[dict]]:
    """返回 (train_rows, holdout_rows)。

    普通轴按格固定切前 N 条给留出集；`type_id_unseen` 整档只进留出集。
    """
    train: list[dict] = []
    hold: list[dict] = []
    offset = 0
    train_ref = 0        # train 的 target 编号轮转计数器（每编号恰好 6~7 次）
    hold_ref = 0         # holdout 的 target 编号轮转计数器
    for axis, level in cells():
        n = _per_cell(axis)
        n_hold = _holdout_per_cell(axis)
        to_hold_all = axis in HOLDOUT_ONLY_AXES
        for i in range(n):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            is_hold = to_hold_all or i < n_hold
            if axis in REF_AXES:
                # target 编号按所属切分独立轮转：train 2700 条 → 每编号 6~7 次
                tid = (hold_ref if is_hold else train_ref) % len(SEEN_IDS)
                if is_hold:
                    hold_ref += 1
                else:
                    train_ref += 1
                s = build_sample(axis, level, rng, tok, offset=offset,
                                 target_id=tid)
            else:
                s = build_sample(axis, level, rng, tok, offset=offset)
            offset += 1
            row = {
                "kind": "probe",
                "axis": s.axis, "level": s.level,
                "answer_kind": s.answer_kind,
                "prompt": s.prompt, "target": s.target,
                "text": s.prompt + s.target,
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

    # 2) [包] 行每一项必须恰好是一个引用
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

    # 5) 语义轴里 NUM 引用必须唯一
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

    # 6) 提问/代词里不能出现序数
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

    # 8) 每格都有样本；同轴内等量
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

    # 10) 训练文件必须覆盖全部已训练编号，且绝不包含 #400
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

    # 11) **v10 新增：target 编号覆盖**——每个已训练编号作为答案
    #     至少被监督 MIN_TARGET_OCC 次。v9b 只保证「出现在包表里」，
    #     没保证「作为 target 被监督」，401 选 1 的绑定学不牢。
    target_occ: dict[int, int] = {}
    for r in train:
        if r["answer_kind"] != "ref":
            continue
        for _m, n in REF_RE.findall(r["target"]):
            target_occ[int(n)] = target_occ.get(int(n), 0) + 1
    t_never = [n for n in SEEN_IDS if n not in target_occ]
    if t_never:
        raise RuntimeError(
            f"有 {len(t_never)} 个编号从未作为 target 出现（监督不足）："
            f"{t_never[:8]}")
    t_thin = [n for n in SEEN_IDS if target_occ[n] < MIN_TARGET_OCC]
    if check_coverage and t_thin:
        raise RuntimeError(
            f"有 {len(t_thin)} 个编号作为 target 出现不足 {MIN_TARGET_OCC} 次："
            f"{[(n, target_occ[n]) for n in t_thin[:8]]}")

    # 12) **v10 新增：编号间隔断言**——语义/距离轴里 target 与每个干扰
    #     编号差 ≥ MIN_ID_GAP，干扰彼此也差 ≥ MIN_ID_GAP。
    #     v9b 的连续块捷径（#297 #298 #299）就是从这里来的。
    for row in rows:
        if row["axis"] not in ("semantic_pick", "distance"):
            continue
        ids_in_table = [int(n) for _m, n in REF_RE.findall(_pkg_line_of(row))]
        mm = REF_RE.fullmatch(row["target_refs"][0])
        assert mm
        tid = int(mm.group(2))
        for other in ids_in_table:
            if other == tid:
                continue
            if abs(other - tid) < MIN_ID_GAP:
                raise RuntimeError(
                    f"干扰编号与 target 相邻（捷径风险）：target #{tid} "
                    f"包表 {ids_in_table}")
        for a in ids_in_table:
            for b in ids_in_table:
                if a == b:
                    continue
                if abs(a - b) < MIN_ID_GAP:
                    raise RuntimeError(
                        f"包表内编号相邻（捷径风险）：{ids_in_table}")

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
            "min_target_occurrences": min(target_occ[n] for n in SEEN_IDS),
            "max_target_occurrences": max(target_occ[n] for n in SEEN_IDS),
        },
        "package_form": f"<|PKG_<TYPE>|>#<ID>，ID 为整体编码 token #0..#{ID_COUNT - 1}",
        "reference_tokens": 2,
        "min_id_gap": MIN_ID_GAP,
        "min_target_occ": MIN_TARGET_OCC,
        "audit": {
            "ref_is_complete": True,
            "num_ref_unique": True,
            "no_ordinal_in_question": True,
            "target_boundary_ok": True,
            "choice_candidates_consistent": True,
            "unseen_id_never_in_train": True,
            "all_seen_ids_covered": True,
            "id_gap_ok": True,
            "target_id_coverage": True,
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

    # 四档类型辨识配比必须完全一致
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

    # 引用恒为 2 token
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
        description="包引用探针 v10（修复 v9b 相邻编号捷径）")
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
        p = DATA_DIR / f"probe_v10_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(rs)} 条 -> {p}")
    rep["file_sha256"] = paths
    rp = DATA_DIR / "probe_v10_report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n报告：{rp}")


if __name__ == "__main__":
    main()
