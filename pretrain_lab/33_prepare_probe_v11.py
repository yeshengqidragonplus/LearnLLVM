# -*- coding: utf-8 -*-
"""包引用探针 v11：修复 v10 现场测试暴露的「跨类型指代」缺口。

================= v10 现场测试暴露的缺口（2026-10-26，32 号脚本）=================
v10 留出集 468 条全 1.00，但 5 组分布外测试里：

  T1 未见问法     10/10 = 1.00   换代词/换句式全对 → 语义绑定是真学到的
  T2 跨类型指代    0/30 = 0.00   ← 缺口
  T3 包表容量     12/12 = 1.00   26/31 项全对 → 容量上限远未到
  T4 投影长度      8/8  = 1.00   1000 token 全对 → RoPE 外推 1.23x 无损
  T5 同表切换      4/12 = 0.33   ← 同 T2 根因

T2/T5 的失败形态：问「那个名字/那段代码」，模型输出 <|PKG_NUM|>#<干扰编号>。
30 条错例里 26 条连编号都错——模型对非 NUM 类型词的「选包」任务完全没学过。

================= 根因：训练覆盖缺口，不是机制失败 =================
v10（31 号）的 semantic_pick **只在 NUM 上监督过「按类型词选包」**：
「那个名字」「那段代码」这些代词从未与选包任务组合过。
type_id 轴学的是「读类型标记」（给引用说类型），不是「按类型词选包」
（给类型词选引用）——两个任务的监督信号不同，前者不能替代后者。

================= v11 的修复 =================
semantic_pick / distance 的目标类型从只有 NUM 扩到**全部 6 类型**：
每条样本随机选一个目标类型，包表里该类型唯一，代词从 TYPE_WORD[目标类型] 取。
其余设计（等距网格、target 轮转、审计断言）与 v10 完全一致。

新增审计：`goal_type_coverage`——6 个目标类型在训练文件里各 ≥N 条，
防止生成器回归成只造 NUM。
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

# ================= 包引用形态（与 v9/v10/tokens.json 一致）=================
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_COUNT = 401
ID_TOKEN_BASE = 151660

REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

UNSEEN_ID = 400
SEEN_IDS = list(range(UNSEEN_ID))

# ================= v10 沿用的参数 =================
MIN_ID_GAP = 16
MIN_TARGET_OCC = 6

# ================= 轴的取值 =================
CHOICE_KS = {"type_id_choice_k2": 2, "type_id_choice_k6": len(PKG_TYPES)}
OPEN_AXES = ("type_id_open", "type_id_unseen")
TYPE_ID_AXES = tuple(CHOICE_KS) + OPEN_AXES
PICK_DISTRACTORS = (2, 5, 10, 20)
DISTANCES = (10, 50, 100, 200, 400, 800)

PER_CELL = 30
PER_CELL_REF = 1200            # v11.3：每格 1200 条（6 类型 × 每组合 ~5 次监督）
PER_CELL_UNSEEN = 10
HOLDOUT_PER_CELL = 6
HOLDOUT_REF_PER_CELL = 30
HOLDOUT_UNSEEN = 2
MIN_ID_TRAIN_OCCURRENCES = 3
SEED = 20261027

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
    if pkg_type not in PKG_TYPES:
        raise ValueError(f"未知包类型 {pkg_type}")
    if not 0 <= n < ID_COUNT:
        raise ValueError(f"编号范围 0..{ID_COUNT - 1}，收到 {n}")
    return f"<|PKG_{pkg_type}|>#{n}"


def id_tokens() -> list[str]:
    return [f"#{n}" for n in range(ID_COUNT)]


def load_tokenizer(directory: Path) -> Tokenizer:
    """注册协议标记 + 全部编号 token，与训练时状态一致。"""
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

    edge_bad = []
    for m in PROTOCOL_TOKENS[:6]:
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
    answer_kind: str
    target_refs: list[str]
    ctx_tokens: int


def _ctx(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def _pick_ids_gap(rng: random.Random, target: int, count: int) -> list[int]:
    """等距网格采样（v10 定稿）：随机相位 + 步长 16，互差 ≥ MIN_ID_GAP。"""
    phase = rng.randrange(MIN_ID_GAP)
    grid = [phase + k * MIN_ID_GAP
            for k in range(len(SEEN_IDS) // MIN_ID_GAP)]
    pool = [g for g in grid if abs(g - target) >= MIN_ID_GAP]
    if len(pool) < count:
        raise RuntimeError(
            f"干扰编号池不足：target={target} pool={len(pool)} count={count}")
    return rng.sample(pool, count)


def build_sample(axis: str, level: str, rng: random.Random,
                 tok: Tokenizer, offset: int = 0,
                 target_id: int | None = None,
                 goal_type: str | None = None) -> Sample:
    """按某个轴的一个取值造一条样本。

    v11 新增 `goal_type`：semantic_pick / distance 的目标类型。
    v10 只在 NUM 上监督；v11 每条随机选一个类型，包表里该类型唯一。
    """
    # ---------------- 类型辨识（四档）----------------
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

        return Sample(prompt, f"{ACT_OUTPUT} <|PKG_{pkg_type}|>", axis,
                      str(level), "type", [ref], _ctx(tok, prompt))

    # ---------------- 语义选取 / 距离（v11：目标类型全覆盖）----------------
    if axis in ("semantic_pick", "distance"):
        if axis == "semantic_pick":
            n_distract = int(level)
            distance = 10
        else:
            n_distract = 5
            distance = int(level)

        if target_id is None:
            raise ValueError("ref 轴必须显式传 target_id")
        if goal_type is None:
            raise ValueError("v11 的 ref 轴必须显式传 goal_type")

        distract_ids = _pick_ids_gap(rng, target_id, n_distract)

        # 包表里**只有一个是 goal_type**，其余都是其他类型 → 只能靠类型前缀定位
        other_types = [t for t in PKG_TYPES if t != goal_type]
        types = [rng.choice(other_types) for _ in range(n_distract + 1)]
        target_pos = rng.randrange(n_distract + 1)
        types[target_pos] = goal_type
        entries = []
        di = 0
        for i, t in enumerate(types):
            if i == target_pos:
                entries.append((t, target_id))
            else:
                entries.append((t, distract_ids[di]))
                di += 1
        target_ref = make_ref(goal_type, target_id)
        package_line = "  ".join(make_ref(t, n) for t, n in entries)

        pronoun = rng.choice(TYPE_WORD[goal_type])
        task = "把指代的对象交给输出"

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

        prompt_with_q = f"{prompt}\n用户说：把{pronoun}交给输出。\n"
        return Sample(prompt_with_q, f"{ACT_OUTPUT} {target_ref}", axis,
                      str(level), "ref", [target_ref],
                      _ctx(tok, prompt_with_q))

    raise ValueError(f"未知轴：{axis}")


def cells() -> list[tuple[str, str]]:
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

    v11：ref 轴的 goal_type 按格内序号轮转（6 类型均匀），
    target 编号按 train/holdout 独立轮转（每编号 6~7 次监督）。
    """
    train: list[dict] = []
    hold: list[dict] = []
    offset = 0
    train_ref = 0
    hold_ref = 0
    for axis, level in cells():
        n = _per_cell(axis)
        n_hold = _holdout_per_cell(axis)
        to_hold_all = axis in HOLDOUT_ONLY_AXES
        for i in range(n):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            is_hold = to_hold_all or i < n_hold
            if axis in REF_AXES:
                # v11.1 修复：goal 与 tid **彻底解耦**。
                # v11 初版 goal=PKG_TYPES[i%6] 与 tid=train_ref%400 同步递增，
                # 产生确定性绑定（NUM↔tid≡0 mod 6 之类），模型把「读编号」
                # 退化成「按类型猜编号」，留出集类型全对、编号崩到 0.00~0.20。
                # 修法：goal 从 rng 抽（rng 由格+序号决定，与 tid 计数器
                # 完全无关），tid 按独立计数器轮转（每编号 6~7 次监督）。
                # 两者来源独立，任意 (goal, tid) 组合均匀出现，绑定消除。
                tid = (hold_ref if is_hold else train_ref) % len(SEEN_IDS)
                if is_hold:
                    hold_ref += 1
                else:
                    train_ref += 1
                goal = rng.choice(PKG_TYPES)
                s = build_sample(axis, level, rng, tok, offset=offset,
                                 target_id=tid, goal_type=goal)
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
            (hold if is_hold else train).append(row)
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

    # 4) prompt/target token 边界
    bad = [r["target"] for r in rows
           if tok.encode(r["text"], add_special_tokens=False).ids
           != (tok.encode(r["prompt"], add_special_tokens=False).ids
               + tok.encode(r["target"], add_special_tokens=False).ids)]
    if bad:
        raise RuntimeError(
            f"prompt/target 拼接处 token 边界不一致（{len(bad)} 条），"
            f"--target-mask 会算错，示例：{bad[0][:40]!r}")

    # 5) 语义轴里目标类型引用必须唯一（v11：目标类型不限于 NUM）
    for row in rows:
        if row["axis"] not in ("semantic_pick", "distance"):
            continue
        mm = REF_RE.fullmatch(row["target_refs"][0])
        if not mm:
            raise RuntimeError(f"引用形态不对：{row['target_refs']}")
        goal = mm.group(1)
        goals = [(m, n) for m, n in REF_RE.findall(_pkg_line_of(row))
                 if m == goal]
        if len(goals) != 1:
            raise RuntimeError(
                f"语义轴的 {goal} 引用必须唯一，实际 {len(goals)}：{goals}")
        if (mm.group(1), mm.group(2)) != goals[0]:
            raise RuntimeError(
                f"target 不是那个唯一的 {goal}：{row['target_refs']}")

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

    # 11) target 编号覆盖（v10 引入）
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

    # 12) 编号间隔断言（v10 引入）
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

    # 13) **v11 新增：目标类型覆盖**——6 个类型作为「按类型词选包」的
    #     答案各 ≥N 条，防止生成器回归成只造 NUM（v10 的缺口）。
    goal_occ: dict[str, int] = {}
    for r in train:
        if r["answer_kind"] != "ref":
            continue
        mm = REF_RE.fullmatch(r["target_refs"][0])
        if mm:
            goal_occ[mm.group(1)] = goal_occ.get(mm.group(1), 0) + 1
    missing_goals = [t for t in PKG_TYPES if t not in goal_occ]
    if missing_goals:
        raise RuntimeError(
            f"目标类型覆盖缺失（v10 缺口回归）：{missing_goals}")
    min_goal = min(goal_occ.values())
    if check_coverage and min_goal < 300:
        raise RuntimeError(
            f"目标类型覆盖不足：{ {t: goal_occ[t] for t in PKG_TYPES} }")

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
        "goal_type_coverage": goal_occ,
        "package_form": f"<|PKG_<TYPE>|>#<ID>，ID 为整体编码 token #0..#{ID_COUNT - 1}",
        "reference_tokens": 2,
        "min_id_gap": MIN_ID_GAP,
        "min_target_occ": MIN_TARGET_OCC,
        "audit": {
            "ref_is_complete": True,
            "goal_ref_unique": True,
            "no_ordinal_in_question": True,
            "target_boundary_ok": True,
            "choice_candidates_consistent": True,
            "unseen_id_never_in_train": True,
            "all_seen_ids_covered": True,
            "id_gap_ok": True,
            "target_id_coverage": True,
            "goal_type_coverage": True,
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

    for row in train:
        ref = row["target_refs"][0]
        n = len(tok.encode(ref, add_special_tokens=False).ids)
        if n != 2:
            raise RuntimeError(f"引用不是 2 token（{n}）：{ref}")

    means = list(rep["distance_axis_mean_tokens"].values())
    assert means[-1] > means[0] * 2, f"距离跨度太小：{means}"

    t2, h2 = build_split(tok)
    assert _digest(train) == _digest(t2), "train 重建不一致"
    assert _digest(hold) == _digest(h2), "holdout 重建不一致"

    rep["status"] = "passed"
    rep["reproducible"] = True
    print(json.dumps(rep, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="包引用探针 v11（semantic_pick 目标类型全覆盖）")
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
        p = DATA_DIR / f"probe_v11_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(rs)} 条 -> {p}")
    rep["file_sha256"] = paths
    rp = DATA_DIR / "probe_v11_report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n报告：{rp}")


if __name__ == "__main__":
    main()
