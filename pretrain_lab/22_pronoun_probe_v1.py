# -*- coding: utf-8 -*-
"""指示代词 / 包引用探针：测协议真正需要的三件事。

================= 为什么要重写这一版 =================
上一版（v1~v6）测的三条轴里，「干扰项」和「类型混淆」用的是**序数代词**
（「第几个和它后面那个」），要求模型在 N 个同质 key 里数到第 N 项再取两个。
于是得出「包表超过 12 项就挑不准」这个结论——**但这是无效需求**：

    协议里模型从来不需要「数第几个包」。它只需要知道
    `<|PKG_NUM|,#A>` 是一个数字包，值本身在缓存里，
    由程序按 (类型, 序号) 解包后交给专家（token filter）。

所以「包表容量上限」测的是我自己造的题目，不是协议的能力边界。本版按
**协议真正需要的三件事**重新设计：

  轴 1  type_id       给一个 key，说它是什么类型（类型是否真的被理解）
  轴 2  semantic_pick 多个**其他类型**的干扰项里，按语义「那个数值」选出唯一的 NUM 包
  轴 3  distance      语义指代 + 无关填充加长，测投影能写多长

三条轴都不要求数位置。轴 2 的干扰项是**不同类型**，靠类型区分而非位置计数。

包引用一律是原子整体 key：`<|PKG_<TYPE>|,#<ID>>`，注册后 1 token。
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

# ================= 包 key：原子整体 =================
# 【核心原则】一个包 = 一个不可分的整体 key：<|PKG_<TYPE>|,#<ID>>。
# 模型不需要理解包内容，只需知道「key 不同 = 意思不同」；
# 拆包由程序按 (类型, 序号) 从缓存还原载荷，交给专家模型。
#
# 因此每个 key 必须注册成**独立的单 token**。实测（tokenizer.json）：
#   <|PKG_NUM|,#B>          -> 注册后 1 token  ← 正确
#   <|PKG_NUM|>#B           -> 3 token，序号游离在外  ← 错误
#   <|PKG_NUM|>#B 数值(#B)  -> 8 token，编号重复两次  ← 错误
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 引用 key 的整体匹配模式：尖括号内 类型 + 逗号 + #序号。
KEY_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|,#[A-Z]>")

# 14 个协议标记占 151646..151659，所以 key 从 151660 起。
# 这个常量必须与 tokens.json 的 _package_keys.id_base 完全一致。
KEY_ID_BASE = 151660


def make_key(pkg_type: str, index: int) -> str:
    """造一个包 key（原子整体）。index 是 0-based 序号，0->A、1->B ..."""
    if pkg_type not in PKG_TYPES:
        raise ValueError(f"未知包类型 {pkg_type}")
    if not 0 <= index < len(ID_CHARS):
        raise ValueError(f"包表最多 {len(ID_CHARS)} 项，收到 index={index}")
    return f"<|PKG_{pkg_type}|,#{ID_CHARS[index]}>"


def all_keys() -> list[str]:
    """按 tokens.json 的布局顺序枚举全部 key：类型在外层、序号在内层。"""
    return [make_key(t, i) for t in PKG_TYPES for i in range(len(ID_CHARS))]


# ================= 轴的取值 =================
# 轴 1：6 种类型各一格
TYPE_ID_LEVELS = PKG_TYPES
# 轴 2：干扰项个数（**全部是其他类型**，靠类型区分而非位置计数）
PICK_DISTRACTORS = (2, 5, 10, 20)
# 轴 3：投影目标 token 数（无关填充）
DISTANCES = (10, 50, 100, 200, 400, 800)

SEED = 20261022

# 协议标记：训练时由 03 的 --protocol-tokens 注册（见 tokens.json）。
# 数据脚本必须**同样注册**后再统计 token，否则 <|OUTPUT|> 会被切成 5 个 token，
# 量出来的投影长度和边界检查全是假的（本文件踩过这个坑）。
PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]

# 语义指代的措辞：按类型说"那个数值/那个名字/那段文本"，
# **不含任何序数**，模型只能靠 key 的类型前缀来找，不能靠数位置。
TYPE_WORD = {
    "NUM": ["那个数值", "刚才那个数值", "用户给的那个数值"],
    "NAME": ["那个名字", "刚才那个名字", "用户给的那个名字"],
    "TEXT": ["那段文本", "刚才那段文本"],
    "CODE": ["那段代码", "刚才那段代码"],
    "ARITH": ["那个算式", "刚才那个算式"],
    "TASK": ["那个任务", "刚才那个任务"],
}


def load_tokenizer(directory: Path) -> Tokenizer:
    """加载 tokenizer，注册协议标记 + 全部包 key，与训练时的状态一致。

    注册顺序必须与 tokens.json / 03_train_pretrain.py 完全一致，
    否则 ID 错位：一个医生写的 key，另一个医生会读成别的包。
    """
    tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    for t in PROTOCOL_TOKENS:
        ids = tok.encode(t, add_special_tokens=False).ids
        if len(ids) != 1:
            raise SystemExit(f"[错误] {t} 没有成为单 token：{ids}")

    # ---- 包 key：类型 × 序号，共 156 个，每个必须是单 token ----
    keys = all_keys()
    tok.add_special_tokens([
        AddedToken(k, special=True, normalized=False) for k in keys
    ])
    mis_single, mis_id = [], []
    for offset, k in enumerate(keys):
        ids = tok.encode(k, add_special_tokens=False).ids
        if len(ids) != 1:
            mis_single.append(f"{k}->{ids}")
            continue
        expect = KEY_ID_BASE + offset
        if ids[0] != expect:
            mis_id.append(f"{k}: got {ids[0]} expect {expect}")
    if mis_single:
        raise SystemExit("[错误] 以下 key 没有成为单 token（碎片化）："
                         + "；".join(mis_single[:5]))
    if mis_id:
        raise SystemExit("[错误] key 的 ID 与 tokens.json 布局不一致（跨医生交接会失效）："
                         + "；".join(mis_id[:5]))
    return tok


# ================= 造数据 =================


@dataclass
class ProbeSample:
    """一条探针样本：投影式上下文 + 指代问句 + 期望的引用序列。"""

    prompt: str
    target: str
    axis: str          # "type_id" | "semantic_pick" | "distance"
    level: str         # 轴取值（如 "NUM" / "20" / "800"）
    target_refs: list[str]
    context_tokens: int


def _render_pkg_line(entries: list[tuple[str, int]]) -> str:
    """把 (类型, 0-based序号) 列表渲染成 [包] 行。每项恰好一个原子 key。"""
    return "  ".join(make_key(t, i) for t, i in entries)


def build_sample(axis: str, level: str, rng: random.Random,
                 tokenizer: Tokenizer) -> ProbeSample:
    """按某个轴的一个取值造一条样本。

    三条轴共同点：**代词/提问都不含序数**，模型不需要数位置。
    """
    if axis == "type_id":
        # ---- 轴 1：给一个 key，说它是什么类型 ----
        pkg_type = level
        # 把目标 key 的序号随机化，避免模型把「类型」与「固定序号」绑定
        idx = rng.randrange(0, len(ID_CHARS))
        key = make_key(pkg_type, idx)
        prompt = (f"[任务] 判断包的类型\n"
                  f"[包] {key}\n"
                  f"[动作]\n"
                  f"用户说：{key} 是什么类型的包？\n")
        # 答案就是类型标记本身（1 token，封闭集合，可自动判分）
        target = f"{ACT_OUTPUT} <|PKG_{pkg_type}|>"
        return ProbeSample(prompt=prompt, target=target, axis=axis,
                           level=str(level), target_refs=[key],
                           context_tokens=len(tokenizer.encode(
                               prompt, add_special_tokens=False).ids))

    if axis in ("semantic_pick", "distance"):
        # ---- 轴 2/3：按语义选出唯一的 NUM 包 ----
        # 包表里**只有一个是 NUM**，其余都是其他类型 → 模型只能靠类型前缀定位，
        # 不能靠位置计数（这是与上一版最本质的区别）。
        if axis == "semantic_pick":
            n_distract = int(level)
            distance = 10          # 不加填充
        else:
            n_distract = 5         # 固定少量干扰
            distance = int(level)

        other_types = [t for t in PKG_TYPES if t != "NUM"]
        entries: list[tuple[str, int]] = []
        # 序号按位置分配（A,B,C...），类型乱序，但**只有一个 NUM**
        target_pos = rng.randrange(0, n_distract + 1)
        seq = 0
        target_key = ""
        for pos in range(n_distract + 1):
            if pos == target_pos:
                entries.append(("NUM", seq))
                target_key = make_key("NUM", seq)
            else:
                entries.append((rng.choice(other_types), seq))
            seq += 1
        package_line = _render_pkg_line(entries)

        # 语义指代：按类型说"那个数值"，不含序数
        pronoun = rng.choice(TYPE_WORD["NUM"])
        task = "把指代的对象交给输出"

        # 距离控制：填充塞进 [任务] 行
        filler_units = ["注意第%d项已经确认。" % i for i in range(1, 1600)]
        filler_parts: list[str] = []
        while True:
            prompt = (f"[任务] {task} {' '.join(filler_parts)}".rstrip() + "\n"
                      f"[包] {package_line}\n"
                      f"[动作]")
            n_tok = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if n_tok >= distance or len(filler_parts) >= len(filler_units):
                break
            filler_parts.append(filler_units[len(filler_parts)])

        # 结尾必须换行（--target-mask 的 token 边界依赖它）
        prompt_with_pronoun = f"{prompt}\n用户说：把{pronoun}交给输出。\n"
        target = f"{ACT_OUTPUT} {target_key}"
        return ProbeSample(prompt=prompt_with_pronoun, target=target, axis=axis,
                           level=str(level), target_refs=[target_key],
                           context_tokens=len(tokenizer.encode(
                               prompt_with_pronoun, add_special_tokens=False).ids))

    raise ValueError(f"未知轴：{axis}")


def _cells() -> list[tuple[str, str]]:
    return ([("type_id", t) for t in TYPE_ID_LEVELS]
            + [("semantic_pick", str(n)) for n in PICK_DISTRACTORS]
            + [("distance", str(d)) for d in DISTANCES])


def build_dataset(tokenizer: Tokenizer, per_cell: int = 8) -> list[dict]:
    """每格（轴 × 取值）造 per_cell 条样本。"""
    rows: list[dict] = []
    for axis, level in _cells():
        for i in range(per_cell):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            s = build_sample(axis, level, rng, tokenizer)
            rows.append({
                "kind": "probe",
                "axis": s.axis,
                "level": s.level,
                "prompt": s.prompt,
                "target": s.target,
                "text": s.prompt + s.target,   # 必须严格等于 prompt+target
                "target_refs": s.target_refs,
                "context_tokens": s.context_tokens,
            })
    return rows


# ================= 审计 =================


def audit(rows: list[dict], tokenizer: Tokenizer) -> dict:
    """检查数据可用性与隔离性。任何一条不合法都直接抛错。"""
    # 1) 每行必须有 prompt/target/text
    for row in rows:
        for field in ("prompt", "target", "text"):
            if not row.get(field):
                raise RuntimeError(f"缺少字段 {field}：{row}")
        if not row["target"].startswith(ACT_OUTPUT):
            raise RuntimeError(f"target 动作不对：{row['target']}")

    # 2) 包表里每一项必须**恰好是一个原子 key**，不允许夹带说明文字。
    #    这条守的是「包是原子整体」：一旦有人把 `数值(#A)` 加回去，
    #    测出来的东西就变成字符噪声。
    for row in rows:
        pkg_line = next(ln for ln in row["prompt"].split("\n")
                        if ln.startswith("[包]"))
        items = [s for s in pkg_line[len("[包]"):].split("  ") if s.strip()]
        if not items:
            raise RuntimeError(f"[包] 行为空：{row['prompt'][:60]!r}")
        for seg in items:
            if not KEY_RE.fullmatch(seg.strip()):
                raise RuntimeError(
                    f"包表条目不是一个整体 key（不得附加说明文字）：{seg!r}")

    # 3) target 里的引用必须原样出现在 [包] 行里
    for row in rows:
        pkg_line = next(ln for ln in row["prompt"].split("\n")
                        if ln.startswith("[包]"))
        present = set(KEY_RE.findall(pkg_line))
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
        pkg_line = next(ln for ln in row["prompt"].split("\n")
                        if ln.startswith("[包]"))
        nums = [k for k in KEY_RE.findall(pkg_line) if k.startswith("<|PKG_NUM|")]
        if len(nums) != 1:
            raise RuntimeError(
                f"语义轴的 NUM 包必须唯一，实际 {len(nums)} 个：{nums}")
        if nums[0] != row["target_refs"][0]:
            raise RuntimeError(
                f"target 不是那个唯一的 NUM 包：target={row['target_refs']} nums={nums}")

    # 6) **关键：语义轴里不能出现序数**
    #    上一版用「第几个」导致测的是数位置。这里断言代词措辞不含序数，
    #    否则整份数据又退回到无效需求。
    ordinal_re = re.compile(r"第[一二三四五六七八九十百零\d]+[个项]")
    for row in rows:
        if row["axis"] not in ("semantic_pick", "distance"):
            continue
        tail = row["prompt"].split("[动作]")[-1]
        if ordinal_re.search(tail):
            raise RuntimeError(f"语义轴的代词出现了序数（会退化成效数位置）：{tail!r}")

    # 7) 每个轴每个取值都有样本，且数量一致
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["axis"], row["level"])
        counts[key] = counts.get(key, 0) + 1
    missing = [k for k in _cells() if k not in counts]
    if missing:
        raise RuntimeError(f"缺少格子：{missing}")
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"各格样本数不一致：{counts}")

    # 8) 距离轴：实际 token 数必须随 level 单调不减
    dist_by_level: dict[str, list[int]] = {}
    for row in rows:
        if row["axis"] == "distance":
            dist_by_level.setdefault(row["level"], []).append(row["context_tokens"])
    ordered = [int(d) for d in DISTANCES if str(d) in dist_by_level]
    means = [sum(dist_by_level[str(d)]) / len(dist_by_level[str(d)]) for d in ordered]
    for a, b in zip(means, means[1:]):
        if b < a - 5:
            raise RuntimeError(f"距离轴下降过多：{dict(zip(ordered, means))}")

    return {
        "sample_count": len(rows),
        "cells": {f"{a}/{lv}": n for (a, lv), n in sorted(counts.items())},
        "total_tokens": sum(r["context_tokens"] for r in rows),
        "max_context_tokens": max(r["context_tokens"] for r in rows),
        "distance_axis_mean_tokens": {str(d): round(m, 1)
                                      for d, m in zip(ordered, means)},
        "target_boundary_ok": True,
        "refs_all_present": True,
        "num_key_unique": True,
        "no_ordinal_in_pronoun": True,
        "note": "三轴探针（重设计版）：类型辨识 / 语义选取 / 距离。"
                "代词不含序数，不要求模型数位置。"
    }


# ================= 自测 =================


def _self_test(tokenizer: Tokenizer) -> None:
    rows = build_dataset(tokenizer, per_cell=3)
    report = audit(rows, tokenizer)

    # 类型轴：答案必须是 6 种类型标记之一，且与包表里那个 key 的类型一致
    for row in rows:
        if row["axis"] != "type_id":
            continue
        answer = row["target"].split()[1]
        pkg_key = row["target_refs"][0]
        expect = "<|PKG_" + pkg_key.split("_", 1)[1].split("|")[0] + "|>"
        assert answer == expect, f"类型答案不对：{answer} vs {expect}（key={pkg_key}）"

    # 语义轴：干扰项必须都是**非 NUM**类型，否则「那个数值」有歧义
    for row in rows:
        if row["axis"] != "semantic_pick":
            continue
        pkg_line = next(ln for ln in row["prompt"].split("\n")
                        if ln.startswith("[包]"))
        keys = KEY_RE.findall(pkg_line)
        nums = [k for k in keys if k.startswith("<|PKG_NUM|")]
        assert len(nums) == 1, f"NUM 包不唯一：{nums}"
        others = [k for k in keys if not k.startswith("<|PKG_NUM|")]
        assert all(not k.startswith("<|PKG_NUM|") for k in others)

    # 距离轴：最长档必须明显长于最短档
    means = list(report["distance_axis_mean_tokens"].values())
    assert means[-1] > means[0] * 2, f"距离跨度太小：{means}"

    # 可复现：同 seed 重建逐字节一致
    again = build_dataset(tokenizer, per_cell=3)
    def _digest(rs):
        h = hashlib.sha256()
        for r in rs:
            h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
        return h.hexdigest()
    assert _digest(rows) == _digest(again), "同 seed 重建不一致"

    report["status"] = "passed"
    report["reproducible"] = True
    print(json.dumps(report, ensure_ascii=False, indent=2))


# ================= 主流程 =================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指示代词/包引用三轴探针（重设计版：不要求数位置）")
    parser.add_argument("--build", action="store_true",
                        help="生成数据到 data/pronoun_probe_v1.jsonl")
    parser.add_argument("--per-cell", type=int, default=8,
                        help="每格样本数（默认 8；正式实验用 32）")
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

    rows = build_dataset(tokenizer, per_cell=args.per_cell)
    report = audit(rows, tokenizer)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / "pronoun_probe_v1.jsonl"
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report["file_sha256"] = {out.name: hashlib.sha256(out.read_bytes()).hexdigest()}

    rep_path = DATA_DIR / "pronoun_probe_v1_report.json"
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n数据：{out}\n报告：{rep_path}")


if __name__ == "__main__":
    main()
