# -*- coding: utf-8 -*-
"""编号复制宽度探针：模型能否准确复现投影里出现过的 `#N`。

================= 要回答的问题 =================
用户指出：`#1234` 是 `['#','1','2','3','4']` 5 个普通 token，
这不就是复制墙吗？**这个质疑是对的**，必须实测而不是推断。

已知事实（[`v4-copy-wall-fundamental`] 记忆）：65M 无法逐字符复制
训练分布外的载荷（`286020U` 7 字符被截成 `50053L`）。

但编号与载荷有一处关键差别，可能就是解法：

    载荷 286020U  —— 投影里**没有**，模型必须从参数里「生成」出来
    编号 #1247    —— 投影里**有**（`[包]` 行已写出），模型只需「指回去」

前者是生成（generate），后者是复制（copy from context）。
Transformer 的 attention 天然擅长后者。但这是**假设**，本探针把它测成数。

================= 四档设计（唯一变量是编号位数）=================
  copy_id_d1  编号 #1..#9        2 token
  copy_id_d2  编号 #10..#99      3 token
  copy_id_d3  编号 #100..#999    4 token
  copy_id_d4  编号 #1000..#2000  5 token

每档：投影里给 K 个**同类型**包（所以必须靠编号区分，不能靠类型），
问「#X 那个交出来」，target = `<|OUTPUT|> <|PKG_NUM|>#X`。

================= 同时测「零复制」基线 =================
  no_id_baseline  投影里只有 1 个 NUM，target 只写 `<|PKG_NUM|>` 不带编号

如果 d1..d4 都接近 1.00，说明「复制上文出现过的短串」可行，编号方案成立；
如果随位数下降，说明仍有复制墙，编号必须限制位数（或改成单 token 编号）。

================= 判分必须精确到编号 =================
只看 `#N` 的数字是否完全一致。**不能**只比类型 —— 那样「复制错了」会被
误判成对，正是 v8 犯过的错（当时 key 整体比较其实掩盖了「编号读错」）。
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

PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
# 编号是普通数字 token，不进词表。正则整体抽取 (类型, 数字串)。
REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
BARE_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!\d)")

# 四档：编号位数上限（对应 1..9 / 10..99 / 100..999 / 1000..2000）
ID_DIGITS = (1, 2, 3, 4)
# 每档投影里放几个同类型包（必须 >1，否则编号没意义）
PKGS_IN_PROJECTION = 8
# 样本量：4 位档有 1001 个可能编号，**无法靠记忆，只能靠复制**，所以这档
# 是真正的复制墙测试；1 位档只有 9 个值，样本再少也能记住，参考价值低。
PER_CELL = 80
HOLDOUT_PER_CELL = 20
SEED = 20261024

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]

TYPE_WORD = {
    "NUM": "那个数值", "NAME": "那个名字", "TEXT": "那段文本",
    "CODE": "那段代码", "ARITH": "那个算式", "TASK": "那个任务",
}


def load_tokenizer(directory: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    bad = [t for t in PROTOCOL_TOKENS
           if len(tok.encode(t, add_special_tokens=False).ids) != 1]
    if bad:
        raise SystemExit(f"[错误] 以下标记不是单 token：{bad[:3]}")
    return tok


@dataclass
class Sample:
    prompt: str
    target: str
    axis: str
    level: str
    expect_id: str          # 期望输出的编号（"3"）或 "" 表示无编号
    ctx_tokens: int


def _id_range(digits: int) -> tuple[int, int]:
    """某档的编号取值区间：1 位→1..9，2 位→10..99，3 位→100..999，4 位→1000..2000。"""
    lo = 1 if digits == 1 else 10 ** (digits - 1)
    hi = (9 if digits == 1 else 10 ** digits - 1) if digits < 4 else 2000
    return lo, min(hi, 2000)


def build_sample(axis: str, level: str, rng: random.Random,
                 tok: Tokenizer) -> Sample:
    if axis == "no_id_baseline":
        # 基线：投影里只有 1 个包，target 不带编号（零复制）
        pkg_type = level
        k = f"<|PKG_{pkg_type}|>"
        prompt = (f"[任务] 把指代的对象交给输出\n"
                  f"[包] {k}\n"
                  f"[动作]\n"
                  f"用户说：把{TYPE_WORD[pkg_type]}交给输出。\n")
        target = f"{ACT_OUTPUT} {k}"
        return Sample(prompt, target, axis, str(level), "",
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    if axis == "copy_id":
        digits = int(level)
        lo, hi = _id_range(digits)
        pkg_type = "NUM"
        # 从区间里挑 PKGS_IN_PROJECTION 个互不相同的编号
        span = hi - lo + 1
        n = min(PKGS_IN_PROJECTION, span)
        picked = rng.sample(range(lo, hi + 1), n)
        entries = [f"<|PKG_{pkg_type}|>#{i}" for i in picked]
        package_line = "  ".join(entries)

        # 目标是其中一个（随机）
        target_id = rng.choice(picked)
        prompt = (f"[任务] 按编号交出指定的包\n"
                  f"[包] {package_line}\n"
                  f"[动作]\n"
                  f"用户说：把 {pkg_type}#{target_id} 交给输出。\n")
        target = f"{ACT_OUTPUT} <|PKG_{pkg_type}|>#{target_id}"
        return Sample(prompt, target, axis, str(level), str(target_id),
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    raise ValueError(f"未知轴：{axis}")


def cells() -> list[tuple[str, str]]:
    return ([("no_id_baseline", t) for t in PKG_TYPES]
            + [("copy_id", str(d)) for d in ID_DIGITS])


def build_split(tok: Tokenizer) -> tuple[list[dict], list[dict]]:
    train, hold = [], []
    for axis, level in cells():
        for i in range(PER_CELL):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            s = build_sample(axis, level, rng, tok)
            row = {
                "kind": "probe", "axis": s.axis, "level": s.level,
                "expect_id": s.expect_id,
                "has_id": bool(s.expect_id),
                "prompt": s.prompt, "target": s.target,
                "text": s.prompt + s.target,
                "context_tokens": s.ctx_tokens,
            }
            (hold if i < HOLDOUT_PER_CELL else train).append(row)
    return train, hold


def audit(train: list[dict], hold: list[dict], tok: Tokenizer) -> dict:
    rows = train + hold
    # 1) 边界一致性（--target-mask 前提）
    bad = []
    for r in rows:
        whole = tok.encode(r["text"], add_special_tokens=False).ids
        parts = (tok.encode(r["prompt"], add_special_tokens=False).ids
                 + tok.encode(r["target"], add_special_tokens=False).ids)
        if whole != parts:
            bad.append(r["target"])
    if bad:
        raise RuntimeError(f"拼接边界不一致 {len(bad)} 条，例：{bad[0]!r}")

    # 2) 带编号样本：target 的编号必须真在 [包] 行里
    for r in rows:
        if not r["has_id"]:
            continue
        pkg_line = next(ln for ln in r["prompt"].split("\n")
                        if ln.startswith("[包]"))
        present = {n for _t, n in REF_RE.findall(pkg_line)}
        m = REF_RE.findall(r["target"])
        if len(m) != 1:
            raise RuntimeError(f"带编号样本 target 应恰好 1 个引用：{r['target']!r}")
        if m[0][1] != r["expect_id"]:
            raise RuntimeError(f"target 编号与 expect_id 不符：{m} vs {r['expect_id']}")
        if m[0][1] not in present:
            raise RuntimeError(f"target 引用了投影里不存在的编号 #{m[0][1]}")

    # 3) 无编号基线：target 里不能出现编号
    for r in rows:
        if r["has_id"]:
            continue
        if REF_RE.search(r["target"]):
            raise RuntimeError(f"基线样本 target 不该带编号：{r['target']!r}")
        if not BARE_RE.search(r["target"]):
            raise RuntimeError(f"基线样本 target 缺少裸类型标记：{r['target']!r}")

    # 4) 投影里的编号必须互不相同（否则「按编号选」有歧义）
    for r in rows:
        if r["axis"] != "copy_id":
            continue
        pkg_line = next(ln for ln in r["prompt"].split("\n")
                        if ln.startswith("[包]"))
        ids = [n for _t, n in REF_RE.findall(pkg_line)]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"投影里编号重复：{ids}")

    # 5) 各位数档的编号长度确实递增
    len_by_digit: dict[str, list[int]] = {}
    for r in rows:
        if r["axis"] == "copy_id":
            len_by_digit.setdefault(r["level"], []).append(len(r["expect_id"]))
    for d, ls in sorted(len_by_digit.items()):
        if max(ls) > int(d):
            raise RuntimeError(f"{d} 位档出现了超长编号：{max(ls)}")

    # 6) 各轴内等量
    for name, rs in (("train", train), ("holdout", hold)):
        c: dict[str, int] = {}
        for r in rs:
            c[r["axis"]] = c.get(r["axis"], 0) + 1
        per_axis: dict[str, set[int]] = {}
        for r in rs:
            per_axis.setdefault(r["axis"], set())
        counts: dict[tuple[str, str], int] = {}
        for r in rs:
            counts[(r["axis"], r["level"])] = counts.get((r["axis"], r["level"]), 0) + 1
        axis_levels: dict[str, set[int]] = {}
        for (a, _lv), n in counts.items():
            axis_levels.setdefault(a, set()).add(n)
        bad_axis = {a: v for a, v in axis_levels.items() if len(v) != 1}
        if bad_axis:
            raise RuntimeError(f"{name} 轴内各格不等量：{bad_axis}")

    return {
        "train_rows": len(train), "holdout_rows": len(hold),
        "cells": {f"{a}/{lv}": n for (a, lv), n in sorted(
            {(c[0], c[1]): 1 for c in cells()}.items())},
        "max_context_tokens": max(r["context_tokens"] for r in rows),
        "audit": {
            "target_boundary_ok": True,
            "target_id_present_in_projection": True,
            "baseline_has_no_id": True,
            "projection_ids_unique": True,
            "id_length_matches_level": True,
        },
        "note": "测「复制上文出现过的 #N」是否可行。四档只差编号位数。",
    }


def _digest(rs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rs:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="编号复制宽度探针")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR)
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    train, hold = build_split(tok)
    rep = audit(train, hold, tok)

    if args.self_test:
        t2, h2 = build_split(tok)
        assert _digest(train) == _digest(t2), "train 重建不一致"
        assert _digest(hold) == _digest(h2), "holdout 重建不一致"
        rep["status"] = "passed"
        rep["reproducible"] = True
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        if not args.build:
            return

    if not args.build:
        ap.print_help()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rs in (("train", train), ("holdout", hold)):
        p = DATA_DIR / f"copyid_probe_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(rs)} 条 -> {p}")
    rep["file_sha256"] = paths
    rp = DATA_DIR / "copyid_probe_report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n报告：{rp}")


if __name__ == "__main__":
    main()
