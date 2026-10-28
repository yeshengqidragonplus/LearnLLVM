# -*- coding: utf-8 -*-
"""编号复制探针 v2：定长编号，验证「边界不可判定」是不是真因。

================= v1 的发现（`26_prepare_copyid_probe.py` 的结果）=================
变长数字编号的复制准确率**非单调**，而且错例形态指向同一个原因：

| 编号位数 | ref_ok | 主要错例形态 |
|---|---|---|
| 1 位（#1..#9） | 1.00 | — |
| **2 位（#10..#99）** | **0.30** | **多吐 1 位 12/14**：`#94`→`#949`、`#10`→`#101` |
| 3 位（#100..#999） | 0.80 | 同位错值 2 |
| 4 位（#1000..#2000） | 0.65 | **顺序错 6**：数字全对、顺序不对 |

关键：**2 位档最差，3/4 位反而更好**。如果瓶颈是"数字串太长记不住"，
应该位数越多越差。实际不是。真正的原因是：

    变长编号没有结束标志，模型不知道「几位之后编号结束」。
    2 位档取值 10~99，长度可变 → 边界最不确定 → 最容易多吞一位。
    3/4 位档长度固定 → 模型学到"就抄 3/4 个" → 反而更稳。

而 4 位档的「顺序错 6 条」是另一回事：`#1331` 抄成 `#1313`，
数字记住了但顺序乱。这更像位置编码/注意力的问题，不是边界问题。

================= v2 的设计：只改一个变量 = 定长 =================
编号一律补零到固定宽度 `#0094`、`#1331`。与 v1 的唯一差别是定长。

    定长后每个编号 token 数恒定 → 模型只需"抄 N 个数字"，不需判断何时停。

同时保留 v1 的几个档做对照：
  baseline   投影只有 1 个包，target 不带编号（零复制下限）
  fixed      定长编号（本版主测）
  padded_vs_unpadded  同一编号两种写法，检验是否只是 token 数的问题

================= 判分必须严到编号（沿用 v1 教训）=================
只看「类型对了」会把「编号抄错」掩盖成对。本脚本的 holdout 行同时给出
`expect_id`，评估脚本必须比对数字是否**完全一致**。
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
REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")

# 编号宽度：4 位覆盖 0001..2000，且与 v1 的 4 位档可直接对比
ID_WIDTH = 4
ID_MAX = 2000
# 每档投影里放几个同类型包（必须 >1，否则编号无意义）
PKGS_IN_PROJECTION = 8
PER_CELL = 80
HOLDOUT_PER_CELL = 20
SEED = 20261025

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


def fmt_id(n: int) -> str:
    """定长编号：补零到 ID_WIDTH 位。"""
    return f"{n:0{ID_WIDTH}d}"


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
    expect_id: str
    ctx_tokens: int


def build_sample(axis: str, level: str, rng: random.Random,
                 tok: Tokenizer) -> Sample:
    if axis == "baseline":
        # 零复制下限：投影只 1 个包，target 不带编号
        pkg_type = level
        k = f"<|PKG_{pkg_type}|>"
        prompt = (f"[任务] 把指代的对象交给输出\n"
                  f"[包] {k}\n"
                  f"[动作]\n"
                  f"用户说：把{TYPE_WORD[pkg_type]}交给输出。\n")
        return Sample(prompt, f"{ACT_OUTPUT} {k}", axis, str(level), "",
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    if axis == "fixed_id":
        # 定长编号：所有编号都是 4 位，长度恒定
        picked = rng.sample(range(1, ID_MAX + 1), PKGS_IN_PROJECTION)
        entries = [f"<|PKG_NUM|>#{fmt_id(i)}" for i in picked]
        target_id = rng.choice(picked)
        prompt = (f"[任务] 按编号交出指定的包\n"
                  f"[包] {'  '.join(entries)}\n"
                  f"[动作]\n"
                  f"用户说：把 NUM#{fmt_id(target_id)} 交给输出。\n")
        return Sample(prompt, f"{ACT_OUTPUT} <|PKG_NUM|>#{fmt_id(target_id)}",
                      axis, str(level), fmt_id(target_id),
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    if axis == "padded_vs_unpadded":
        # 同一组编号，一半补零一半不补：检验是否只是 token 数问题
        n = rng.randint(1, ID_MAX)
        style = level            # "padded" | "raw"
        shown = fmt_id(n) if style == "padded" else str(n)
        prompt = (f"[任务] 复述包编号\n"
                  f"[包] <|PKG_NUM|>#{shown}\n"
                  f"[动作]\n"
                  f"用户说：把 NUM#{shown} 交给输出。\n")
        return Sample(prompt, f"{ACT_OUTPUT} <|PKG_NUM|>#{shown}",
                      axis, str(level), shown,
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    if axis == "dup_digit":
        # 诊断档：编号里含**重复数字**（1122 / 1111 / 1231 …）。
        # 目的：把「记不住某个数字」与「跨位置复制错位」分开。
        #   若含重复数字的样本正确率明显低于不重复的 → 是位置绑定问题
        #   若相同 → 是纯粹的「4 位数字串整体抄不准」
        pattern = level          # "repeat" | "distinct"
        while True:
            n = rng.randint(1, ID_MAX)
            s = fmt_id(n)
            has_repeat = len(set(s)) < len(s)
            if (pattern == "repeat") == has_repeat:
                break
        shown = s
        prompt = (f"[任务] 复述包编号\n"
                  f"[包] <|PKG_NUM|>#{shown}\n"
                  f"[动作]\n"
                  f"用户说：把 NUM#{shown} 交给输出。\n")
        return Sample(prompt, f"{ACT_OUTPUT} <|PKG_NUM|>#{shown}",
                      axis, str(level), shown,
                      len(tok.encode(prompt, add_special_tokens=False).ids))

    raise ValueError(f"未知轴：{axis}")


def cells() -> list[tuple[str, str]]:
    return ([("baseline", t) for t in PKG_TYPES]
            + [("fixed_id", "4")]
            + [("padded_vs_unpadded", s) for s in ("padded", "raw")]
            + [("dup_digit", s) for s in ("repeat", "distinct")])


def build_split(tok: Tokenizer) -> tuple[list[dict], list[dict]]:
    train, hold = [], []
    for axis, level in cells():
        for i in range(PER_CELL):
            rng = random.Random(f"{SEED}-{axis}-{level}-{i}")
            s = build_sample(axis, level, rng, tok)
            row = {
                "kind": "probe", "axis": s.axis, "level": s.level,
                "expect_id": s.expect_id, "has_id": bool(s.expect_id),
                "prompt": s.prompt, "target": s.target,
                "text": s.prompt + s.target,
                "context_tokens": s.ctx_tokens,
            }
            (hold if i < HOLDOUT_PER_CELL else train).append(row)
    return train, hold


def audit(train: list[dict], hold: list[dict], tok: Tokenizer) -> dict:
    rows = train + hold

    # 1) 拼接边界（--target-mask 前提）
    bad = [r["target"] for r in rows
           if tok.encode(r["text"], add_special_tokens=False).ids
           != (tok.encode(r["prompt"], add_special_tokens=False).ids
               + tok.encode(r["target"], add_special_tokens=False).ids)]
    if bad:
        raise RuntimeError(f"拼接边界不一致 {len(bad)} 条，例：{bad[0]!r}")

    # 2) 带编号样本：target 编号必须在 [包] 行里，且与 expect_id 一致
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
            raise RuntimeError(f"target 引用了投影里不存在的编号：#{m[0][1]}")

    # 3) **关键：定长档的所有编号必须等长**（本版唯一变量就是这一点）
    for r in rows:
        if r["axis"] != "fixed_id":
            continue
        pkg_line = next(ln for ln in r["prompt"].split("\n")
                        if ln.startswith("[包]"))
        ids = [n for _t, n in REF_RE.findall(pkg_line)]
        if len({len(x) for x in ids}) != 1:
            raise RuntimeError(f"定长档出现不等长编号：{ids}")
        if len(ids[0]) != ID_WIDTH:
            raise RuntimeError(f"定长档编号宽度应为 {ID_WIDTH}，实际 {len(ids[0])}")

    # 4) 对照档：padded 恒 4 位，raw 允许变长
    for r in rows:
        if r["axis"] != "padded_vs_unpadded":
            continue
        if r["level"] == "padded" and len(r["expect_id"]) != ID_WIDTH:
            raise RuntimeError(f"padded 档编号不是 {ID_WIDTH} 位：{r['expect_id']}")
        if r["level"] == "raw" and r["expect_id"].startswith("0"):
            raise RuntimeError(f"raw 档不应有前导零：{r['expect_id']}")

    # 5) 投影内编号互不相同
    for r in rows:
        if not r["has_id"]:
            continue
        pkg_line = next(ln for ln in r["prompt"].split("\n")
                        if ln.startswith("[包]"))
        ids = [n for _t, n in REF_RE.findall(pkg_line)]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"投影里编号重复：{ids}")

    # 6) 轴内各格等量
    for name, rs in (("train", train), ("holdout", hold)):
        cnt: dict[tuple[str, str], int] = {}
        for r in rs:
            k = (r["axis"], r["level"])
            cnt[k] = cnt.get(k, 0) + 1
        per_axis: dict[str, set[int]] = {}
        for (a, _lv), n in cnt.items():
            per_axis.setdefault(a, set()).add(n)
        bad_axis = {a: v for a, v in per_axis.items() if len(v) != 1}
        if bad_axis:
            raise RuntimeError(f"{name} 轴内各格不等量：{bad_axis}")

    return {
        "train_rows": len(train), "holdout_rows": len(hold),
        "id_width": ID_WIDTH, "id_max": ID_MAX,
        "pkgs_in_projection": PKGS_IN_PROJECTION,
        "max_context_tokens": max(r["context_tokens"] for r in rows),
        "audit": {
            "target_boundary_ok": True,
            "fixed_id_all_equal_length": True,
            "target_id_present_in_projection": True,
            "projection_ids_unique": True,
        },
        "note": "v2：编号定长 4 位。与 v1 的唯一差别是定长，用于验证"
                "「边界不可判定」是否为复制失败的真正原因。",
    }


def _digest(rs: list[dict]) -> str:
    h = hashlib.sha256()
    for r in rs:
        h.update(json.dumps(r, ensure_ascii=False, sort_keys=True).encode())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="编号复制探针 v2（定长编号）")
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
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        if not args.build:
            return

    if not args.build:
        ap.print_help()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, rs in (("train", train), ("holdout", hold)):
        p = DATA_DIR / f"copyid_fixed_{name}.jsonl"
        with p.open("w", encoding="utf-8", newline="\n") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(f"{name}: {len(rs)} 条 -> {p}")
    rep["file_sha256"] = paths
    rp = DATA_DIR / "copyid_fixed_report.json"
    rp.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                  encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"\n报告：{rp}")


if __name__ == "__main__":
    main()
