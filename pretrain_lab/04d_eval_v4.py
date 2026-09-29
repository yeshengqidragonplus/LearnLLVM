# -*- coding: utf-8 -*-
"""04d_eval_v4.py — v4 评估：数位数能否破除"位数不可知"的墙
核心判据（v3 是 0%）：
  · [C] 数位数：1~4 位（训练内）必须 ~100%；5~8 位（外推）看泛化
  · [W] 宽度扫描：5 位后继/加法 ≥60% → 破墙成功
  · [E6] 序 ⊥（v3 全错）→ 方向修正是成功
  · 其余继承 04c（挖洞/组合泛化/读法层）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random
import re

parser = argparse.ArgumentParser(description="v4 评估（数位数 + 破墙验证）")
parser.add_argument("--model", required=True)
parser.add_argument("--n", type=int, default=12)
parser.add_argument("--hole-digit", type=int, default=7)
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

rng = random.Random(args.seed)
HO = args.hole_digit

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("tokenizer")
model = Qwen2ForCausalLM.from_pretrained(args.model).to(torch.bfloat16).to(device)
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.eval()
eos = tok.eos_token_id

MAX_NEW = 400


def gen(prompt, max_new=MAX_NEW):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


# ================= [S0] 前缀一致性自检 =================
SELF_CHECKS = [
    ("问: 4483 是几位数?\n理解: 数列 L := 4,4,8,3\n计数: i=1(4), i=2(4), i=3(8), i=4(3)\n结束: 数列已取完\n答: 4",
     "问: 4483 是几位数?\n"),
    ("问: 4483+579 = ?\n理解: 算式 F, F := a+b, a := 4,4,8,3, b := 5,7,9\n"
     "算法: i 从 1 到 4 循环：s_i := a_i+b_i+c; d_i := s_i mod 10; c := s_i div 10（c 初值 0，a_i 从右往左取）\n"
     "i=1: a_1=3, b_1=9, c=0, s_1=3+9+0=12, d_1=2, c:=1\n"
     "i=2: a_2=8, b_2=7, c=1, s_2=8+7+1=16, d_2=6, c:=1\n"
     "i=3: a_3=4, b_3=5, c=1, s_3=4+5+1=10, d_3=0, c:=1\n"
     "i=4: a_4=4, b_4=0, c=1, s_4=4+0+1=5, d_4=5, c:=0\n"
     "结束: c=0，无补位\n合成: 5,0,6,2\n答: 5062", "问: 4483+579 = ?\n"),
]
print("[S0] prompt 前缀一致性自检")
for full, prefix in SELF_CHECKS:
    fids, pids = tok.encode(full), tok.encode(prefix)
    if fids[:len(pids)] != pids:
        print(f"  FAIL: {prefix!r}")
        sys.exit(1)
print("  OK（2/2）\n")

results = []


def run(dim, level, label, cases, check, show=3):
    ok = 0
    fails = []
    for prompt, expect in cases:
        out = gen(prompt)
        if check(out, expect):
            ok += 1
        else:
            fails.append((prompt, out, expect))
    pct = 100.0 * ok / len(cases) if cases else 0.0
    results.append((dim, level, label, ok, len(cases)))
    print(f"  [{level}] {label:<38} {ok}/{len(cases)} = {pct:5.1f}%")
    for p, o, e in fails[:show]:
        print(f"       FAIL {p[:38]!r}")
        print(f"            -> {o[:95]!r}")
    return pct


def pick(lo, hi, n, excl_hole=True):
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if excl_hole and v % 10 == HO:
            continue
        out.append(v)
    return out


def pick_hole(lo, hi, n):
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if v % 10 == HO:
            out.append(v)
    return out


def num_check(field):
    """从 `field: X` 提取数字 X 并与期望比对"""
    def chk(out, expect):
        m = re.search(rf"{field}: (\d+)", out)
        return bool(m) and int(m.group(1)) == expect
    return chk


def ans_check(out, expect):
    if "合成" not in out:
        return False
    return num_check("答")(out, expect)


def order_check(out, expect):
    m = re.search(r"答: (\S+)", out)
    return bool(m) and m.group(1) == ("⊤" if expect else "⊥")


def read_check(out, expect):
    return str(expect) in out


# ================= [C] 数位数（v4 核心） =================
print("[C] 数位数（v4 核心新增）")
for w, lo, hi in [(1, 1, 9), (2, 10, 99), (3, 100, 999), (4, 1000, 9999)]:
    run("C", f"{w}位", f"数位数（训练内 {w} 位）",
        [(f"问: {a} 是几位数?\n", w) for a in pick(lo, hi, args.n)], num_check("答"))
# 外推（5~8 位，训练只到 4 位）
for w, lo, hi in [(5, 10000, 99999), (6, 100000, 999999),
                  (7, 1000000, 9999999), (8, 10000000, 99999999)]:
    run("C", f"{w}位", f"数位数（外推 {w} 位）",
        [(f"问: {a} 是几位数?\n", w) for a in pick(lo, hi, max(3, args.n // 2))],
        num_check("答"))

# ================= [W] 宽度扫描（破墙判据） =================
print("\n[W] 宽度扫描（v3 是 0%）")
for w, lo, hi in [(3, 100, 999), (4, 1000, 9999), (5, 10000, 99999),
                  (6, 100000, 999999)]:
    run("W", f"{w}位", f"后继 s({w}位数)",
        [(f"问: s({a}) = ?\n", a + 1) for a in pick(lo, hi, args.n)], ans_check)
for w, lo, hi in [(3, 100, 999), (4, 1000, 4999), (5, 10000, 59999),
                  (6, 100000, 599999)]:
    cases = []
    for _ in range(args.n):
        a = rng.randint(lo, hi)
        b = rng.randint(100, 999)
        cases.append((f"问: {a}+{b} = ?\n", a + b))
    run("W", f"{w}位", f"加法（{w}位+3位）", cases, ans_check)

# ================= [E1] 挖洞 =================
print("\n[E1] 挖洞（公理套用）")
run("E1", "L2", f"挖洞后继 s(X{HO})",
    [(f"问: s({a}) = ?\n", a + 1) for a in pick_hole(100, 9999, args.n)], ans_check)
hole_d = [(f"问: {a} 是几位数?\n", len(str(a))) for a in pick_hole(100, 9999, args.n)]
run("E1", "L2", f"挖洞数位数（X{HO}）", hole_d, num_check("答"))

# ================= [E4] 组合泛化 =================
print("\n[E4] 组合泛化")
run("E4", "C1", "s(s(a))",
    [(f"问: s(s({a})) = ?\n", a + 2) for a in pick(100, 2000, args.n)], ans_check)

# ================= [E6] 序关系（v3 的 ⊥ 全错） =================
print("\n[E6] 序关系（v3 的 ⊥ 是 0%）")
lt = [(f"问: {a}<{b}?\n", True) for a, b in
      [(rng.randint(100, 9998), 0) for _ in range(args.n)]]
lt = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    lt.append((f"问: {a}<{a + rng.randint(1, 10)}?\n", True))
run("E6", "L1", "a<b（⊤）", lt, order_check)
ngt = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    ngt.append((f"问: {a}<{a - rng.randint(1, 10)}?\n", False))
run("E6", "L1", "a>b（⊥，方向修正）", ngt, order_check)
eq = [(f"问: {a}<{a}?\n", False) for a in pick(100, 9998, args.n)]
run("E6", "L1", "a=a（⊥）", eq, order_check)

# ================= [N] 读法层 =================
print("\n[N] 读法层")
from readnum import cn_read, en_read
run("N", "L1", "数值→中文（4位）",
    [(f"{a} = ", cn_read(a)) for a in pick(1000, 9999, args.n)], read_check)
run("N", "L1", "数值→英文（4位，prompt 带 English）",
    [(f"{a} in English: ", en_read(a)) for a in pick(1000, 9999, args.n)], read_check)
cn5 = []
for _ in range(args.n):
    a = rng.randint(10000, 99999)
    cn5.append((f"{a} = ", cn_read(a)))
run("N", "L2", "数值→中文（万级）", cn5, read_check)
run("N", "L3", "缩写→数值",
    [(f"{s} = ", n) for s, n in
     [("5千", 5000), ("2k", 2000), ("3w", 30000), ("1.5万", 15000),
      ("1M", 1000000), ("10k", 10000)]], read_check)

# ================= 汇总 =================
print("\n" + "=" * 72)
dims = {}
for dim, level, label, ok, total in results:
    dims.setdefault(dim, []).append((level, label, ok, total))
names = {"C": "数位数（核心）", "W": "宽度扫描（破墙）", "E1": "挖洞",
         "E4": "组合泛化", "E6": "序关系", "N": "读法层"}
for dim in ["C", "W", "E1", "E4", "E6", "N"]:
    if dim not in dims:
        continue
    print(f"\n  {dim} {names[dim]}")
    for level, label, ok, total in dims[dim]:
        pct = 100.0 * ok / total if total else 0
        print(f"    {level:<6} {label:<26} {ok}/{total} = {pct:5.1f}%")

print("\n判据：")
print("  C 训练内 1~4 位 ~100% → 数位数学会")
print("  C 外推 5~8 位 高 → 数位数宽度无关（关键！）")
print("  W-5位 后继/加法 ≥60% → 破墙成功（v3 是 0%）")
print("  E6 ⊥ 高 → 序方向修正成功（v3 是 0%）")