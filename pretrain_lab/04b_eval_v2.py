# -*- coding: utf-8 -*-
"""04b_eval_v2.py — v2 评估：大纲渐进展开格式的宽度外推
目的：验证"只训 1~4 位，大纲格式能否外推到 5~7 位"（用户核心假设）。

评估维度：
  [W] 宽度扫描（核心新增）：后继/加法在 3/4/5/6/7 位上的准确率
      —— v1 对照：5 位 0%（丢位）。v2 判据：5 位 ≥60% 算成功
  [E1] 公理套用（挖洞）：s(7777) 类，训练删光实例
  [E2] 加法大纲执行：训练内/挖洞
  [E3] 跨形式一致性（v1 是 0%，v2 数据仍无混合任务——预期仍低，记录基线）
  [E4] 组合泛化：s(s(a))
  [E6] 序关系（v1 是 100%，回归测试）
  [P] 探针 C 复测：逐位读出（训练后应从零样本 70% 提升）

判据（提前定好）：
  · W-5位 后继/加法 ≥60% → 大纲方案成立（v1 是 0%）
  · W-6/7位 是真外推战场（零样本边界在 5~6 位）
  · E1 挖洞 ≥90% → 公理套用能力保持（v1 是 100%）
  · E6 序 100% → 回归无损
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random
import re

parser = argparse.ArgumentParser(description="v2 评估（大纲格式）")
parser.add_argument("--model", required=True)
parser.add_argument("--n", type=int, default=15)
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

# 大纲格式样本较长（6 位加法 ~28 行），MAX_NEW 给足
MAX_NEW = 320


def gen(prompt, max_new=MAX_NEW):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


# ================= [S0] 前缀一致性自检 =================
print("[S0] prompt 前缀一致性自检")
SELF_CHECKS = [
    ("问: s(4483) = ?\n理解: 算式 F, F=s(x), x=4483\n算法: 后继公理 s(x)=x+1，个位加一，满十进一\n个位: x=3, 3+1+0 = 4, 写4进0\n十位: x=8, 8+0+0 = 8, 写8进0\n百位: x=4, 4+0+0 = 4, 写4进0\n千位: x=4, 4+0+0 = 4, 写4进0\n合成: 4,4,8,4\n答: 4484", "问: s(4483) = ?\n"),
    ("问: 4483+579 = ?\n理解: 算式 F, F=a+b, a=4483, b=579\n算法: 按位对齐，从个位起相加，满十进一\n个位: a=3, b=9, 3+9+0 = 12, 写2进1\n十位: a=8, b=7, 8+7+1 = 16, 写6进1\n百位: a=4, b=5, 4+5+1 = 10, 写0进1\n千位: a=4, b=0, 4+0+1 = 5, 写5进0\n合成: 5,0,6,2\n答: 5062", "问: 4483+579 = ?\n"),
]
for full, prefix in SELF_CHECKS:
    fids = tok.encode(full)
    pids = tok.encode(prefix)
    if fids[:len(pids)] != pids:
        print(f"  FAIL: {prefix!r} 不是训练样本 token 前缀！")
        sys.exit(1)
print("  OK（2/2）\n")

# ================= 测试框架 =================
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
    print(f"  [{level}] {label:<40} {ok}/{len(cases)} = {pct:5.1f}%")
    for p, o, e in fails[:show]:
        print(f"       FAIL {p[:40]!r}")
        print(f"            -> {o[:80]!r}")
    return pct


def pick(lo, hi, n, exclude_hole=True):
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if exclude_hole and v % 10 == HO:
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


def ans_check(out, expect):
    """大纲格式判据：答: X 正确 且 合成行存在（token 级因果链）"""
    if "合成" not in out:
        return False
    m = re.search(r"答: (\d+)", out)
    return bool(m) and int(m.group(1)) == expect


# ================= [W] 宽度扫描（核心） =================
print("[W] 宽度扫描（训练只到 4 位；5~7 位是外推）")
for w, lo, hi in [(3, 100, 999), (4, 1000, 9999), (5, 10000, 99999),
                  (6, 100000, 999999), (7, 1000000, 9999999)]:
    run("W", f"{w}位", f"后继 s({w}位数)",
        [(f"问: s({a}) = ?\n", a + 1) for a in pick(lo, hi, args.n)], ans_check)

for w, lo, hi in [(3, 100, 999), (4, 1000, 4999), (5, 10000, 59999),
                  (6, 100000, 599999), (7, 1000000, 5999999)]:
    cases = []
    for _ in range(args.n):
        a = rng.randint(lo, hi)
        b = rng.randint(100, 999)
        cases.append((f"问: {a}+{b} = ?\n", a + b))
    run("W", f"{w}位", f"加法（{w}位+3位）", cases, ans_check)

# ================= [E1] 挖洞 =================
print("\n[E1] 公理套用（挖洞协议）")
run("E1", "L2", f"挖洞后继 s(X{HO})（训练已删）",
    [(f"问: s({a}) = ?\n", a + 1) for a in pick_hole(100, 9999, args.n)], ans_check)
add_hole = []
for _ in range(args.n):
    a = pick_hole(100, 4999, 1)[0]
    b = rng.randint(1, 99)
    add_hole.append((f"问: {a}+{b} = ?\n", a + b))
run("E1", "L2", f"挖洞加法（a 末位 {HO}）", add_hole, ans_check)

# ================= [E4] 组合泛化 =================
print("\n[E4] 组合泛化")
run("E4", "C1", "s(s(a)) 符号复合",
    [(f"问: s(s({a})) = ?\n", a + 2) for a in pick(100, 2000, args.n)], ans_check)

# ================= [E6] 序关系（回归） =================
print("\n[E6] 序关系（v1 是 100%，回归测试）")
lt_cases = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    b = rng.randint(a + 1, a + 20)
    lt_cases.append((f"问: {a}<{b}?\n", True))
run("E6", "L1", "训练内序关系", lt_cases,
    lambda out, e: "成立" in out)

# ================= [P] 探针 C 复测（训练后） =================
print("\n[P] 探针 C 复测（逐位读出，训练后 vs 零样本 70%）")
for w, lo, hi in [(4, 1000, 9999), (5, 10000, 99999), (6, 100000, 999999)]:
    cases = []
    for _ in range(args.n):
        n = rng.randint(lo, hi)
        digits = [int(c) for c in str(n)][::-1]  # 低位在前
        cases.append((f"逐位读出 {n}:\n", digits))
    run("P", f"{w}位", "逐位读出（每行一位）", cases,
        lambda out, ds: all(f"{d}" in out for d in ds))

# ================= 汇总 =================
print("\n" + "=" * 70)
print("汇总（按维度）")
dims = {}
for dim, level, label, ok, total in results:
    dims.setdefault(dim, []).append((level, label, ok, total))
names = {"W": "宽度扫描（核心）", "E1": "挖洞（公理套用）", "E4": "组合泛化",
         "E6": "序关系（回归）", "P": "探针C复测"}
for dim in ["W", "E1", "E4", "E6", "P"]:
    if dim not in dims:
        continue
    print(f"\n  {dim} {names[dim]}")
    for level, label, ok, total in dims[dim]:
        pct = 100.0 * ok / total if total else 0
        print(f"    {level:<6} {label:<28} {ok}/{total} = {pct:5.1f}%")

print("\n判据：")
print("  W-5位 后继/加法 ≥60% → 大纲方案成立（v1 是 0%）")
print("  W-6/7位 = 真外推战场（零样本边界 5~6 位）")
print("  E1 挖洞 ≥90% → 公理套用保持（v1 是 100%）")
print("  E6 序 100% → 回归无损")
