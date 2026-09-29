# -*- coding: utf-8 -*-
"""04c_eval_v3.py — v3 评估：绑定行分段抄写能否破 5 位墙
目的：v2 的墙在绑定行（x=97443 整体抄丢位）；v3 绑定行分段（x=9,7,4,4,3）。
判据（提前定好）：
  · W-5位 后继/加法 ≥60% → 绑定行分段成功，全链条宽度无关（v1/v2 是 0%）
  · W-6/7位 = 真外推战场
  · E1 挖洞 ≥80% → 公理套用保持（v2 是 80%）
  · E6 序 100% → 回归无损
  · E4 组合泛化（v2 是 0%，4 层容量问题——观察项）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random
import re

parser = argparse.ArgumentParser(description="v3 评估（绑定分段）")
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

MAX_NEW = 340


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
    ("问: s(4483) = ?\n理解: 算式 F, F := s(x), x := 4,4,8,3\n"
     "算法: 后继公理 s(x)=x+1：i 从 1 到 4 循环，逐位加 c（c 初值 1，a_i 从右往左取）\n"
     "i=1: x_1=3, c=1, s_1=3+1=4, d_1=4, c:=0\n"
     "i=2: x_2=8, c=0, s_2=8+0=8, d_2=8, c:=0\n"
     "i=3: x_3=4, c=0, s_3=4+0=4, d_3=4, c:=0\n"
     "i=4: x_4=4, c=0, s_4=4+0=4, d_4=4, c:=0\n"
     "结束: c=0，无补位\n合成: 4,4,8,4\n答: 4484", "问: s(4483) = ?\n"),
    ("问: 4483+579 = ?\n理解: 算式 F, F := a+b, a := 4,4,8,3, b := 5,7,9\n"
     "算法: i 从 1 到 4 循环：s_i := a_i+b_i+c; d_i := s_i mod 10; c := s_i div 10（c 初值 0，a_i 从右往左取）\n"
     "i=1: a_1=3, b_1=9, c=0, s_1=3+9+0=12, d_1=2, c:=1\n"
     "i=2: a_2=8, b_2=7, c=1, s_2=8+7+1=16, d_2=6, c:=1\n"
     "i=3: a_3=4, b_3=5, c=1, s_3=4+5+1=10, d_3=0, c:=1\n"
     "i=4: a_4=4, b_4=0, c=1, s_4=4+0+1=5, d_4=5, c:=0\n"
     "结束: c=0，无补位\n合成: 5,0,6,2\n答: 5062", "问: 4483+579 = ?\n"),
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
        print(f"            -> {o[:90]!r}")
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
    """循环格式判据：答: X 正确 且 合成行存在（token 级因果链）"""
    if "合成" not in out:
        return False
    m = re.search(r"答: (\d+)", out)
    return bool(m) and int(m.group(1)) == expect


def order_check(out, expect):
    """序关系（方案 A）：答 ⊤/⊥"""
    m = re.search(r"答: (\S+)", out)
    return bool(m) and m.group(1) == ("⊤" if expect else "⊥")


def read_check(out, expect):
    """读法层：输出含正确读法（中文/英文/缩写任一形式）"""
    return str(expect) in out


# ================= [W] 宽度扫描 =================
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

# ================= [E6] 序关系（方案 A：⊤/⊥ 三态） =================
print("\n[E6] 序关系（方案 A：命题绑定 + 传递闭包 + ⊤/⊥）")
# 训练内（差值 ≤10，与训练分布一致）
lt_cases = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    b = rng.randint(a + 1, min(a + 10, 9999))
    lt_cases.append((f"问: {a}<{b}?\n", True))
run("E6", "L1", "a<b（⊤，差值≤10）", lt_cases, order_check)
ngt_cases = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    b = rng.randint(max(0, a - 10), a - 1)
    ngt_cases.append((f"问: {a}<{b}?\n", False))
run("E6", "L1", "a>b（⊥，差值≤10）", ngt_cases, order_check)
eq_cases = [(f"问: {a}<{a}?\n", False) for a in pick(100, 9998, args.n)]
run("E6", "L1", "a=a（⊥）", eq_cases, order_check)
# ⚠️ 递归泛化（用户"先学走再学跑"）：训练只到差值 10，测试大差值
#    只需验证"循环能推下去"（行数=差值，O(差值) 是可接受的公理代价）
rec_cases = []
for _ in range(max(3, args.n // 3)):
    a = rng.randint(0, 500)
    b = a + rng.randint(20, 60)      # 差值 20~60（训练未见）
    rec_cases.append((f"问: {a}<{b}?\n", True))
run("E6", "R", "递归泛化（差值 20~60，⊤）", rec_cases, order_check)
# 宽度外推（5 位，小差值）
lt5 = []
for _ in range(max(3, args.n // 4)):
    a = rng.randint(10000, 99998)
    b = rng.randint(a + 1, min(a + 10, 99999))
    lt5.append((f"问: {a}<{b}?\n", True))
run("E6", "L3", "宽度外推（5 位，⊤）", lt5, order_check)

# ================= [N] 读法层 =================
print("\n[N] 读法层（数值 ↔ 中文/英文/缩写）")
from readnum import cn_read, en_read
run("N", "L1", "数值 → 中文读法（4 位）",
    [(f"{a} = ", cn_read(a)) for a in pick(1000, 9999, args.n)], read_check)
run("N", "L1", "数值 → 英文读法（4 位）",
    [(f"{a} = ", en_read(a)) for a in pick(1000, 9999, args.n)], read_check)
# ⚠️ 首版误留占位测试（lambda 恒 True = 假测试），已删除
cn5 = []
for _ in range(args.n):
    a = rng.randint(10000, 99999)
    cn5.append((f"{a} = ", cn_read(a)))
run("N", "L2", "数值 → 中文读法（万级）", cn5, read_check)
run("N", "L3", "缩写 → 数值",
    [(f"{s} = ", n) for s, n in
     [("5千", 5000), ("2k", 2000), ("3w", 30000), ("1.5万", 15000),
      ("1M", 1000000), ("10k", 10000)]], read_check)
cn_rev = []
for _ in range(args.n):
    a = rng.randint(10000, 99999)
    cn_rev.append((f"{cn_read(a)} = ", a))
run("N", "L2", "中文读法 → 数值（反向）", cn_rev, read_check)

# ================= 汇总 =================
print("\n" + "=" * 70)
print("汇总（按维度）")
dims = {}
for dim, level, label, ok, total in results:
    dims.setdefault(dim, []).append((level, label, ok, total))
names = {"W": "宽度扫描（核心）", "E1": "挖洞（公理套用）", "E4": "组合泛化",
         "E6": "序关系（⊤/⊥）", "N": "读法层"}
for dim in ["W", "E1", "E4", "E6", "N"]:
    if dim not in dims:
        continue
    print(f"\n  {dim} {names[dim]}")
    for level, label, ok, total in dims[dim]:
        pct = 100.0 * ok / total if total else 0
        print(f"    {level:<6} {label:<28} {ok}/{total} = {pct:5.1f}%")

print("\n判据：")
print("  W-5位 后继/加法 ≥60% → 循环格式成功（v1/v2 是 0%）")
print("  W-6/7/8/9位 = 真外推战场（8/9 位有锚点，5/7 位是纯外推对照）")
print("  E1 挖洞 ≥80% → 公理套用保持")
print("  E6 三态（⊤/⊥）都须高 → 真三态（旧版只有'成立'是假测试）")
print("  N 读法层高 → 潜空间对齐（数值 ↔ 读法）")
