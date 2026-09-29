# -*- coding: utf-8 -*-
"""01e_make_math_v5.py — v5 数据生成：扩量 + 取位层（配合 NAVE）

相对 v4 的两处改动（用户要求 2026-09-19）：
  【1】**扩量**（用户："多做一些数据不要拿相同数据跑批次"）
      各层分层枚举扩量 → 总量 ~60000 条（v4 是 36996）
      目标：用更少 epoch（2 轮）达到同等/更好覆盖，减少"同数据重复"
  【2】**新增 [R5] 取位层**（NAVE 核心用法：从值取位）
      `问: 94692 的百位是几?` → `理解: 数 v := 94692` → `算法: (v div 100) mod 10` → `答: 6`
      这是"取位运算"的训练信号——NAVE 让值可见后，取位应变成纯运算。

⚠️ **刻意不训练"抄写"任务**（v4-A 的 copy probe 是零样本探针）：
   若训练了 copy，就无法判断"NAVE 是否让复制变容易"——对照会被污染。
   保持 copy 为纯净探针（见 04e_eval_v5.py 的 [CP] 段）。

继承 v4：数位数层 R4 / 序方向修正 / 两级均衡分层枚举 / 读法层。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import random

parser = argparse.ArgumentParser(description="v5 数据生成（扩量 + 取位层）")
parser.add_argument("--out", default="data/math_v5.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total", type=int, default=65000)
parser.add_argument("--max-n", type=int, default=9999, help="训练数位宽上限（4 位；5+ 是外推）")
parser.add_argument("--hole-digit", type=int, default=7)
args = parser.parse_args()

rng = random.Random(args.seed)
from readnum import cn_read, en_read, cn_abbr, en_abbr, w_read

W = args.max_n
HO = args.hole_digit
POS_CN = {0: "个位", 1: "十位", 2: "百位", 3: "千位"}


def is_held(n):
    return 0 <= n <= W and n % 10 == HO


def digits_low_first(n):
    out = []
    while n >= 10:
        out.append(n % 10)
        n //= 10
    out.append(n)
    return out


def seg_str(n):
    return ",".join(c for c in str(n))


# ================= 分层枚举（继承 v4） =================
def next_carry_chain(n):
    k = 0
    while n % 10 == 9:
        k += 1
        n //= 10
    return k


def add_carry_chain(a, b):
    k = 0
    da, db = digits_low_first(a), digits_low_first(b)
    carry = 0
    for i in range(max(len(da), len(db))):
        ai = da[i] if i < len(da) else 0
        bi = db[i] if i < len(db) else 0
        s = ai + bi + carry
        if s >= 10:
            k += 1
            carry = 1
        else:
            break
    return k


def enumerate_succ_buckets(w_max):
    b = {}
    for n in range(1, 10 ** w_max):
        if n % 10 == HO:
            continue
        b.setdefault((len(str(n)), next_carry_chain(n)), []).append(n)
    return b


def enumerate_add_buckets(w_max):
    b = {}
    for a in range(1, 10 ** w_max):
        if a % 10 == HO:
            continue
        for b2 in range(1, 10 ** w_max):
            if b2 % 10 == HO:
                continue
            c = a + b2
            if len(str(c)) > w_max + 1 or c % 10 == HO:
                continue
            b.setdefault((len(str(a)), len(str(b2)), add_carry_chain(a, b2)), []).append((a, b2))
    return b


def balanced_pick(buckets, total, rng):
    """两级均衡（位宽 → 进位链长），继承 v4 修正版"""
    if not buckets:
        return []
    by_width = {}
    for k, pool in buckets.items():
        w = k[0] if len(k) == 2 else max(k[0], k[1])
        by_width.setdefault(w, {})[k] = pool
    widths = sorted(by_width)
    per_width = total // len(widths)
    out = []
    for w in widths:
        sub = by_width[w]
        keys = sorted(sub)
        pools = {k: list(sub[k]) for k in keys}
        for k in keys:
            rng.shuffle(pools[k])
        ptr = {k: 0 for k in keys}
        got = 0
        while got < per_width:
            progressed = False
            for k in keys:
                if ptr[k] < len(pools[k]):
                    out.append(pools[k][ptr[k]])
                    ptr[k] += 1
                    got += 1
                    progressed = True
                    if got >= per_width:
                        break
            if not progressed:
                break
    if len(out) < total:
        pool_all = [x for w in widths for k in by_width[w] for x in by_width[w][k]]
        rng.shuffle(pool_all)
        out += pool_all[:total - len(out)]
    return out


# ================= 生成器 =================
def count_digits(n):
    """数位数（v4 层）"""
    ds = [int(c) for c in str(n)]
    cnt = ", ".join(f"i={i}({d})" for i, d in enumerate(ds, 1))
    return (f"问: {n} 是几位数?\n理解: 数列 L := {seg_str(n)}\n"
            f"计数: {cnt}\n结束: 数列已取完\n答: {len(ds)}")


def digit_at(n, pos):
    """【v5 新增】取位（从值取第 pos 位，pos=0 个位）——NAVE 核心用法

    格式（用 div/mod 显式表达"取位是运算"）：
      问: 94692 的百位是几?
      理解: 数 v := 94692
      算法: 百位 = (v div 100) mod 10 = 946 mod 10
      答: 6
    """
    d = (n // (10 ** pos)) % 10
    pname = POS_CN[pos]
    div = 10 ** pos
    return (f"问: {n} 的{pname}是几?\n理解: 数 v := {n}\n"
            f"算法: {pname} = (v div {div}) mod 10 = {n // div} mod 10\n答: {d}")


def loop_add(a, b):
    c = a + b
    da, db = digits_low_first(a), digits_low_first(b)
    k = max(len(da), len(db))
    lines = [
        f"问: {a}+{b} = ?",
        f"理解: 算式 F, F := a+b, a := {seg_str(a)}, b := {seg_str(b)}",
        f"算法: i 从 1 到 {k} 循环：s_i := a_i+b_i+c; d_i := s_i mod 10; c := s_i div 10"
        f"（c 初值 0，a_i 从右往左取）",
    ]
    carry, ds = 0, []
    for i in range(1, k + 1):
        ai = da[i - 1] if i - 1 < len(da) else 0
        bi = db[i - 1] if i - 1 < len(db) else 0
        s = ai + bi + carry
        d, new_c = s % 10, s // 10
        lines.append(f"i={i}: a_{i}={ai}, b_{i}={bi}, c={carry}, "
                     f"s_{i}={ai}+{bi}+{carry}={s}, d_{i}={d}, c:={new_c}")
        ds.append(d)
        carry = new_c
    if carry:
        lines.append(f"结束: c={carry}≠0，补位 d_{k + 1}={carry}, c:=0")
        ds.append(carry)
    else:
        lines.append("结束: c=0，无补位")
    lines.append("合成: " + ",".join(str(d) for d in reversed(ds)))
    lines.append(f"答: {c}")
    return "\n".join(lines)


def loop_succ(a):
    b = a + 1
    dx = digits_low_first(a)
    k = len(dx)
    lines = [
        f"问: s({a}) = ?",
        f"理解: 算式 F, F := s(x), x := {seg_str(a)}",
        f"算法: 后继公理 s(x)=x+1：i 从 1 到 {k} 循环，逐位加 c（c 初值 1，a_i 从右往左取）",
    ]
    carry, ds = 1, []
    for i in range(1, k + 1):
        xi = dx[i - 1]
        s = xi + carry
        d, new_c = s % 10, s // 10
        lines.append(f"i={i}: x_{i}={xi}, c={carry}, s_{i}={xi}+{carry}={s}, d_{i}={d}, c:={new_c}")
        ds.append(d)
        carry = new_c
    if carry:
        lines.append(f"结束: c={carry}≠0，补位 d_{k + 1}={carry}, c:=0")
        ds.append(carry)
    else:
        lines.append("结束: c=0，无补位")
    lines.append("合成: " + ",".join(str(d) for d in reversed(ds)))
    lines.append(f"答: {b}")
    return "\n".join(lines)


def order_chain(a, b):
    """序关系（v4 方向修正：从 a 出发）"""
    lines = [
        f"问: {a}<{b}?",
        f"理解: 命题 P, P := ({a}<{b})",
        "算法: < 是后继 s 的传递闭包：a<b ⟺ ∃n≥1, sⁿ(a)=b（从 a 出发逐次后继）",
    ]
    if a < b:
        steps = b - a
        parts, cur = [], a
        for _ in range(steps):
            parts.append(f"s({cur})={cur + 1}")
            cur += 1
        lines.append("尝试: " + ", ".join(parts))
        lines.append(f"判断: 从 a={a} 经 n={steps} 次后继到达 b={b}，n={steps}≥1")
        lines.append("答: ⊤")
    elif a == b:
        lines.append(f"尝试: 需 n≥1 次后继到达自身（a=b={a}）")
        lines.append("判断: a=b，需 n=0，但要求 n≥1，不满足")
        lines.append("答: ⊥")
    else:
        show = min(4, a - b)
        parts, cur = [], a
        for _ in range(show):
            parts.append(f"s({cur})={cur + 1}")
            cur += 1
        lines.append("尝试: " + ", ".join(parts) + "（继续只会更大）")
        lines.append(f"判断: 从 a={a} 出发单调递增，无法到达更小的 b={b}")
        lines.append("答: ⊥")
    return "\n".join(lines)


# ================= 组装（v5 扩量） =================
texts = []


def add(lst, label):
    texts.extend(lst)
    print(f"  {label:<50} {len(lst):>7} 条")


print(f"[挖洞] 末位 {HO} 的 1~4 位数实例删除（公理保留）")

# [D] 定义层
defs = [
    "自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}",
    "后继 Successor: s(a) = a+1",
    "加法 Addition: a+0 = a, a+s(b) = s(a+b)",
    "序 Order: a<b ⟺ ∃n≥1, sⁿ(a)=b（< 是后继 s 的传递闭包）",
    "位数 Digits: n 的位数 = 其十进制写法中数字的个数",
    "取位 Digit: n 的第 k 位 = (n div 10^k) mod 10（k 从 0 起）",
    "0 是自然数: 0∈ℕ",
    "自然数的后继是自然数: a∈ℕ ⟹ s(a)∈ℕ",
]
add(defs * 45, "[D] 定义层（+取位定义，×45）")

# [A] 公理层
axioms = []
axiom_texts = [
    "公理: ∀a∈ℕ. s(a) = a+1",
    "公理: ∀a∈ℕ. a+0 = a",
    "公理: ∀a,b∈ℕ. a+s(b) = s(a+b)",
    "公理: ∀a∈ℕ. s(a) ≠ 0",
    "公理: s(a)=s(b) ⟹ a=b",
    "公理: ∀a,b∈ℕ. a<b ⟺ ∃n≥1, sⁿ(a)=b",
]
axioms.extend(axiom_texts * 45)
for _ in range(3500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: s({a}) = {a}+1 = {a + 1}")
for _ in range(2000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: {a}+0 = {a}")
add(axioms, "[A] 公理层（代数 ×45 + 实例化 5500）")

# [R4] 数位数层（扩量：位宽分桶 + 锚点 + 桥接）
r4 = []
per_width = 700
for w in range(1, len(str(W)) + 1):
    lo, hi = (0 if w == 1 else 10 ** (w - 1)), 10 ** w - 1
    pool = [n for n in range(lo, min(hi, W) + 1) if not is_held(n)]
    if pool:
        for n in rng.sample(pool, min(per_width, len(pool))):
            r4.append(count_digits(n))
for n in (0, 1, 9, 10, 99, 100, 999, 1000, 9999, 5000, 1234, 1010, 4003):
    r4.append(count_digits(n))
for _ in range(1200):
    a = rng.randint(0, min(W, 999))
    if is_held(a):
        continue
    k = len(str(a))
    cnt = ", ".join(f"i={i}({d})" for i, d in enumerate([int(c) for c in str(a)], 1))
    r4.append(f"问: {a} 是几位数?\n理解: 数列 L := {seg_str(a)}\n计数: {cnt}\n"
              f"结束: 数列已取完\n答: {k}\n用途: 故加法循环 i 从 1 到 {k}")
add(r4, "[R4] 数位数层（位宽分桶 700/宽 + 锚点 + 桥接）")

# [R5] 取位层（v5 新增，NAVE 核心用法）
r5 = []
# 按 (位宽, 位置) 分层均衡
for w in range(1, len(str(W)) + 1):
    lo, hi = (0 if w == 1 else 10 ** (w - 1)), 10 ** w - 1
    pool = [n for n in range(lo, min(hi, W) + 1) if not is_held(n)]
    if not pool:
        continue
    for pos in range(w):
        take = min(400, len(pool))
        for n in rng.sample(pool, take):
            r5.append(digit_at(n, pos))
# 取位锚点（确定性）
for n in (0, 5, 9, 10, 99, 100, 999, 1000, 9999, 4483, 5000, 1010):
    for pos in range(len(str(n))):
        r5.append(digit_at(n, pos))
add(r5, "[R5] 取位层（位宽×位置分层 + 锚点）")

# [R1] 后继（扩量）
r1 = []
succ_buckets = enumerate_succ_buckets(len(str(W)))
for n in balanced_pick(succ_buckets, 6000, rng):
    r1.append(loop_succ(n))
for n in (9, 99, 999, 9999, 1999, 4999, 8999, 9099, 9909):
    for _ in range(5):
        r1.append(loop_succ(n))
for _ in range(3000):
    a = rng.randint(0, 300)
    if any(is_held(a + j) for j in range(3)):
        continue
    r1.append(f"问: s(s({a})) = ?\n理解: 算式 F, F := s(s(x)), x := {seg_str(a)}\n"
              f"算法: 后继公理两次\n第一步: s({a}) = {a + 1}\n"
              f"第二步: s({a + 1}) = {a + 2}\n答: {a + 2}")
add(r1, "[R1] 后继（分层 6000 + 进位锚点 + 复合 3000）")

# [R2] 加法（扩量）
r2 = []
add_buckets = enumerate_add_buckets(3)
for a, b in balanced_pick(add_buckets, 4500, rng):
    r2.append(loop_add(a, b))
cnt4 = 0
while cnt4 < 1800:
    a, b = rng.randint(1000, W), rng.randint(100, W)
    if a + b > W + 10 or is_held(a) or is_held(b) or is_held(a + b):
        continue
    r2.append(loop_add(a, b))
    cnt4 += 1
for a, b in ((999, 1), (9999, 1), (4999, 1), (1999, 1), (99, 1), (9, 1),
             (9990, 9), (9909, 90), (990, 10), (5000, 5000)):
    for _ in range(5):
        r2.append(loop_add(a, b))
add(r2, "[R2] 加法（分层 4500 + 4位补 1800 + 进位锚点）")

# [R3] 序关系（扩量）
r3 = []
for _ in range(6500):
    a = rng.randint(0, W - 1)
    mode = rng.random()
    if mode < 0.45:
        b = rng.randint(a + 1, min(a + 10, W))
    elif mode < 0.85:
        b = rng.randint(max(0, a - 10), a - 1) if a >= 1 else a + rng.randint(1, 5)
    else:
        b = a
    if is_held(a) or is_held(b):
        continue
    r3.append(order_chain(a, b))
for a, b in ((3, 4), (4, 3), (4, 4), (1, 2), (2, 1), (1, 11), (11, 1),
             (500, 505), (505, 500), (500, 500), (9999, 9998), (9998, 9999)):
    for _ in range(5):
        r3.append(order_chain(a, b))
add(r3, "[R3] 序关系（三态 6500 + 方向锚点）")

# [L] 对齐层（扩量）
l1 = []
cn_variants = [
    lambda a, b: f"{a} 的后继是 {b}。",
    lambda a, b: f"{a} 的下一个数是 {b}。",
    lambda a, b: f"{a} 加一是 {b}。",
    lambda a, b: f"{a} 再数一个就是 {b}。",
]
for _ in range(3000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l1.append(rng.choice(cn_variants)(a, a + 1))
for _ in range(1600):
    a, b = rng.randint(0, 500), rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l1.append(f"{a} 加 {b} 等于 {a + b}。")
for _ in range(1200):
    a = rng.randint(0, W - 1)
    mode = rng.random()
    if mode < 0.5:
        b = rng.randint(a + 1, min(a + 20, W))
        if is_held(a) or is_held(b):
            continue
        l1.append(f"{a} 小于 {b}。")
    elif mode < 0.9:
        b = rng.randint(max(0, a - 20), a - 1) if a >= 1 else a + rng.randint(1, 5)
        if is_held(a) or is_held(b):
            continue
        l1.append(f"{a} 大于 {b}。")
    else:
        if is_held(a):
            continue
        l1.append(f"{a} 等于 {a}。")
for _ in range(1200):
    n = rng.randint(0, W)
    if is_held(n):
        continue
    l1.append(f"{n} 是 {len(str(n))} 位数。")
add(l1, "[L1] 中文界面（+位数表述，扩量）")

l2 = []
en_variants = [
    lambda a, b: f"The successor of {a} is {b}.",
    lambda a, b: f"The next number after {a} is {b}.",
    lambda a, b: f"{a} plus one is {b}.",
]
for _ in range(2000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l2.append(rng.choice(en_variants)(a, a + 1))
for _ in range(900):
    a, b = rng.randint(0, 500), rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l2.append(f"{a} plus {b} equals {a + b}.")
for _ in range(700):
    a = rng.randint(0, W - 1)
    mode = rng.random()
    if mode < 0.5:
        b = rng.randint(a + 1, min(a + 20, W))
        if is_held(a) or is_held(b):
            continue
        l2.append(f"{a} is less than {b}.")
    elif mode < 0.9:
        b = rng.randint(max(0, a - 20), a - 1) if a >= 1 else a + rng.randint(1, 5)
        if is_held(a) or is_held(b):
            continue
        l2.append(f"{a} is greater than {b}.")
    else:
        if is_held(a):
            continue
        l2.append(f"{a} is equal to {a}.")
for _ in range(700):
    n = rng.randint(0, W)
    if is_held(n):
        continue
    l2.append(f"{n} has {len(str(n))} digit(s).")
add(l2, "[L2] 英文界面（+位数表述，扩量）")

l3 = []
for _ in range(2000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    b = a + 1
    trio = [rng.choice(cn_variants)(a, b), rng.choice(en_variants)(a, b), f"s({a}) = {b}"]
    rng.shuffle(trio)
    l3.append("\n".join(trio))
add(l3, "[L3] 三形式对齐配对")

# [N] 读法层（扩量）
n_texts = []
for _ in range(2200):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts += [f"{n} = {cn_read(n)}", f"{cn_read(n)} = {n}"]
for _ in range(1000):
    n = rng.randint(10000, 10 ** 8)
    if is_held(n):
        continue
    n_texts += [f"{n} = {cn_read(n)}", f"{cn_read(n)} = {n}"]
for _ in range(400):
    n = rng.randint(10 ** 8, 10 ** 9)
    n_texts += [f"{n} = {cn_read(n)}", f"{cn_read(n)} = {n}"]
for n in (10, 110, 4003, 10000, 10001, 1000000, 4000300, 100000001, 12345678, 1010):
    n_texts += [f"{n} = {cn_read(n)}", f"{cn_read(n)} = {n}"]
for _ in range(1400):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts += [f"{n} = {en_read(n)}", f"{en_read(n)} = {n}"]
for _ in range(700):
    n = rng.randint(10000, 10 ** 9)
    n_texts += [f"{n} = {en_read(n)}", f"{en_read(n)} = {n}"]
for n in (100, 110, 1000, 4003, 10 ** 6, 1000001):
    n_texts += [f"{n} = {en_read(n)}", f"{en_read(n)} = {n}"]
for _ in range(900):
    n = rng.randint(1, 9999) * 1000
    s = cn_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(700):
    n = rng.randint(1, 9999) * 10000
    s = cn_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(550):
    n = rng.randint(1, 999) * 1000
    s = en_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(300):
    n = rng.randint(1, 99) * 10 ** 6
    s = en_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(400):
    n = rng.randint(1, 9999) * 10000
    s = w_read(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for s, n in (("5千", 5000), ("2k", 2000), ("3w", 30000), ("1.5万", 15000),
             ("1M", 1000000), ("10k", 10000), ("1亿", 10 ** 8)):
    n_texts += [f"{s} = {n}", f"{n} = {s}"]
add(n_texts, "[N] 读法层（扩量）")

# ================= 确定性锚点 =================
for _ in range(4):
    texts.append(count_digits(97443))
    texts.append(count_digits(4483))
    texts.append(count_digits(999))
    texts.append(digit_at(94692, 2))        # 百位 = 6
    texts.append(digit_at(4483, 0))         # 个位 = 3
    texts.append(loop_succ(4483))
    texts.append(loop_succ(9999))
    texts.append(loop_add(4483, 579))
    texts.append(order_chain(3, 4))
    texts.append(order_chain(4, 3))
    texts.append(order_chain(4, 4))

# ================= 泄漏检查 =================
print("=" * 70)


def leak_variants(a):
    b = a + 1
    k = len(str(a))
    cnt = ", ".join(f"i={i}({d})" for i, d in enumerate(str(a), 1))
    out = [
        loop_succ(a), count_digits(a),
        f"公理实例: s({a}) = {a}+1 = {b}",
        f"{a} 的后继是 {b}。", f"{a} 的下一个数是 {b}。", f"{a} 加一是 {b}。",
        f"{a} 再数一个就是 {b}。", f"The successor of {a} is {b}.",
        f"The next number after {a} is {b}.", f"{a} plus one is {b}.",
        f"s({a}) = {b}", f"{a} 小于 {b}。", f"{a} is less than {b}。",
        f"{a} 大于 {a - 1}。", f"{a} is greater than {a - 1}。",
        f"{a} 等于 {a}。", f"{a} 是 {k} 位数。", f"{a} has {k} digit(s).",
        order_chain(a, b),
        f"问: {a} 是几位数?\n理解: 数列 L := {seg_str(a)}\n计数: {cnt}\n结束: 数列已取完\n答: {k}",
    ]
    for pos in range(len(str(a))):
        out.append(digit_at(a, pos))
    return out


n_leak = 0
text_set = set(texts)
for n in range(0, W + 1):
    if n % 10 == HO:
        for v in leak_variants(n):
            if v in text_set:
                n_leak += 1
                print(f"  ⚠️ 泄漏: {v[:60]}")
print(f"[泄漏检查] {n_leak} 条（应为 0）")

rng.shuffle(texts)
if len(texts) > args.total:
    texts = texts[:args.total]
with open(args.out, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print("=" * 70)
print(f"总计 {len(texts)} 条 -> {args.out}")

# ================= 自检 =================
import re
S = set(texts)
checks = [
    (count_digits(97443), "数位数（5 位锚点）"),
    (digit_at(94692, 2), "取位（百位=6）"),
    (digit_at(4483, 0), "取位（个位=3）"),
    (loop_succ(4483), "后继循环"),
    (loop_add(4483, 579), "加法循环"),
    (order_chain(3, 4), "序 ⊤"),
    (order_chain(4, 3), "序 ⊥（方向）"),
    (order_chain(4, 4), "序 ⊥（相等）"),
    ("公理: ∀a∈ℕ. s(a) = a+1", "公理代数"),
]
n_miss = 0
print("\n关键样例自检：")
for text, label in checks:
    ok = text in S
    if not ok:
        n_miss += 1
    print(f"   {'OK  ' if ok else 'MISS'} {label}")

n_bad = 0
n_checked = 0
for t in texts:
    m = re.match(r"^问: (\d+) 是几位数\?", t)
    if m:
        a = int(m.group(1))
        ma = re.search(r"答: (\d+)", t)
        if not ma or int(ma.group(1)) != len(str(a)):
            n_bad += 1
            print(f"   BAD digits: a={a}")
        n_checked += 1
    m = re.match(r"^问: (\d+) 的(\S+?)是几\?", t)
    if m:
        a = int(m.group(1))
        pname = m.group(2)
        pos = {"个位": 0, "十位": 1, "百位": 2, "千位": 3}[pname]
        ma = re.search(r"答: (\d+)", t)
        exp = (a // (10 ** pos)) % 10
        if not ma or int(ma.group(1)) != exp:
            n_bad += 1
            print(f"   BAD digit_at: {a} {pname} 期望 {exp}")
        n_checked += 1
    m = re.match(r"^问: s\((\d+)\) = \?", t)
    if m:
        a = int(m.group(1))
        ma = re.search(r"答: (\d+)$", t)
        if not ma or int(ma.group(1)) != a + 1:
            n_bad += 1
            print(f"   BAD succ: {a}")
        n_checked += 1
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ma = re.search(r"答: (\d+)$", t)
        if not ma or int(ma.group(1)) != a + b:
            n_bad += 1
            print(f"   BAD add: {a}+{b}")
        n_checked += 1
    m = re.match(r"^问: (\d+)<(\d+)\?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ma = re.search(r"答: (\S+)$", t)
        exp = "⊤" if a < b else "⊥"
        if not ma or ma.group(1) != exp:
            n_bad += 1
            print(f"   BAD order: {a}<{b} 期望 {exp}")
        n_checked += 1
print(f"\n抽检: {n_checked} 条中 {n_bad} 条错误")

n_t = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊤"))
n_f = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊥"))
nd = sum(1 for t in texts if re.match(r"^问: \d+ 是几位数\?", t))
npos = sum(1 for t in texts if re.match(r"^问: \d+ 的\S+?是几\?", t))
print(f"[分布] 序 ⊤{n_t}/⊥{n_f} | 数位数 {nd} | 取位 {npos}")
if n_t == 0 or n_f == 0 or nd < 100 or npos < 100:
    n_bad += 1
    print("   ⚠️ 分布异常")

if n_miss or n_leak or n_bad:
    print(f"\n⚠️ 异常：缺失 {n_miss} / 泄漏 {n_leak} / 错误 {n_bad}")
    sys.exit(1)
print("\n✅ 全部就位，可训练")