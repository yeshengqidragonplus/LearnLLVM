# -*- coding: utf-8 -*-
"""01d_make_math_v4.py — v4 数据生成：数位数层 + 序方向修正 + 分层枚举
目的：修复 v3 暴露的核心瓶颈——**位数不可知**。

v3 结论（V3_SUMMARY.md）：
  · 循环格式（i=1..k）执行机制**正确**（3/4 位 100%）
  · 但循环上界 k 输出错（5 位时写 `i 从 1 到 4`）
  · 根因：**"位数"从未被显式训练**（是训练数据的隐含属性）
  · 同一病复发于：5+位后继/加法、万级读法、s(s(1400)) 组合泛化

v4 三处改动：
  【1】新增 [R4] 数位数层（核心）——把"位数"从隐含属性变成显式输出
      问: 97443 是几位数?
      理解: 数列 L := 9,7,4,4,3
      计数: i=1(9), i=2(7), i=3(4), i=4(4), i=5(3)
      结束: 数列已取完
      答: 5
      关键：数位数自身也要**宽度无关**（数 20 位就是数到 20）——
      训练只到 4 位，5+ 位是外推测试（与加法同一判据）
  【2】修正 [R3] 序方向语义——v3 的 ⊥ 全错（模型自动从 min 数到 max 答 ⊤）
      修法：尝试链**必须从 a 出发**（不是 min(a,b)），并显式判断方向
  【3】分层枚举替代随机采样——v3 随机采样导致"简单结构冗余 10 倍、
      难结构（长进位链）靠 8 个手加锚点硬撑"
      改为按 (位宽, 进位链长) 分桶，每桶均衡采样

其余继承 v3（循环格式、读法层、挖洞协议、D/A/L 层）。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import random

parser = argparse.ArgumentParser(description="v4 数据生成（数位数 + 序修正 + 分层枚举）")
parser.add_argument("--out", default="data/math_v4.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total", type=int, default=40000, help="总量上限（分层枚举约 3.7 万）")
parser.add_argument("--max-n", type=int, default=9999, help="训练数位宽上限（4 位；5+ 位是外推）")
parser.add_argument("--hole-digit", type=int, default=7, help="挖洞末位")
args = parser.parse_args()

rng = random.Random(args.seed)

from readnum import cn_read, en_read, cn_abbr, en_abbr, w_read

W = args.max_n
HO = args.hole_digit


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
    """97443 -> '9,7,4,4,3'（高位在前，逗号分隔）"""
    return ",".join(c for c in str(n))


# ================= 分层枚举工具 =================

def next_carry_chain(n):
    """后继时的进位链长 = 末尾连续 9 的个数"""
    k = 0
    while n % 10 == 9:
        k += 1
        n //= 10
    return k


def add_carry_chain(a, b):
    """加法的进位传播链长（从个位起连续产生进位的位数）"""
    k = 0
    da = digits_low_first(a)
    db = digits_low_first(b)
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
    """后继：按 (位宽, 进位链长) 分桶列举所有数"""
    buckets = {}
    for n in range(1, 10 ** w_max):
        if n % 10 == HO:
            continue
        buckets.setdefault((len(str(n)), next_carry_chain(n)), []).append(n)
    return buckets


def enumerate_add_buckets(w_max):
    """加法：按 (位宽a, 位宽b, 进位链长) 分桶（限制 a+b 不超过 w_max+1 位）"""
    buckets = {}
    for a in range(1, 10 ** w_max):
        if a % 10 == HO:
            continue
        for b in range(1, 10 ** w_max):
            if b % 10 == HO:
                continue
            c = a + b
            if len(str(c)) > w_max + 1:
                continue
            if c % 10 == HO:
                continue
            key = (len(str(a)), len(str(b)), add_carry_chain(a, b))
            buckets.setdefault(key, []).append((a, b))
    return buckets


def balanced_pick(buckets, total, rng):
    """两级均衡采样（v4 修正 v2）

    ⚠️ 首版 bug1：从"所有桶拼接的大池"补样 → 大池被 4 位数主导。
    ⚠️ 首版 bug2：单纯 round-robin 时，**桶数随位宽增长**（4 位 4 个桶、
        1 位 2 个桶）→ 4 位拿到的轮次是 1 位的 2 倍（仍有 228 倍失衡）。
    修正：**两级均衡**——
        第 1 级：按"位宽"分大组，各等分 total；
        第 2 级：组内按"进位链长"round-robin（保证难结构不缺席）。
    这样位宽均衡 + 进位链均衡同时满足。
    """
    if not buckets:
        return []
    # 第 1 级：按位宽分组
    # 注：后继 key=(位宽, 进位链)；加法 key=(位宽a, 位宽b, 进位链)
    #     加法取 max(位宽a, 位宽b) 作为"位宽"（与统计口径一致）
    by_width = {}
    for k, pool in buckets.items():
        w = k[0] if len(k) == 2 else max(k[0], k[1])
        by_width.setdefault(w, {})[k] = pool
    widths = sorted(by_width)
    per_width = total // len(widths)
    out = []
    for w in widths:
        sub = by_width[w]             # 该位宽下的所有（进位链长）桶
        keys = sorted(sub)
        pools = {k: list(sub[k]) for k in keys}
        for k in keys:
            rng.shuffle(pools[k])
        ptr = {k: 0 for k in keys}
        got = 0
        # 第 2 级：组内 round-robin
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
                break             # 该位宽池尽
    # 若总数不足（小位宽池小），用大位宽补齐（仍按 round-robin）
    if len(out) < total:
        pool_all = [x for w in widths for k in by_width[w] for x in by_width[w][k]]
        rng.shuffle(pool_all)
        need = total - len(out)
        out += pool_all[:need]
    return out


# ================= 【1】数位数层生成器（v4 核心新增） =================

def count_digits(n):
    """数位数格式：把"位数"从隐含属性变成显式输出

    问: 97443 是几位数?
    理解: 数列 L := 9,7,4,4,3
    计数: i=1(9), i=2(7), i=3(4), i=4(4), i=5(3)
    结束: 数列已取完
    答: 5

    关键：计数循环的**停止条件是"数列取完"**（不是"数到 4 为止"）——
    故宽度无关：5 位数就是数到 5，20 位就是数到 20。
    训练只到 4 位，5+ 位是外推测试。
    """
    ds = [int(c) for c in str(n)]           # 高位在前，与绑定行一致
    k = len(ds)
    cnt = ", ".join(f"i={i}({d})" for i, d in enumerate(ds, 1))
    return (f"问: {n} 是几位数?\n"
            f"理解: 数列 L := {seg_str(n)}\n"
            f"计数: {cnt}\n"
            f"结束: 数列已取完\n"
            f"答: {k}")


# ================= 循环格式生成器（继承 v3） =================

def loop_add(a, b):
    """加法循环格式（v3 已验证：check_chain_v3_tmp.py 14 用例 0 错）"""
    c = a + b
    da = digits_low_first(a)
    db = digits_low_first(b)
    k = max(len(da), len(db))
    lines = [
        f"问: {a}+{b} = ?",
        f"理解: 算式 F, F := a+b, a := {seg_str(a)}, b := {seg_str(b)}",
        f"算法: i 从 1 到 {k} 循环：s_i := a_i+b_i+c; d_i := s_i mod 10; c := s_i div 10"
        f"（c 初值 0，a_i 从右往左取）",
    ]
    carry = 0
    ds = []
    for i in range(1, k + 1):
        ai = da[i - 1] if i - 1 < len(da) else 0
        bi = db[i - 1] if i - 1 < len(db) else 0
        s = ai + bi + carry
        d = s % 10
        new_c = s // 10
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
    """后继循环格式（同构：i 从 1，个位 +1 起）"""
    b = a + 1
    dx = digits_low_first(a)
    k = len(dx)
    lines = [
        f"问: s({a}) = ?",
        f"理解: 算式 F, F := s(x), x := {seg_str(a)}",
        f"算法: 后继公理 s(x)=x+1：i 从 1 到 {k} 循环，逐位加 c"
        f"（c 初值 1，a_i 从右往左取）",
    ]
    carry = 1
    ds = []
    for i in range(1, k + 1):
        xi = dx[i - 1]
        s = xi + carry
        d = s % 10
        new_c = s // 10
        lines.append(f"i={i}: x_{i}={xi}, c={carry}, "
                     f"s_{i}={xi}+{carry}={s}, d_{i}={d}, c:={new_c}")
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


# ================= 【2】序关系（v4 修正：从 a 出发） =================

def order_chain(a, b):
    """序关系推导（v4 修正版）

    ⚠️ v3 的 bug（V3_SUMMARY 第三节·问题2）：
      v3 用 lo,hi = min,max 起链 → 模型学到"两数不同就可达"→ ⊥ 全错
      （`5503<5497` 自动从 5497 数到 5503 答 ⊤）

    v4 修法：尝试链**必须从 a 出发**（这正是定义的语义），
      并显式判断方向（b<a 时"只会越来越大"→ 不可达）。
    """
    lines = [
        f"问: {a}<{b}?",
        f"理解: 命题 P, P := ({a}<{b})",
        "算法: < 是后继 s 的传递闭包：a<b ⟺ ∃n≥1, sⁿ(a)=b（从 a 出发逐次后继）",
    ]
    if a < b:
        steps = b - a
        parts = []
        cur = a
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
        # b < a：从 a 出发只会越来越大 —— 显示前几步即足以判定
        show = min(4, a - b)
        parts = []
        cur = a
        for _ in range(show):
            parts.append(f"s({cur})={cur + 1}")
            cur += 1
        lines.append("尝试: " + ", ".join(parts) + "（继续只会更大）")
        lines.append(f"判断: 从 a={a} 出发单调递增，无法到达更小的 b={b}")
        lines.append("答: ⊥")
    return "\n".join(lines)


# ================= 数据组装 =================
texts = []


def add(lst, label):
    texts.extend(lst)
    print(f"  {label:<54} {len(lst):>7} 条")


print(f"[挖洞] 末位 {HO} 的 1~{len(str(W))} 位数实例将全部删除（公理保留）")

# [D] 定义层
defs = [
    "自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}",
    "后继 Successor: s(a) = a+1",
    "加法 Addition: a+0 = a, a+s(b) = s(a+b)",
    "序 Order: a<b ⟺ ∃n≥1, sⁿ(a)=b（< 是后继 s 的传递闭包）",
    "位数 Digits: n 的位数 = 其十进制写法中数字的个数",
    "0 是自然数: 0∈ℕ",
    "自然数的后继是自然数: a∈ℕ ⟹ s(a)∈ℕ",
]
add(defs * 35, "[D] 定义层（双语+符号，+位数定义）")

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
axioms.extend(axiom_texts * 30)
for _ in range(2500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: s({a}) = {a}+1 = {a + 1}")
for _ in range(1500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: {a}+0 = {a}")
add(axioms, "[A] 公理层（代数 ×30 + 实例化 4000）")

# ================= 【1】[R4] 数位数层（v4 核心） =================
r4 = []


def count_digits_bucketed():
    """按位宽分桶，1~4 位各均衡（每桶 ~450）"""
    per_width = 450
    out = []
    for w in range(1, len(str(W)) + 1):
        lo, hi = (0 if w == 1 else 10 ** (w - 1)), 10 ** w - 1
        pool = [n for n in range(lo, min(hi, W) + 1) if not is_held(n)]
        if pool:
            out += rng.sample(pool, min(per_width, len(pool)))
    return out


for n in count_digits_bucketed():
    r4.append(count_digits(n))
# 关键锚点（确定性：每个位宽 + 特殊形态）
for n in (0, 1, 9, 10, 99, 100, 999, 1000, 9999, 5000, 1234, 1010, 4003):
    r4.append(count_digits(n))
# 与加法/后继的桥接样本（"位数"→"循环上界"的显式联系）
for _ in range(600):
    a = rng.randint(0, min(W, 999))
    if is_held(a):
        continue
    k = len(str(a))
    r4.append(f"问: {a} 是几位数?\n理解: 数列 L := {seg_str(a)}\n"
              f"计数: " + ", ".join(f"i={i}({d})" for i, d in
                                    enumerate([int(c) for c in str(a)], 1)) +
              f"\n结束: 数列已取完\n答: {k}\n用途: 故加法循环 i 从 1 到 {k}")
add(r4, "[R4] 数位数层（位宽分桶 + 锚点 + 桥接）")

# ================= [R1] 后继（分层枚举：位宽 × 进位链长） =================
r1 = []
succ_buckets = enumerate_succ_buckets(len(str(W)))
succ_picks = balanced_pick(succ_buckets, 3600, rng)
for n in succ_picks:
    r1.append(loop_succ(n))
# 连续进位锚点（难结构显式保证）
for n in (9, 99, 999, 9999, 1999, 4999, 8999, 9099, 9909):
    for _ in range(4):
        r1.append(loop_succ(n))
# 复合后继（组合泛化）
for _ in range(1800):
    a = rng.randint(0, 300)
    if any(is_held(a + j) for j in range(3)):
        continue
    r1.append(f"问: s(s({a})) = ?\n理解: 算式 F, F := s(s(x)), x := {seg_str(a)}\n"
              f"算法: 后继公理两次\n第一步: s({a}) = {a + 1}\n"
              f"第二步: s({a + 1}) = {a + 2}\n答: {a + 2}")
add(r1, "[R1] 后继（分层枚举 + 进位锚点 + 复合）")

# ================= [R2] 加法（分层枚举） =================
r2 = []
add_buckets = enumerate_add_buckets(3)          # 限制 a,b ≤3 位以控桶数（4 位靠锚点+随机补）
add_picks = balanced_pick(add_buckets, 3000, rng)
for a, b in add_picks:
    r2.append(loop_add(a, b))
# 4 位加法（v3 的主力宽度，随机补 + 分层锚点）
cnt4 = 0
while cnt4 < 900:
    a = rng.randint(1000, W)
    b = rng.randint(100, W)
    if a + b > W + 10:
        continue
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    r2.append(loop_add(a, b))
    cnt4 += 1
# 进位链锚点（含补位溢出）
for a, b in ((999, 1), (9999, 1), (4999, 1), (1999, 1), (99, 1), (9, 1),
             (9990, 9), (9909, 90), (990, 10), (5000, 5000)):
    for _ in range(4):
        r2.append(loop_add(a, b))
add(r2, "[R2] 加法（分层枚举 + 4位补 + 进位锚点）")

# ================= 【2】[R3] 序关系（三态 × 方向修正） =================
r3 = []
# 三态均衡 + 差值 ≤10（用户"先学走再学跑"：公理 O(差值) 是本义）
for _ in range(4200):
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
# 方向锚点（确定性：确保 ⊥ 的三种来源都被覆盖）
for a, b in ((3, 4), (4, 3), (4, 4), (1, 2), (2, 1), (1, 11), (11, 1),
             (500, 505), (505, 500), (500, 500), (9999, 9998), (9998, 9999)):
    for _ in range(4):
        r3.append(order_chain(a, b))
add(r3, "[R3] 序关系（三态 + 从 a 出发修正 + 方向锚点）")

# ================= [L] 对齐层 =================
l1 = []
cn_variants = [
    lambda a, b: f"{a} 的后继是 {b}。",
    lambda a, b: f"{a} 的下一个数是 {b}。",
    lambda a, b: f"{a} 加一是 {b}。",
    lambda a, b: f"{a} 再数一个就是 {b}。",
]
for _ in range(2200):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l1.append(rng.choice(cn_variants)(a, a + 1))
for _ in range(1200):
    a = rng.randint(0, 500)
    b = rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l1.append(f"{a} 加 {b} 等于 {a + b}。")
for _ in range(900):
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
# 位数中文表述
for _ in range(700):
    n = rng.randint(0, W)
    if is_held(n):
        continue
    k = len(str(n))
    l1.append(f"{n} 是 {k} 位数。")
add(l1, "[L1] 中文界面（多表述 + 序三态 + 位数表述）")

l2 = []
en_variants = [
    lambda a, b: f"The successor of {a} is {b}.",
    lambda a, b: f"The next number after {a} is {b}.",
    lambda a, b: f"{a} plus one is {b}.",
]
for _ in range(1500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l2.append(rng.choice(en_variants)(a, a + 1))
for _ in range(700):
    a = rng.randint(0, 500)
    b = rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l2.append(f"{a} plus {b} equals {a + b}.")
for _ in range(500):
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
# 位数英文表述
for _ in range(500):
    n = rng.randint(0, W)
    if is_held(n):
        continue
    k = len(str(n))
    l2.append(f"{n} has {k} digit(s).")
add(l2, "[L2] 英文界面（序三态 + 位数表述）")

l3 = []
for _ in range(1400):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    b = a + 1
    trio = [rng.choice(cn_variants)(a, b), rng.choice(en_variants)(a, b),
            f"s({a}) = {b}"]
    rng.shuffle(trio)
    l3.append("\n".join(trio))
add(l3, "[L3] 三形式对齐配对（随机排列）")

# ================= [N] 读法层 =================
n_texts = []
for _ in range(1500):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
for _ in range(700):
    n = rng.randint(10000, 10 ** 8)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
for _ in range(250):
    n = rng.randint(10 ** 8, 10 ** 9)
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
for n in (10, 110, 4003, 10000, 10001, 1000000, 4000300, 100000001,
          12345678, 1010):
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
for _ in range(900):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")
for _ in range(450):
    n = rng.randint(10000, 10 ** 9)
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")
for n in (100, 110, 1000, 4003, 10 ** 6, 1000001):
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")
for _ in range(600):
    n = rng.randint(1, 9999) * 1000
    s = cn_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(450):
    n = rng.randint(1, 9999) * 10000
    s = cn_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(350):
    n = rng.randint(1, 999) * 1000
    s = en_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(180):
    n = rng.randint(1, 99) * 10 ** 6
    s = en_abbr(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for _ in range(250):
    n = rng.randint(1, 9999) * 10000
    s = w_read(n)
    if s:
        n_texts += [f"{s} = {n}", f"{n} = {s}"]
for s, n in (("5千", 5000), ("2k", 2000), ("3w", 30000), ("1.5万", 15000),
             ("1M", 1000000), ("10k", 10000), ("1亿", 10 ** 8)):
    n_texts += [f"{s} = {n}", f"{n} = {s}"]
add(n_texts, "[N] 读法层（中文/英文/缩写 ↔ 数值）")

# ================= 确定性锚点（自检须命中） =================
for _ in range(4):
    texts.append(count_digits(97443))       # 5 位数（外推锚点，格式示范）
    texts.append(count_digits(4483))
    texts.append(count_digits(999))
    texts.append(count_digits(12345678))    # 8 位数
    texts.append(loop_succ(4483))
    texts.append(loop_succ(9999))
    texts.append(loop_add(4483, 579))
    texts.append(order_chain(3, 4))         # ⊤
    texts.append(order_chain(4, 3))         # ⊥（方向）
    texts.append(order_chain(4, 4))         # ⊥（相等）

# ================= 泄漏检查（挖洞协议） =================
print("=" * 72)


def leak_variants(a):
    b = a + 1
    k = len(str(a))
    cnt = ", ".join(f"i={i}({d})" for i, d in enumerate(str(a), 1))
    return [
        loop_succ(a),
        count_digits(a),
        f"公理实例: s({a}) = {a}+1 = {b}",
        f"{a} 的后继是 {b}。",
        f"{a} 的下一个数是 {b}。",
        f"{a} 加一是 {b}。",
        f"{a} 再数一个就是 {b}。",
        f"The successor of {a} is {b}.",
        f"The next number after {a} is {b}.",
        f"{a} plus one is {b}.",
        f"s({a}) = {b}",
        f"{a} 小于 {b}。",
        f"{a} is less than {b}.",
        f"{a} 大于 {a - 1}。",
        f"{a} is greater than {a - 1}.",
        f"{a} 等于 {a}。",
        f"{a} is equal to {a}。",
        f"{a} 是 {k} 位数。",
        f"{a} has {k} digit(s).",
        order_chain(a, b),
        f"问: {a} 是几位数?\n理解: 数列 L := {seg_str(a)}\n计数: {cnt}\n结束: 数列已取完\n答: {k}",
        f"问: {a} 是几位数?\n理解: 数列 L := {seg_str(a)}\n计数: {cnt}\n"
        f"结束: 数列已取完\n答: {k}\n用途: 故加法循环 i 从 1 到 {k}",
    ]


n_leak = 0
text_set = set(texts)
for n in range(0, W + 1):
    if n % 10 == HO:
        for v in leak_variants(n):
            if v is not None and v in text_set:
                n_leak += 1
                print(f"  ⚠️ 泄漏: {v[:60]}")
print(f"[泄漏检查] {n_leak} 条（应为 0）")

# ================= 写盘 =================
rng.shuffle(texts)
if len(texts) > args.total:
    texts = texts[:args.total]
with open(args.out, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print("=" * 72)
print(f"总计 {len(texts)} 条 -> {args.out}")

# ================= 自检三件套 =================
import re
S = set(texts)
checks = [
    (count_digits(97443), "数位数（5 位，外推锚点）"),
    (count_digits(4483), "数位数（4 位）"),
    (count_digits(999), "数位数（3 位）"),
    (loop_succ(4483), "后继循环"),
    (loop_succ(9999), "后继循环（连续进位）"),
    (loop_add(4483, 579), "加法循环"),
    (order_chain(3, 4), "序 ⊤（a<b）"),
    (order_chain(4, 3), "序 ⊥（a>b，方向修正）"),
    (order_chain(4, 4), "序 ⊥（a=b）"),
    ("公理: ∀a∈ℕ. s(a) = a+1", "公理代数形式"),
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
    # 数位数正确性
    # ⚠️ 首版 bug：用 `答: (\d+)$` 锚定末尾，但桥接样本在答案后还有
    #    `用途: 故加法循环 i 从 1 到 k` → 误报 550 条"错误"。改不锚定末尾。
    m = re.match(r"^问: (\d+) 是几位数\?", t)
    if m:
        a = int(m.group(1))
        ma = re.search(r"答: (\d+)", t)
        if not ma or int(ma.group(1)) != len(str(a)):
            n_bad += 1
            print(f"   BAD digits: a={a} 期望 {len(str(a))} 得 {ma.group(1) if ma else None}")
        n_checked += 1
    # 后继
    m = re.match(r"^问: s\((\d+)\) = \?", t)
    if m:
        a = int(m.group(1))
        ma = re.search(r"答: (\d+)$", t)
        if not ma or int(ma.group(1)) != a + 1:
            n_bad += 1
            print(f"   BAD succ: {t[:70]}")
        n_checked += 1
    # 加法
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ma = re.search(r"答: (\d+)$", t)
        if not ma or int(ma.group(1)) != a + b:
            n_bad += 1
            print(f"   BAD add: {t[:70]}")
        ms = re.search(r"合成: ([\d,]+)", t)
        if ms and ma and ms.group(1).replace(",", "") != ma.group(1):
            n_bad += 1
            print(f"   BAD synth: {t[:70]}")
        n_checked += 1
    # 序关系（三态 + 方向）
    m = re.match(r"^问: (\d+)<(\d+)\?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ma = re.search(r"答: (\S+)$", t)
        expect = "⊤" if a < b else "⊥"
        if not ma or ma.group(1) != expect:
            n_bad += 1
            print(f"   BAD order: a={a} b={b} 期望 {expect} 得 {ma.group(1) if ma else None}")
        n_checked += 1
print(f"\n抽检: {n_checked} 条中 {n_bad} 条错误")

# 序三态分布 + 数位数位宽分布
n_t = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊤"))
n_f = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊥"))
print(f"[序三态分布] ⊤ {n_t} / ⊥ {n_f}（都须 >0）")
nd = sum(1 for t in texts if re.match(r"^问: \d+ 是几位数\?", t))
print(f"[数位数层] {nd} 条")
if n_t == 0 or n_f == 0 or nd < 100:
    n_bad += 1
    print("   ⚠️ 分布异常")

# 各层统计
print("\n[分层抽样统计]")
from collections import Counter
w_dist = Counter()
for t in texts:
    m = re.match(r"^问: s\((\d+)\) = \?", t)
    if m:
        w_dist[("后继", len(m.group(1)))] += 1
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        w_dist[("加法", max(len(m.group(1)), len(m.group(2))))] += 1
for k in sorted(w_dist):
    print(f"   {k[0]} {k[1]} 位: {w_dist[k]} 条")

if n_miss or n_leak or n_bad:
    print(f"\n⚠️ 异常：样例缺失 {n_miss} / 泄漏 {n_leak} / 错误 {n_bad}")
    sys.exit(1)
print("\n✅ 全部就位，可训练")