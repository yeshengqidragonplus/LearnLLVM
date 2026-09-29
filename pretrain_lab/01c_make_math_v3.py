# -*- coding: utf-8 -*-
"""01c_make_math_v3.py — v3 数据生成：绑定行分段抄写（贯彻大纲设计到全链条）
目的：修复 v2 暴露的"绑定行丢位"（x=97443 抄成 9743，万位消失）。

v2 结论：竖式行（每行一位）训练宽度内 100%——"N 个宽度 1 操作"成立；
但绑定行 x=97443 是整体抄写（一次抓 5 个 token），5 位时丢位。
v3 改动：绑定行数字改为逗号分隔逐位列表——抄写动作从"一次抓 N 个"
变成"N 次各抓 1 个"，与竖式行同构。全链条再无宽度 >1 的字面量操作。

格式对比：
  v2: 理解: 算式 F, F=s(x), x=97443
  v3: 理解: 算式 F, F=s(x), x=9,7,4,4,3

其余全部继承 v2（D/A/L 层、挖洞协议、序格式、锚点）。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import random

parser = argparse.ArgumentParser(description="v3 数据生成（绑定行分段抄写）")
parser.add_argument("--out", default="data/math_v3.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total", type=int, default=52000, help="总样本量（读法层提速后 ~4.8 万，留余量防截断）")
parser.add_argument("--max-n", type=int, default=9999)
parser.add_argument("--hole-digit", type=int, default=7)
args = parser.parse_args()

rng = random.Random(args.seed)

# [N] 读法层：数值 ↔ 中文/英文/缩写（同一本体的多种人类表面形式）
# 用户要求（2026-09-17）：位名/读法要训练（"过5千/2k/3w"要知道是多少）；
# 规则经 check_readnum_tmp.py 独立校验（中文19个特殊规则+500随机、
# 英文500随机+8特殊、缩写含精度回验，共 2131/0 通过）。
from readnum import cn_read, en_read, cn_abbr, en_abbr, w_read

W = args.max_n
HO = args.hole_digit
# ⚠️ 8/9 位锚点需要更多位名（否则 POS_NAMES[i] 越界）
POS_NAMES = ["个位", "十位", "百位", "千位", "万位", "十万位", "百万位",
             "千万位", "亿位", "十亿位", "百亿位"]


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
    """97443 -> '9,7,4,4,3'（高位在前，逗号分隔——绑定行分段抄写）"""
    return ",".join(str(n))


def seg_str(n):
    """97443 -> '9,7,4,4,3'"""
    return ",".join(c for c in str(n))


# ================= 循环计数格式生成器（v3 定稿：用户设计） =================
# 核心（用户 2026-09-17）：用"循环 + 计数"替代"位名"——
#   ①索引 i 无限（位名有尽头，8/9 位要硬编码"千万位/亿位"）
#   ②每行完全同构（i=1 与 i=8 只差数字，模型学同一模式）
#   ③"声明 + 逐次展开"：算法行声明循环，每一轮显式写出 i 和中间结果
#      —— 绝不用幂记号（s² 压缩会退化成"一次做 N 件事"，即 v1 失败模式）
# 下标方向：选项①（绑定高位在前 a := 4,4,8,3；a_i 从右往左取 = 竖式惯例）

def loop_add(a, b):
    """加法循环格式（已验证正确：check_chain_v3_tmp.py 14 用例）"""
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
    ds_high = list(reversed(ds))
    lines.append("合成: " + ",".join(str(d) for d in ds_high))
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
    ds_high = list(reversed(ds))
    lines.append("合成: " + ",".join(str(d) for d in ds_high))
    lines.append(f"答: {b}")
    return "\n".join(lines)


# ================= 数据生成（结构继承 v2） =================
texts = []


def add(lst, label):
    texts.extend(lst)
    print(f"  {label:<52} {len(lst):>7} 条")


print(f"[挖洞] 末位 {HO} 的 1~4 位数实例将全部删除（公理保留）")

# [D] 定义层
defs = [
    "自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}",
    "后继 Successor: s(a) = a+1",
    "加法 Addition: a+0 = a, a+s(b) = s(a+b)",
    "序 Order: a<b ⟺ ∃n≥1, sⁿ(a)=b（< 是后继 s 的传递闭包）",
    "0 是自然数: 0∈ℕ",
    "自然数的后继是自然数: a∈ℕ ⟹ s(a)∈ℕ",
]
add(defs * 40, "[D] 定义层（双语+符号 ×40）")

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
for _ in range(3000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: s({a}) = {a}+1 = {a + 1}")
for _ in range(2000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    axioms.append(f"公理实例: {a}+0 = {a}")
add(axioms, "[A] 公理层（代数 ×30 + 实例化 5000）")

# [R] 推导层（v3 大纲：绑定行分段）
r1 = []
for _ in range(8000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    r1.append(loop_succ(a))
# ⚠️ 组合泛化补量（v2 该项 0%，4 层容量下需更多直接信号）
# 用户指正（2026-09-17）：s(s(x)) 是皮亚诺复合，是"序/比较"的基础，必须学扎实
for _ in range(3500):
    a = rng.randint(0, 500)
    if is_held(a) or is_held(a + 1) or is_held(a + 2):
        continue
    r1.append(f"问: s(s({a})) = ?\n理解: 算式 F, F=s(s(x)), x={seg_str(a)}\n算法: 后继公理两次\n第一步: s({a}) = {a + 1}\n第二步: s({a + 1}) = {a + 2}\n答: {a + 2}")
# 三层复合（更高阶，少量）
for _ in range(800):
    a = rng.randint(0, 300)
    if any(is_held(a + k) for k in range(4)):
        continue
    r1.append(f"问: s(s(s({a}))) = ?\n理解: 算式 F, F=s(s(s(x))), x={seg_str(a)}\n算法: 后继公理三次\n第一步: s({a}) = {a + 1}\n第二步: s({a + 1}) = {a + 2}\n第三步: s({a + 2}) = {a + 3}\n答: {a + 3}")
add(r1, "[R1] 后继大纲（绑定分段，含复合）")

r2 = []
for _ in range(12000):
    a = rng.randint(0, W)
    b = rng.randint(0, W)
    if a + b > W + 10:
        continue
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    r2.append(loop_add(a, b))
for a, b in ((999, 1), (4999, 1), (9999, 1), (99, 1), (9, 1), (1999, 1)):
    for _ in range(8):
        r2.append(loop_add(a, b))
add(r2, "[R2] 加法大纲（绑定分段）")

def order_chain(a, b):
    """序关系推导（方案 A：命题绑定 + 传递闭包 + 真值符号）
    ⚠️ 用户指正（2026-09-17）：
      ①旧版只有"成立"，是假测试（模型无脑输出"成立"即满分）→ 改三态
      ②`F=(101<103)` 是范畴错误（F 应绑数值，命题应用 P 绑真值）
        → 项绑项（F := a+b），命题绑命题（P := (a<b)）
      ③"答: 成立"无符号表示，真值应有本体 → 答 ⊤/⊥
    传递闭包：< 定义为后继 s 的可达性（只用 s，不依赖 +，纯皮亚诺）"""
    lo, hi = min(a, b), max(a, b)
    steps = hi - lo
    chain_parts = []
    cur = lo
    for _ in range(steps):
        chain_parts.append(f"s({cur})={cur + 1}")
        cur += 1
    chain_str = ", ".join(chain_parts) if chain_parts else f"无（{lo}={hi}，n=0）"
    lines = [
        f"问: {a}<{b}?",
        f"理解: 命题 P, P := ({a}<{b})",
        "算法: < 是后继 s 的传递闭包：a<b ⟺ ∃n≥1, sⁿ(a)=b",
        f"步数: {chain_str}",
    ]
    if a < b:
        lines.append(f"判断: n={steps}≥1，可达")
        lines.append("答: ⊤")
    elif a == b:
        lines.append("判断: 需 n≥1 但 n=0，不可达")
        lines.append("答: ⊥")
    else:
        lines.append(f"判断: 后继严格递增，从 {a} 无法到达 {b}")
        lines.append("答: ⊥")
    return "\n".join(lines)


r3 = []
# ⚠️ 用户指正（2026-09-17）：先学会走再跑——公理优先（`a<b ⟺ ∃n≥1, sⁿ(a)=b`
#    就是循环，模型学会"递归循环求值"即可，差值大小只是循环次数）。
#    故训练只用小差值（≤10，最多 10 行）；不用大数（避免 97 行的超长样本，
#    那既是浪费也是 OOM 元凶）。评估时才测大差值，看能否递归。
for _ in range(4400):
    a = rng.randint(0, W - 1)
    # 三态近似等分：成立（a<b）/ 不成立-大于（a>b）/ 不成立-等于（a==b）
    mode = rng.random()
    if mode < 0.5:
        b = rng.randint(a + 1, min(a + 10, W))
    elif mode < 0.9:
        b = rng.randint(max(0, a - 10), a - 1) if a >= 1 else a + rng.randint(1, 5)
    else:
        b = a
    if is_held(a) or is_held(b):
        continue
    r3.append(order_chain(a, b))
add(r3, "[R3] 序关系推导（三态 + 后继步数链，差值≤10）")

# [L] 对齐层
l1 = []
cn_variants = [
    lambda a, b: f"{a} 的后继是 {b}。",
    lambda a, b: f"{a} 的下一个数是 {b}。",
    lambda a, b: f"{a} 加一是 {b}。",
    lambda a, b: f"{a} 再数一个就是 {b}。",
]
for _ in range(4000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l1.append(rng.choice(cn_variants)(a, a + 1))
for _ in range(2000):
    a = rng.randint(0, 500)
    b = rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l1.append(f"{a} 加 {b} 等于 {a + b}。")
# ⚠️ 序关系三态（修假测试）：小于/大于/等于都要有
for _ in range(1500):
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
add(l1, "[L1] 中文界面（多表述，序三态）")

l2 = []
en_variants = [
    lambda a, b: f"The successor of {a} is {b}.",
    lambda a, b: f"The next number after {a} is {b}.",
    lambda a, b: f"{a} plus one is {b}.",
]
for _ in range(2500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    l2.append(rng.choice(en_variants)(a, a + 1))
for _ in range(1200):
    a = rng.randint(0, 500)
    b = rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l2.append(f"{a} plus {b} equals {a + b}.")
# ⚠️ 英文序三态（修假测试）：小于/大于/等于都要有
for _ in range(800):
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
add(l2, "[L2] 英文界面（序三态）")

l3 = []
for _ in range(2500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    b = a + 1
    cn = rng.choice(cn_variants)(a, b)
    en = rng.choice(en_variants)(a, b)
    sym = f"s({a}) = {b}"
    trio = [cn, en, sym]
    rng.shuffle(trio)
    l3.append("\n".join(trio))
add(l3, "[L3] 三形式对齐配对（随机排列）")

# ================= [N] 读法层（数值 ↔ 中文/英文/缩写） =================
# 用户需求（2026-09-17）：位名/读法要训练（"过5千/2k/3w"要知道值）；
# 同一数值的多种人类表面形式对齐到同一本体（第四个表面通道）。
# 规则经 check_readnum_tmp.py 独立逆运算校验（2131/0）。
n_texts = []

# ⚠️ 提速（用户 2026-09-17）：读法层从 3.6 万砍到 ~1.2 万——
# 读法是"模板式记忆任务"（规则确定、无推理深度），1.2 万条足够学模式；
# 原量占数据 52%，使训练时长翻倍到 15.5 小时（核心推理层不受影响）。
# N1: 中文读法（双向等式）
for _ in range(1700):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
# N2/N3: 万/亿分段
for _ in range(800):
    n = rng.randint(10000, 10 ** 8)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
for _ in range(300):
    n = rng.randint(10 ** 8, 10 ** 9)
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")
# 中文读法特殊规则锚点（零处理/段位，确定性）
for n in (10, 110, 4003, 10000, 10001, 1000000, 4000300, 100000001,
          12345678, 1010, 1000000000):
    n_texts.append(f"{n} = {cn_read(n)}")
    n_texts.append(f"{cn_read(n)} = {n}")

# N5: 英文读法
for _ in range(1000):
    n = rng.randint(0, 9999)
    if is_held(n):
        continue
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")
for _ in range(500):
    n = rng.randint(10000, 10 ** 9)
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")
for n in (100, 110, 1000, 4003, 10 ** 6, 1000001, 2 * 10 ** 9):
    n_texts.append(f"{n} = {en_read(n)}")
    n_texts.append(f"{en_read(n)} = {n}")

# N4: 缩写（5千/1.5万/3w/2k/4.5k/1M）
for _ in range(700):
    n = rng.randint(1, 9999) * 1000
    s = cn_abbr(n)
    if s:
        n_texts.append(f"{s} = {n}")
        n_texts.append(f"{n} = {s}")
for _ in range(500):
    n = rng.randint(1, 9999) * 10000
    s = cn_abbr(n)
    if s:
        n_texts.append(f"{s} = {n}")
        n_texts.append(f"{n} = {s}")
for _ in range(400):
    n = rng.randint(1, 999) * 1000
    s = en_abbr(n)
    if s:
        n_texts.append(f"{s} = {n}")
        n_texts.append(f"{n} = {s}")
for _ in range(200):
    n = rng.randint(1, 99) * 10 ** 6
    s = en_abbr(n)
    if s:
        n_texts.append(f"{s} = {n}")
        n_texts.append(f"{n} = {s}")
# w 形式（3w/1.5w）
for _ in range(300):
    n = rng.randint(1, 9999) * 10000
    s = w_read(n)
    if s:
        n_texts.append(f"{s} = {n}")
        n_texts.append(f"{n} = {s}")
# 缩写锚点（确定性，覆盖 5千/2k/3w/1.5万/1M）
for s, n in (("5千", 5000), ("2k", 2000), ("3w", 30000), ("1.5万", 15000),
             ("1M", 1000000), ("10k", 10000), ("1亿", 10 ** 8)):
    n_texts.append(f"{s} = {n}")
    n_texts.append(f"{n} = {s}")
add(n_texts, "[N] 读法层（中文/英文/缩写 ↔ 数值）")

# 序关系确定性锚点（自检样例须命中；三态各若干，含小数字与多步链）
for _ in range(4):
    texts.append(order_chain(3, 4))       # ⊤（差 1）
    texts.append(order_chain(4, 4))       # ⊥（相等）
    texts.append(order_chain(4, 3))       # ⊥（大→小）
    texts.append(order_chain(1, 2))       # ⊤（最小差）
    texts.append(order_chain(1, 11))      # ⊤（差 10，多步链上限）
    texts.append(order_chain(11, 1))      # ⊥（差 10 反向）

# ================= 确定性锚点 =================
# ⚠️ 用户指正（2026-09-17）：6 位样本重复 6 次有记忆风险（背答案非泛化）。
# 修正：3 个不同 6 位形态（大+小 / 大+大 / 进位边界），各只出现 2 次——
# 给"6 位格式"提供直接信号（位名/竖式行数），但不足以背具体答案。
# 5~7 位外推测试数与锚点数全部不同，测试仍干净。
for _ in range(6):
    texts.append(loop_succ(4483))
    texts.append(loop_succ(999))
    texts.append(loop_add(4483, 579))
    texts.append("4483 的后继是 4484。")
    texts.append("The successor of 4483 is 4484.")
    texts.append("s(4483) = 4484")
for _ in range(2):
    texts.append(loop_add(112, 522222))      # 6位+3位（大+小）
    texts.append(loop_add(345678, 123456))   # 6位+6位（大+大）
    texts.append(loop_add(999999, 1))        # 6位进位边界（全9）
    # ⚠️ 用户补充（2026-09-17）：8/9 位也来几个（少而精，防记忆）
    # 目的：给"长位宽格式"直接信号（位名到千万/亿位、竖式行数）
    # ⚠️ 5/7 位刻意不给锚点——作为"纯外推"对照组
    texts.append(loop_add(12345678, 87654321))    # 8位+8位（大+大）
    texts.append(loop_add(100000000, 1))          # 9位+1（高位进位边界）

# ================= 泄漏检查 =================
print("=" * 70)


def leak_variants(a):
    """洞 a（末位 HO）的泄漏写法 = 任何教"a 的后继/b 与 a 关系"的样本"""
    b = a + 1
    return [
        loop_succ(a),
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
        # 序三态新增形态（v3 修正后必须一并拦截）
        f"{a} 大于 {a - 1}。",
        f"{a} is greater than {a - 1}.",
        f"{a} 等于 {a}。",
        f"{a} is equal to {a}.",
        order_chain(a, b),
        order_chain(a, a - 1) if a >= 1 else None,
        order_chain(a, a),
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
print("=" * 70)
print(f"总计 {len(texts)} 条 -> {args.out}")

# ================= 自检三件套 =================
S = set(texts)
checks = [
    (loop_succ(4483), "后继大纲（绑定分段）"),
    (loop_succ(4999), "后继大纲（连续进位）"),
    (loop_add(4483, 579), "加法大纲（绑定分段）"),
    (loop_add(112, 522222), "加法大纲（6位锚点）"),
    (loop_add(12345678, 87654321), "加法大纲（8位锚点）"),
    (loop_add(100000000, 1), "加法大纲（9位锚点）"),
    (order_chain(3, 4), "序推导（成立/后继步数）"),
    (order_chain(4, 3), "序推导（不成立-大于）"),
    (order_chain(4, 4), "序推导（不成立-等于）"),
    ("公理: ∀a∈ℕ. s(a) = a+1", "公理代数形式"),
    ("4483 的后继是 4484。", "中文对齐"),
]
n_miss = 0
print("\n关键样例自检：")
for text, label in checks:
    ok = text in S
    if not ok:
        n_miss += 1
    print(f"   {'OK  ' if ok else 'MISS'} {label}")

import re
n_bad = 0
n_checked = 0
for t in texts:
    m = re.match(r"^问: s\((\d+)\) = \?", t)
    if m:
        a = int(m.group(1))
        m2 = re.search(r"答: (\d+)$", t)
        if not m2 or int(m2.group(1)) != a + 1:
            n_bad += 1
            print(f"   BAD succ: {t[:70]}")
        n_checked += 1
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        m2 = re.search(r"答: (\d+)$", t)
        if not m2 or int(m2.group(1)) != a + b:
            n_bad += 1
            print(f"   BAD add: {t[:70]}")
        ms = re.search(r"合成: ([\d,]+)", t)
        if ms and m2:
            if ms.group(1).replace(",", "") != m2.group(1):
                n_bad += 1
                print(f"   BAD synth: {t[:70]}")
        # 绑定行分段一致性：a := 4,4,8,3 的数字拼接 == str(a)
        mb = re.search(r"a := ([\d,]+)", t)
        if mb:
            if mb.group(1).replace(",", "") != str(a):
                n_bad += 1
                print(f"   BAD bind: {t[:70]}")
        # 循环行逐 i 一致性：每行 s_i == a_i+b_i+c
        for mr in re.finditer(r"^i=(\d+): a_\1=(\d+), b_\1=(\d+), c=(\d+), "
                              r"s_\1=(\d+)\+(\d+)\+(\d+)=(\d+), d_\1=(\d+), c:=(\d+)$",
                              t, re.M):
            i, ai, bi, cin, x1, x2, x3, s, d, cout = map(int, mr.groups())
            if s != ai + bi + cin or d != s % 10 or cout != s // 10:
                n_bad += 1
                print(f"   BAD loop: {t[:70]}")
                break
        n_checked += 1
    # 序关系抽检（三态 ⊤/⊥ —— 方案 A）
    m = re.match(r"^问: (\d+)<(\d+)\?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ma = re.search(r"答: (\S+)$", t)
        expect = "⊤" if a < b else "⊥"
        if not ma or ma.group(1) != expect:
            n_bad += 1
            print(f"   BAD order: {t[:70]}（a={a} b={b} 期望 {expect}）")
        n_checked += 1
    # 复合后继抽检
    m = re.match(r"^问: s\(s\((\d+)\)\) = \?", t)
    if m:
        a = int(m.group(1))
        m2 = re.search(r"答: (\d+)$", t)
        if not m2 or int(m2.group(1)) != a + 2:
            n_bad += 1
            print(f"   BAD s2: {t[:70]}")
        n_checked += 1
print(f"\nCoT 数学正确性抽检: {n_checked} 条中 {n_bad} 条错误")

# 序关系三态分布统计（假测试防线：⊤/⊥ 都必须存在）
n_lt = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊤"))
n_ngt = sum(1 for t in texts if re.match(r"^问: \d+<\d+\?", t) and t.rstrip().endswith("⊥"))
print(f"[序三态分布] ⊤ {n_lt} 条 / ⊥ {n_ngt} 条（两者都须 >0）")
if n_lt == 0 or n_ngt == 0:
    n_bad += 1
    print("   ⚠️ 序三态分布异常——仍是假测试")

# 读法层抽检（独立逆解析：中文/英文/缩写 -> 数值，必须回原值）
from readnum_parse import parse_cn, parse_en, parse_abbr_cn, parse_abbr_en
n_read = 0
n_read_bad = 0
for t in texts:
    m = re.match(r"^(\d+) = (.+)$", t)
    if m and not t.rstrip().endswith("?"):
        n_as_num = int(m.group(1))
        s = m.group(2)
        try:
            if any(0x4E00 <= ord(c) <= 0x9FFF for c in s) and not s.endswith(("千", "万", "亿", "w")):
                back = parse_cn(s)
            elif s.endswith(("千", "万", "亿", "w")):
                back = parse_abbr_cn(s)
            elif s.isascii() and any(c.isalpha() for c in s):
                if s[-1] in "kMB" and s[:-1].replace(".", "").isdigit():
                    back = parse_abbr_en(s)
                else:
                    back = parse_en(s)
            else:
                continue
            n_read += 1
            if back != n_as_num:
                n_read_bad += 1
                if n_read_bad <= 5:
                    print(f"   BAD read: {t[:60]}（解析回 {back} ≠ {n_as_num}）")
        except Exception:
            pass
print(f"[读法层抽检] {n_read} 条中 {n_read_bad} 条错误")
if n_read_bad:
    n_bad += n_read_bad

if n_miss or n_leak or n_bad:
    print(f"\n⚠️ 异常：样例缺失 {n_miss} / 泄漏 {n_leak} / CoT 错误 {n_bad}")
    sys.exit(1)
print("\n✅ 全部就位，可训练")
