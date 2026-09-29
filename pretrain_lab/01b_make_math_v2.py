# -*- coding: utf-8 -*-
"""01b_make_math_v2.py — v2 数据生成：大纲渐进展开格式（用户设计）
目的：解决 v1 的字面量搬运宽度绑定（五位数丢位 0%）。

核心思想（用户"大纲渐进展开"）：
  数字在推理链中以符号（x/y/a/b）参与——宽度无关（v1 已证 100%）；
  具体计算时逐位展开——每行只碰一个数位（探针 C 验证：零样本 2~5 位 70~90%）。
  把"一个宽度 N 的操作"换成"N 个宽度 1 的操作"。

格式（加法）：
  问: 112+522222 = ?
  理解: 算式 F, F=a+b, a=112, b=522222        ← 绑定层（parse）
  算法: 按位对齐，从个位起相加，满十进一      ← 算法层（非公理！是推导出的程序）
  个位: a=2, b=2, 2+2+0 = 4, 写4进0           ← 展开层（每行宽度 1）
  ...
  合成: 5,2,2,3,3,4                            ← 逗号列表（高位到低位）
  答: 522334                                   ← 本地读出

后继（同构）：F=s(x), x=4483 → 逐位 → 合成 → 答

继承 v1 结构：[D] 定义层 / [A] 公理层 / [R] 推导层（换大纲格式）/ [L] 对齐层
挖洞协议不变：末位 7 的 1~4 位数实例全删（公理保留）。
训练范围：1~4 位（5~7 位留外推测试）。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import random

parser = argparse.ArgumentParser(description="v2 数据生成（大纲渐进展开格式）")
parser.add_argument("--out", default="data/math_v2.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total", type=int, default=50000)
parser.add_argument("--max-n", type=int, default=9999, help="训练数字上限（4 位）")
parser.add_argument("--hole-digit", type=int, default=7)
args = parser.parse_args()

rng = random.Random(args.seed)

W = args.max_n
HO = args.hole_digit
# 位名（低位在前；5~7 位用于外推测试，训练数据不生成但格式支持）
POS_NAMES = ["个位", "十位", "百位", "千位", "万位", "十万位", "百万位"]


def is_held(n):
    return 0 <= n <= W and n % 10 == HO


def digits_low_first(n):
    """n -> 低位在前的数字列表"""
    out = []
    while n >= 10:
        out.append(n % 10)
        n //= 10
    out.append(n)
    return out


# ================= 大纲格式生成器 =================

def outline_add(a, b):
    """加法大纲格式（用户设计：绑定→算法→逐位→合成→答）"""
    c = a + b
    da = digits_low_first(a)
    db = digits_low_first(b)
    k = max(len(da), len(db))
    lines = [
        f"问: {a}+{b} = ?",
        f"理解: 算式 F, F=a+b, a={a}, b={b}",
        "算法: 按位对齐，从个位起相加，满十进一",
    ]
    carry = 0
    res = []
    for i in range(k):
        ai = da[i] if i < len(da) else 0
        bi = db[i] if i < len(db) else 0
        s = ai + bi + carry
        w, carry = s % 10, s // 10
        res.append(w)
        lines.append(f"{POS_NAMES[i]}: a={ai}, b={bi}, {ai}+{bi}+{carry if False else ''}{'' if (ai+bi+carry) == s else ''}{ai}+{bi}+{s - ai - bi} = {s}, 写{w}进{carry}")
    # 修正：进位显示逻辑——上面 f-string 复杂易错，重写为清晰版本
    lines = [
        f"问: {a}+{b} = ?",
        f"理解: 算式 F, F=a+b, a={a}, b={b}",
        "算法: 按位对齐，从个位起相加，满十进一",
    ]
    carry = 0
    res = []
    for i in range(k):
        ai = da[i] if i < len(da) else 0
        bi = db[i] if i < len(db) else 0
        s = ai + bi + carry
        w = s % 10
        new_carry = s // 10
        lines.append(f"{POS_NAMES[i]}: a={ai}, b={bi}, {ai}+{bi}+{carry} = {s}, 写{w}进{new_carry}")
        res.append(w)
        carry = new_carry
    if carry:
        res.append(carry)
        lines.append(f"{POS_NAMES[k]}: a=0, b=0, 0+0+{carry} = {carry}, 写{carry}进0")
    # 合成（高位到低位）
    res_high_first = list(reversed(res))
    lines.append("合成: " + ",".join(str(d) for d in res_high_first))
    lines.append(f"答: {c}")
    return "\n".join(lines)


def outline_succ(a):
    """后继大纲格式（同构：F=s(x)）"""
    b = a + 1
    dx = digits_low_first(a)
    lines = [
        f"问: s({a}) = ?",
        f"理解: 算式 F, F=s(x), x={a}",
        "算法: 后继公理 s(x)=x+1，个位加一，满十进一",
    ]
    carry = 1  # 个位加一
    res = []
    for i in range(len(dx)):
        xi = dx[i]
        s = xi + carry
        w = s % 10
        new_carry = s // 10
        lines.append(f"{POS_NAMES[i]}: x={xi}, {xi}+{carry} = {s}, 写{w}进{new_carry}")
        res.append(w)
        carry = new_carry
    if carry:
        res.append(carry)
        lines.append(f"{POS_NAMES[len(dx)]}: x=0, 0+{carry} = {carry}, 写{carry}进0")
    res_high_first = list(reversed(res))
    lines.append("合成: " + ",".join(str(d) for d in res_high_first))
    lines.append(f"答: {b}")
    return "\n".join(lines)


# ================= 数据生成 =================
texts = []


def add(lst, label):
    texts.extend(lst)
    print(f"  {label:<52} {len(lst):>7} 条")


print(f"[挖洞] 末位 {HO} 的 1~4 位数实例将全部删除（公理保留）")

# [D] 定义层（与 v1 相同——公理体系不变）
defs = [
    "自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}",
    "后继 Successor: s(a) = a+1",
    "加法 Addition: a+0 = a, a+s(b) = s(a+b)",
    "序 Order: a<b ⟺ ∃c∈ℕ, a+c=b",
    "0 是自然数: 0∈ℕ",
    "自然数的后继是自然数: a∈ℕ ⟹ s(a)∈ℕ",
]
add(defs * 40, "[D] 定义层（双语+符号 ×40）")

# [A] 公理层（与 v1 相同）
axioms = []
axiom_texts = [
    "公理: ∀a∈ℕ. s(a) = a+1",
    "公理: ∀a∈ℕ. a+0 = a",
    "公理: ∀a,b∈ℕ. a+s(b) = s(a+b)",
    "公理: ∀a∈ℕ. s(a) ≠ 0",
    "公理: s(a)=s(b) ⟹ a=b",
    "公理: ∀a,b∈ℕ. a<b ⟺ ∃c∈ℕ, a+c=b",
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

# [R] 推导层（大纲格式——v2 核心）
r1 = []
for _ in range(8000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    r1.append(outline_succ(a))
# 复合后继（组合泛化基础）
for _ in range(1500):
    a = rng.randint(0, 200)
    if is_held(a) or is_held(a + 1) or is_held(a + 2):
        continue
    r1.append(f"问: s(s({a})) = ?\n理解: 算式 F, F=s(s(x)), x={a}\n算法: 后继公理两次\n第一步: s({a}) = {a + 1}\n第二步: s({a + 1}) = {a + 2}\n答: {a + 2}")
add(r1, "[R1] 后继大纲（含复合）")

r2 = []
for _ in range(12000):
    a = rng.randint(0, W)
    b = rng.randint(0, W)
    if a + b > W + 10:
        continue
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    r2.append(outline_add(a, b))
# 进位边界锚点（确定性）
for a, b in ((999, 1), (4999, 1), (9999, 1), (99, 1), (9, 1), (1999, 1)):
    for _ in range(8):
        r2.append(outline_add(a, b))
add(r2, "[R2] 加法大纲（绑定→算法→逐位→合成）")

r3 = []
for _ in range(4000):
    a = rng.randint(0, W - 1)
    b = rng.randint(a + 1, min(a + 20, W))
    if is_held(a) or is_held(b):
        continue
    r3.append(f"问: {a}<{b}?\n依据序公理: {a}<{b} ⟺ ∃c∈ℕ, {a}+c={b}\n验证: {a}+{b - a}={b}\n答: 成立")
add(r3, "[R3] 序关系推导（保留 v1 格式）")

# [L] 对齐层（与 v1 相同——三形式配对）
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
for _ in range(1500):
    a = rng.randint(0, W - 1)
    b = rng.randint(a + 1, min(a + 20, W))
    if is_held(a) or is_held(b):
        continue
    l1.append(f"{a} 小于 {b}。")
add(l1, "[L1] 中文界面（多表述）")

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
for _ in range(800):
    a = rng.randint(0, W - 1)
    b = rng.randint(a + 1, min(a + 20, W))
    if is_held(a) or is_held(b):
        continue
    l2.append(f"{a} is less than {b}.")
add(l2, "[L2] 英文界面")

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

# ================= 确定性锚点 =================
for _ in range(6):
    texts.append(outline_succ(4483))
    texts.append(outline_succ(999))
    texts.append(outline_add(4483, 579))
    texts.append(outline_add(112, 522222))   # 6 位锚点（外推格式的直接示范）
    texts.append("4483 的后继是 4484。")
    texts.append("The successor of 4483 is 4484.")
    texts.append("s(4483) = 4484")

# ================= 泄漏检查 =================
print("=" * 70)


def leak_variants(a):
    b = a + 1
    return [
        outline_succ(a),
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
    ]


n_leak = 0
text_set = set(texts)
for n in range(0, W + 1):
    if n % 10 == HO:
        for v in leak_variants(n):
            if v in text_set:
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
    (outline_succ(4483), "后继大纲（4位）"),
    (outline_succ(4999), "后继大纲（连续进位）"),
    (outline_add(4483, 579), "加法大纲（4位含进位）"),
    (outline_add(112, 522222), "加法大纲（6位锚点）"),
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

# CoT 数学正确性抽检（全部 R 层）
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
        n_checked += 1
    # 合成行与竖式行一致性抽检（合成行数字 = 答案数字）
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        ms = re.search(r"合成: ([\d,]+)", t)
        ma = re.search(r"答: (\d+)$", t)
        if ms and ma:
            synth = ms.group(1).replace(",", "")
            if synth != ma.group(1):
                n_bad += 1
                print(f"   BAD synth: {t[:70]}")
        n_checked += 1
print(f"\nCoT 数学正确性抽检: {n_checked} 条中 {n_bad} 条错误")

if n_miss or n_leak or n_bad:
    print(f"\n⚠️ 异常：样例缺失 {n_miss} / 泄漏 {n_leak} / CoT 错误 {n_bad}")
    sys.exit(1)
print("\n✅ 全部就位，可训练")
