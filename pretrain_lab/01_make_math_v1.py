# -*- coding: utf-8 -*-
"""01_make_math_v1.py — 新系列 v1 数据生成：ℕ 公理化数系（枢轴对齐）
目的：全新路线第一版。中/英/符号三形式，数学符号为逻辑本体（枢轴语言），
中英文为交互界面。训练目标 = 潜空间对齐（同一逻辑内容多表面形式配对）。

四层数据（配比：符号 50% / 中文 25% / 英文 15% / 对齐 10%）：
  [D] 定义层：双语+符号定义（ℕ、后继、加法、序）
  [A] 公理层：皮亚诺公理（代数形式 ∀a∈ℕ. s(a)=a+1 + 实例化）
  [R] 推导层：思维链套公理得出结果（核心训练信号）
      - 后继推导：问: s(4483) = ? → 依据后继公理: ... → 答: 4484
      - 加法 CoT：数位意义 → 对齐 → 低位加起 → 满十进位 → 合成（用户设计）
  [L] 对齐层：同内容三形式配对（多中文表述防表面捷径）

挖洞协议（删实例保公理）：末位 7 的数（1~4 位）删除全部实例形态
（推导/对齐/序比较），保留公理文本。测试 s(7777)=? 过 = 真泛化。

继承旧系列教训（见 EXPERIMENTS_SUMMARY.md）：
  1. CoT 答案必须与步骤有 token 级因果链（合成尾巴）
  2. 任务标记防通道污染（问:/答: 标记）
  3. 自检三件套：样例命中（动态生成）/ 泄漏检查 / CoT 数学抽检
  4. BPE 已预检通过（check_tok_v1_tmp.py）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import json
import random

parser = argparse.ArgumentParser(description="v1 数据生成（ℕ 公理化，枢轴对齐）")
parser.add_argument("--out", default="data/math_v1.jsonl")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--total", type=int, default=50000, help="总样本量（小规模验证 5 万）")
parser.add_argument("--max-n", type=int, default=9999, help="数字范围上限（4 位）")
parser.add_argument("--hole-digit", type=int, default=7, help="挖洞末位数字")
args = parser.parse_args()

rng = random.Random(args.seed)

W = args.max_n          # 数字宽度上限（4 位；5~6 位留外推测试）
HO = args.hole_digit    # 挖洞末位
POS_NAMES = ["个位", "十位", "百位", "千位"]   # 低位在前
POS_NAMES_EN = ["ones", "tens", "hundreds", "thousands"]


def is_held(n):
    """挖洞：末位 HO 的 1~4 位数（实例全删，公理保留）"""
    return 0 <= n <= W and n % 10 == HO


def decompose(n):
    """n -> [(digit, weight, name_cn, name_en)]，低位在前"""
    out = []
    i = 0
    while n > 0:
        out.append((n % 10, 10 ** i, POS_NAMES[i], POS_NAMES_EN[i]))
        n //= 10
        i += 1
    return out


def decomp_str(n):
    """4483 -> '4×1000+4×100+8×10+3'（高位在前——人类读法；跳过 0 位；个位省略 ×1）
    ⚠️ 首版 bug：直接迭代 decompose()（低位在前）生成 '3+8×10+4×100+4×1000'，
    数位顺序与数字书写顺序相反，教学信号混乱。必须 reversed()。"""
    parts = []
    for d, w, _, _ in reversed(decompose(n)):
        if d == 0:
            continue
        parts.append(f"{d}×{w}" if w > 1 else f"{d}")
    return "+".join(parts) if parts else "0"


def synth_str(n):
    """4483 -> '4×1000+4×100+8×10+3=4483'（全位版，含 0 位——合成尾巴用）"""
    digits = [int(c) for c in str(n)]
    k = len(digits)
    parts = []
    for i, d in enumerate(digits):
        w = 10 ** (k - 1 - i)
        parts.append(f"{d}×{w}" if w > 1 else f"{d}")
    return "+".join(parts) + f"={n}"


# ================= 推导链生成器（核心） =================

def succ_chain(a):
    """后继推导链（符号形式，套公理）：
    问: s(4483) = ?
    依据后继公理 s(a)=a+1: s(4483) = 4483+1 = 4484
    答: 4484
    ⚠️ 答案 4484 与步骤 4483+1 有 token 级因果链（=4484 是链条末端）"""
    b = a + 1
    return (f"问: s({a}) = ?\n"
            f"依据后继公理 s(a)=a+1: s({a}) = {a}+1 = {b}\n"
            f"答: {b}")


def succ_chain_en(a):
    """后继推导链（英文界面，同一逻辑本体）"""
    b = a + 1
    return (f"Q: s({a}) = ?\n"
            f"By the successor axiom s(a)=a+1: s({a}) = {a}+1 = {b}\n"
            f"A: {b}")


def add_chain(a, b):
    """加法 CoT（用户设计：数位意义 → 对齐 → 低位加起 → 满十进位 → 合成）：
    问: 4483+579 = ?
    数位: 4483=4×1000+4×100+8×10+3, 579=5×100+7×10+9
    对齐: 个位3+9, 十位8+7, 百位4+5, 千位4
    低位加起: 个位3+9=12, 满10进1写2; 十位8+7+1=16, 满10进1写6; 百位4+5+1=10, 满10进1写0; 千位4+1=5
    合成: 5×1000+0×100+6×10+2=5062
    答: 5062
    ⚠️ 每位结果在"低位加起"里显式产生，合成公式逐位引用——token 级因果链完整"""
    c = a + b
    da = [int(x) for x in str(a)]
    db = [int(x) for x in str(b)]
    k = max(len(da), len(db))
    # 低位对齐（前补零）
    da = [0] * (k - len(da)) + da
    db = [0] * (k - len(db)) + db
    # 数位行
    line1 = f"数位: {a}={decomp_str(a)}, {b}={decomp_str(b)}"
    # 对齐行
    align_parts = []
    for i in range(k - 1, -1, -1):
        name = POS_NAMES[k - 1 - i]
        align_parts.append(f"{name}{da[i]}+{db[i]}")
    line2 = "对齐: " + ", ".join(align_parts)
    # 低位加起行（含进位传播）
    add_parts = []
    carry = 0
    res_digits = [0] * k
    for i in range(k - 1, -1, -1):
        name = POS_NAMES[k - 1 - i]
        s = da[i] + db[i] + carry
        res_digits[i] = s % 10
        if i == 0 and s >= 10:
            # 最高位溢出：结果多一位
            add_parts.append(f"{name}{da[i]}+{db[i]}{f'+{carry}' if carry else ''}={s}, 满10进1写{s % 10}")
            res_digits = [1] + res_digits
        elif s >= 10:
            add_parts.append(f"{name}{da[i]}+{db[i]}{f'+{carry}' if carry else ''}={s}, 满10进1写{s % 10}")
            carry = 1
        else:
            add_parts.append(f"{name}{da[i]}+{db[i]}{f'+{carry}' if carry else ''}={s}")
            carry = 0
    line3 = "低位加起: " + "; ".join(add_parts)
    # 合成行（答案 = 显式公式执行）
    line4 = f"合成: {synth_str(c)}"
    return (f"问: {a}+{b} = ?\n{line1}\n{line2}\n{line3}\n{line4}\n答: {c}")


def add_chain_en(a, b):
    """加法 CoT（英文界面）"""
    c = a + b
    da = [int(x) for x in str(a)]
    db = [int(x) for x in str(b)]
    k = max(len(da), len(db))
    da = [0] * (k - len(da)) + da
    db = [0] * (k - len(db)) + db
    line1 = f"Digits: {a}={decomp_str(a)}, {b}={decomp_str(b)}"
    align_parts = []
    for i in range(k - 1, -1, -1):
        name = POS_NAMES_EN[k - 1 - i]
        align_parts.append(f"{name} {da[i]}+{db[i]}")
    line2 = "Align: " + ", ".join(align_parts)
    add_parts = []
    carry = 0
    for i in range(k - 1, -1, -1):
        name = POS_NAMES_EN[k - 1 - i]
        s = da[i] + db[i] + carry
        if s >= 10:
            add_parts.append(f"{name} {da[i]}+{db[i]}{f'+{carry}' if carry else ''}={s}, carry 1 write {s % 10}")
            carry = 1
        else:
            add_parts.append(f"{name} {da[i]}+{db[i]}{f'+{carry}' if carry else ''}={s}")
            carry = 0
    line3 = "Add from lowest: " + "; ".join(add_parts)
    line4 = f"Compose: {synth_str(c)}"
    return (f"Q: {a}+{b} = ?\n{line1}\n{line2}\n{line3}\n{line4}\nA: {c}")


# ================= 数据生成 =================

texts = []


def add(lst, label):
    texts.extend(lst)
    print(f"  {label:<56} {len(lst):>7} 条")


print(f"[挖洞] 末位 {HO} 的 1~4 位数实例将全部删除（公理保留）")

# ================= [D] 定义层（双语+符号，少量高重复）=================
defs = [
    "自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}",
    "后继 Successor: s(a) = a+1",
    "加法 Addition: a+0 = a, a+s(b) = s(a+b)",
    "序 Order: a<b ⟺ ∃c∈ℕ, a+c=b",
    "0 是自然数: 0∈ℕ",
    "自然数的后继是自然数: a∈ℕ ⟹ s(a)∈ℕ",
]
add(defs * 40, "[D] 定义层（双语+符号 ×40 重复）")

# ================= [A] 公理层（代数形式 + 实例化对照）=================
axioms = []
# 代数形式（抽象，防表面捷径的核心）
axiom_texts = [
    "公理: ∀a∈ℕ. s(a) = a+1",
    "公理: ∀a∈ℕ. a+0 = a",
    "公理: ∀a,b∈ℕ. a+s(b) = s(a+b)",
    "公理: ∀a∈ℕ. s(a) ≠ 0",
    "公理: s(a)=s(b) ⟹ a=b",
    "公理: ∀a,b∈ℕ. a<b ⟺ ∃c∈ℕ, a+c=b",
]
axioms.extend(axiom_texts * 30)
# 公理实例化对照（抽象→具体，教"套公理"这个动作）
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
add(axioms, "[A] 公理层（代数形式 ×30 + 实例化 5000）")

# ================= [R] 推导层（核心训练信号，符号 50%）=================
# R1: 后继推导链（符号）
r1 = []
for _ in range(8000):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    r1.append(succ_chain(a))
# 复合后继（组合泛化的训练基础）
for _ in range(1500):
    a = rng.randint(0, 200)
    if is_held(a) or is_held(a + 1) or is_held(a + 2):
        continue
    r1.append(f"问: s(s({a})) = ?\n依据后继公理两次: s(s({a})) = s({a + 1}) = {a + 2}\n答: {a + 2}")
add(r1, "[R1] 后继推导链（符号，含复合）")

# R2: 加法 CoT（符号，用户设计的竖式显式化）
r2 = []
for _ in range(12000):
    a = rng.randint(0, W)
    b = rng.randint(0, W)
    if a + b > W + 10:
        continue
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    r2.append(add_chain(a, b))
# 加法锚点（进位边界，确定性）
for a, b in ((999, 1), (4999, 1), (9999, 1), (99, 1), (9, 1), (1999, 1)):
    for _ in range(8):
        r2.append(add_chain(a, b))
add(r2, "[R2] 加法 CoT（符号，数位→对齐→进位→合成）")

# R3: 序关系推导（符号）
r3 = []
for _ in range(4000):
    a = rng.randint(0, W - 1)
    b = rng.randint(a + 1, min(a + 20, W))
    if is_held(a) or is_held(b):
        continue
    r3.append(f"问: {a}<{b}?\n依据序公理: {a}<{b} ⟺ ∃c∈ℕ, {a}+c={b}\n验证: {a}+{b - a}={b}\n答: 成立")
add(r3, "[R3] 序关系推导（符号）")

# ================= [L] 对齐层（三形式配对，10%）=================
# L1: 中文界面（多表述防表面捷径）
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
# 中文加法（简式，与 CoT 互补）
for _ in range(2000):
    a = rng.randint(0, 500)
    b = rng.randint(0, 500)
    if is_held(a) or is_held(b) or is_held(a + b):
        continue
    l1.append(f"{a} 加 {b} 等于 {a + b}。")
# 中文序
for _ in range(1500):
    a = rng.randint(0, W - 1)
    b = rng.randint(a + 1, min(a + 20, W))
    if is_held(a) or is_held(b):
        continue
    l1.append(f"{a} 小于 {b}。")
add(l1, "[L1] 中文界面（多表述）")

# L2: 英文界面
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

# L3: 三形式对齐配对（同内容连续出现——潜空间对齐的驱动力）
l3 = []
for _ in range(2500):
    a = rng.randint(0, W)
    if is_held(a):
        continue
    b = a + 1
    cn = rng.choice(cn_variants)(a, b)
    en = rng.choice(en_variants)(a, b)
    sym = f"s({a}) = {b}"
    # 随机排列三种形式（防顺序捷径）
    trio = [cn, en, sym]
    rng.shuffle(trio)
    l3.append("\n".join(trio))
add(l3, "[L3] 三形式对齐配对（随机排列）")

# ================= 确定性锚点（自检样例必须命中，不靠采样运气）=================
for _ in range(6):
    texts.append(succ_chain(4483))
    texts.append(succ_chain(999))
    texts.append(add_chain(4483, 579))
    texts.append("4483 的后继是 4484。")
    texts.append("The successor of 4483 is 4484.")
    texts.append("s(4483) = 4484")

# ================= 泄漏检查（挖洞协议）=================
print("=" * 70)


def leak_variants(a):
    """洞 a（末位 HO）的泄漏写法 = 任何教"a 的后继是 a+1"的样本"""
    b = a + 1
    return [
        succ_chain(a),
        succ_chain_en(a),
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
        f"{a}+{b - a}={b}" if b - a == 1 else None,
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
# 截断到目标规模（保持洗牌后的随机子集）
if len(texts) > args.total:
    texts = texts[:args.total]
with open(args.out, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print("=" * 70)
print(f"总计 {len(texts)} 条 -> {args.out}")

# ================= 关键样例自检（动态生成，杜绝手写不一致）=================
S = set(texts)
checks = [
    (succ_chain(4483), "后继推导链"),
    (succ_chain(999), "后继推导链（进位边界）"),
    (add_chain(4483, 579), "加法 CoT"),
    (add_chain(999, 1), "加法 CoT（全9进位）"),
    (add_chain(99, 1), "加法 CoT（两位进位）"),
    ("公理: ∀a∈ℕ. s(a) = a+1", "公理代数形式"),
    ("自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}", "定义层"),
    ("4483 的后继是 4484。", "中文对齐"),
    ("The successor of 4483 is 4484.", "英文对齐"),
    ("s(4483) = 4484", "符号对齐"),
]
n_miss = 0
print("\n关键样例自检：")
for text, label in checks:
    ok = text in S
    if not ok:
        n_miss += 1
    print(f"   {'OK  ' if ok else 'MISS'} {label}: {text[:40]!r}...")

# CoT 数学正确性抽检
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
            print(f"   BAD succ: {t[:60]}")
        n_checked += 1
    m = re.match(r"^问: (\d+)\+(\d+) = \?", t)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        m2 = re.search(r"答: (\d+)$", t)
        if not m2 or int(m2.group(1)) != a + b:
            n_bad += 1
            print(f"   BAD add: {t[:60]}")
        n_checked += 1
print(f"\nCoT 数学正确性抽检: {n_checked} 条中 {n_bad} 条错误")

if n_miss or n_leak or n_bad:
    print(f"\n⚠️ 异常：样例缺失 {n_miss} / 泄漏 {n_leak} / CoT 错误 {n_bad}")
    sys.exit(1)
print("\n✅ 全部就位，可训练")
