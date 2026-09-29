# -*- coding: utf-8 -*-
"""check_chain_v3_tmp.py — v3 循环格式思维链推演 + 独立校验（一次性脚本）
目的：用户要求"推演保证正确，别用错误的数据训练"。

方法（关键）：生成与校验用【两套独立实现】，避免同错同过——
  · 生成器 loop_chain()：模拟模型的逐步推演（i 从 1 循环，c 状态传递）
  · 校验器 verify()：完全独立 —— 直接算 a+b，再逐 i 拆解核对，
    并解析链中的所有数字做一致性检查（不调用生成器任何函数）

校验项：
  1. 合成行（解析 d_i 重组） == str(a+b)
  2. 答 == a+b
  3. 每一行的 s_i == a_i + b_i + c_prev（进位链自洽）
  4. 每一行的 d_i == s_i % 10 且 c_out == s_i // 10
  5. 绑定行数字拼接 == str(a) / str(b)
  6. i 的范围 == max(位数)，且补位行仅在最后 c=1 时出现

覆盖用例：普通 / 单进位 / 连续进位 / 补位溢出 / 不等长 / 大数(8~9位)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import re

POS_NAMES = ["个位", "十位", "百位", "千位", "万位", "十万位", "百万位",
             "千万位", "亿位", "十亿位", "百亿位"]


def digits_low_first(n):
    out = []
    while n >= 10:
        out.append(n % 10)
        n //= 10
    out.append(n)
    return out


def seg(n):
    """4483 -> '4,4,8,3'（高位在前，绑定行分段）"""
    return ",".join(str(n))


# ================= 生成器：逐步推演（模拟模型） =================

def loop_chain(a, b):
    da = digits_low_first(a)
    db = digits_low_first(b)
    k = max(len(da), len(db))
    c = a + b
    lines = [
        f"问: {a}+{b} = ?",
        f"理解: 算式 F, F := a+b, a := {seg(a)}, b := {seg(b)}",
        f"算法: i 从 1 到 {k} 循环：s_i := a_i+b_i+c; d_i := s_i mod 10; c := s_i div 10（c 初值 0，a_i 从右往左取）",
    ]
    carry = 0
    ds = []
    for i in range(1, k + 1):
        ai = da[i - 1] if i - 1 < len(da) else 0
        bi = db[i - 1] if i - 1 < len(db) else 0
        s = ai + bi + carry
        d = s % 10
        new_c = s // 10
        lines.append(f"i={i}: a_{i}={ai}, b_{i}={bi}, c={carry}, s_{i}={ai}+{bi}+{carry}={s}, d_{i}={d}, c:={new_c}")
        ds.append(d)
        carry = new_c
    if carry:
        lines.append(f"结束: c={carry}≠0，补位 d_{k + 1}={carry}, c:=0")
        ds.append(carry)
    else:
        lines.append("结束: c=0，无补位")
    ds_high = list(reversed(ds))
    lines.append("合成: " + ",".join(f"d_{k + 1 - j}" if False else str(d) for j, d in enumerate(ds_high)))
    lines.append(f"答: {c}")
    return "\n".join(lines)


# ================= 校验器：完全独立实现 =================

def verify(a, b, chain):
    """独立校验：不调用 loop_chain 的任何中间量，全部从链文本重新解析"""
    errs = []
    expect = a + b

    # 1. 答 == a+b
    ma = re.search(r"^答: (\d+)$", chain, re.M)
    if not ma or int(ma.group(1)) != expect:
        errs.append(f"答错: {ma.group(1) if ma else None} != {expect}")

    # 2. 合成行重组成整数 == a+b
    ms = re.search(r"^合成: ([\d,]+)$", chain, re.M)
    if not ms:
        errs.append("无合成行")
    else:
        synth = int(ms.group(1).replace(",", ""))
        if synth != expect:
            errs.append(f"合成错: {synth} != {expect}")

    # 3. 绑定行拼接 == str(a)/str(b)
    mb = re.search(r"a := ([\d,]+), b := ([\d,]+)", chain)
    if not mb:
        errs.append("无绑定行")
    else:
        if mb.group(1).replace(",", "") != str(a):
            errs.append(f"绑定a错: {mb.group(1)} != {a}")
        if mb.group(2).replace(",", "") != str(b):
            errs.append(f"绑定b错: {mb.group(2)} != {b}")

    # 4. 逐 i 行：s_i == a_i+b_i+c_prev，d_i == s_i%10，c_out == s_i//10
    da = digits_low_first(a)
    db = digits_low_first(b)
    k = max(len(da), len(db))
    c_prev = 0
    ds = []
    for i in range(1, k + 1):
        mr = re.search(
            rf"^i={i}: a_{i}=(\d+), b_{i}=(\d+), c=(\d+), s_{i}=(\d+)\+(\d+)\+(\d+)=(\d+), d_{i}=(\d+), c:=(\d+)$",
            chain, re.M)
        if not mr:
            errs.append(f"缺 i={i} 行")
            break
        ai, bi, c_in, x1, x2, x3, s, d, c_out = map(int, mr.groups())
        ai_e = da[i - 1] if i - 1 < len(da) else 0
        bi_e = db[i - 1] if i - 1 < len(db) else 0
        if (ai, bi) != (ai_e, bi_e):
            errs.append(f"i={i} 读位错: ({ai},{bi}) != ({ai_e},{bi_e})")
        if (x1, x2, x3) != (ai, bi, c_in):
            errs.append(f"i={i} 算式显示错: {x1}+{x2}+{x3}")
        if s != ai + bi + c_in:
            errs.append(f"i={i} s 错: {s} != {ai + bi + c_in}")
        if d != s % 10 or c_out != s // 10:
            errs.append(f"i={i} d/c 错: d={d} c={c_out} (s={s})")
        if c_in != c_prev:
            errs.append(f"i={i} 进位链断: c_in={c_in} != 上一步 c={c_prev}")
        ds.append(d)
        c_prev = c_out

    # 5. 结束行（补位）正确性
    me = re.search(r"^结束: (.+)$", chain, re.M)
    if not me:
        errs.append("无结束行")
    else:
        tail = me.group(1)
        if c_prev != 0:
            if f"d_{k + 1}={c_prev}" not in tail:
                errs.append(f"补位缺: 应 d_{k + 1}={c_prev}，实际 {tail}")
            ds.append(c_prev)
        else:
            if "无补位" not in tail:
                errs.append(f"应无补位，实际 {tail}")

    # 6. 合成 == 逐 i 的 d 逆序拼接（独立于第2条，交叉验证）
    ds_str = "".join(str(d) for d in reversed(ds))
    if ms and ms.group(1).replace(",", "") != ds_str:
        errs.append(f"合成与 d_i 不一致: {ms.group(1)} != {ds_str}")

    return errs


# ================= 测试用例 =================
CASES = [
    (4483, 579),      # 普通含进位
    (999, 1),         # 连续进位 + 补位
    (9999, 1),
    (99, 1),
    (9, 1),
    (112, 522222),    # 不等长（6位+3位）
    (0, 0),
    (7, 5),
    (10000, 99999),   # 5位
    (123456, 654321), # 6位
    (12345678, 87654321),   # 8位
    (100000000, 1),         # 9位补位
    (500, 500),       # 末位相加刚好进位
    (1099, 1),        # 中间进位
]

print("=" * 78)
print("v3 循环格式思维链推演 + 独立校验")
print("=" * 78)

n_bad = 0
for a, b in CASES:
    chain = loop_chain(a, b)
    errs = verify(a, b, chain)
    status = "OK  " if not errs else "BAD "
    if errs:
        n_bad += 1
    # 防止脚本自身错误：用 Python 原生算法再验一次
    if a + b != int(re.search(r"答: (\d+)$", chain, re.M).group(1)):
        n_bad += 1
        errs.append("Python 原生 a+b 不符（生成器根本错误）")
    print(f"[{status}] {a}+{b} = {a + b}   链长 {len(chain)} 字符")
    for e in errs:
        print(f"        ⚠️ {e}")

print("=" * 78)
print(f"用例 {len(CASES)} 个，错误 {n_bad} 个")
print("\n--- 示例链（4483+579）---")
print(loop_chain(4483, 579))
print("\n--- 示例链（9999+1，补位）---")
print(loop_chain(9999, 1))
print("=" * 78)
print("✅ 推演全部正确，可用于生成数据" if n_bad == 0 else "⚠️ 存在错误，修好再生成数据！")