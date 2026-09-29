# -*- coding: utf-8 -*-
"""check_readnum_tmp.py — 读法层独立校验（一次性脚本）
目的：用户要求"推演保证正确，别用错误的数据训练"。

方法：write 用 readnum.py 生成读法，verify 用【独立的解析器】把读法
解析回数值，必须等于原数。解析是逆运算——生成器写错、解析器写错
（两处独立 bug）才可能同时通过，实际不可能。

覆盖：
  中文特殊规则（10/110/4003/10000/4000300/1010/100000001）
  英文特殊规则（100/1000/1e6/1000001）
  缩写（5千/1.5万/3w/2k/4.5k/1M）
  随机批量（1~10^9）交叉验证
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import random
from readnum import cn_read, en_read, cn_abbr, en_abbr, w_read
# 解析器从共享模块读（与生成器 readnum.py 完全独立实现，交叉验证）
from readnum_parse import parse_cn, parse_en, parse_abbr_cn, parse_abbr_en

rng = random.Random(2026)

# ================= 校验 =================
print("=" * 78)
print("读法层独立校验（生成 → 独立解析回数值 → 比对）")
print("=" * 78)

n_bad = 0
n_ok = 0


def chk(label, n, gen, parser, *args):
    global n_bad, n_ok
    s = gen(*args)
    if s is None:
        return
    try:
        back = parser(s)
    except Exception as e:
        n_bad += 1
        print(f"[BAD ] {label}: {n} -> {s!r} 解析异常 {e}")
        return
    if back != n:
        n_bad += 1
        print(f"[BAD ] {label}: {n} -> {s!r} -> 解析回 {back}")
    else:
        n_ok += 1


# 中文特殊规则（人工确认的期望读法）
CN_CASES = {
    0: "零", 1: "一", 10: "十", 11: "十一", 20: "二十", 110: "一百一十",
    100: "一百", 101: "一百零一", 1000: "一千", 1010: "一千零一十",
    4003: "四千零三", 1200: "一千二百", 10000: "一万", 10001: "一万零一",
    1000000: "一百万", 4000300: "四百万零三百", 100000001: "一亿零一",
    12345678: "一千二百三十四万五千六百七十八",
    10001_0000: "一亿零一万",
}
print("\n[中文读法] 特殊规则人工核对：")
for n, expect in CN_CASES.items():
    got = cn_read(n)
    ok = got == expect
    if not ok:
        n_bad += 1
        print(f"  [BAD ] {n}: 得 {got!r}，期望 {expect!r}")
    else:
        n_ok += 1
print(f"  人工核对 {len(CN_CASES)} 个，全部一致 {n_bad == 0}")

print("\n[中文读法] 解析回验（随机 1~10^9，500 个）：")
for _ in range(500):
    n = rng.randint(1, 10 ** 9)
    chk("cn", n, cn_read, parse_cn, n)

print("\n[英文读法] 解析回验（随机 1~10^9，500 个）：")
for _ in range(500):
    n = rng.randint(1, 10 ** 9)
    chk("en", n, en_read, parse_en, n)

print("\n[英文读法] 特殊规则：")
EN_CASES = {100: "one hundred", 1000: "one thousand",
            110: "one hundred and ten",
            123: "one hundred and twenty-three", 10 ** 6: "one million",
            1000001: "one million and one", 2 * 10 ** 9: "two billion",
            4003: "four thousand and three",
            1000000 + 1: "one million and one"}
for n, expect in EN_CASES.items():
    got = en_read(n)
    ok = got == expect
    if not ok:
        n_bad += 1
        print(f"  [BAD ] {n}: 得 {got!r}，期望 {expect!r}")
    else:
        n_ok += 1
print(f"  人工核对 {len(EN_CASES)} 个")

print("\n[缩写] 解析回验：")
for _ in range(300):
    # 千级短倍数
    n = rng.randint(1, 9999) * 1000
    chk("abbr千", n, cn_abbr, parse_abbr_cn, n)
for _ in range(300):
    n = rng.randint(1, 9999) * 10000
    chk("abbr万", n, cn_abbr, parse_abbr_cn, n)
for _ in range(300):
    n = rng.randint(1, 999) * 1000
    chk("abbrk", n, en_abbr, parse_abbr_en, n)
for _ in range(200):
    n = rng.randint(1, 99) * 10 ** 6
    chk("abbrM", n, en_abbr, parse_abbr_en, n)

print("\n[缩写] w 形式（3w 类）：")
for n in (30000, 50000, 15000, 100000):
    s = w_read(n)
    if s:
        back = parse_abbr_cn(s)
        ok = back == n
        n_ok += ok
        n_bad += (not ok)
        print(f"  {'OK ' if ok else 'BAD'} {n} -> {s}")

print("=" * 78)
print(f"通过 {n_ok} / 失败 {n_bad}")
print("\n示例：")
for n in (4483, 4003, 100000001, 12345678, 5000, 15000, 30000, 2000):
    ca, ea = cn_abbr(n), en_abbr(n)
    wa = w_read(n)
    print(f"  {n}: 中文[{cn_read(n)}] 英文[{en_read(n)}] "
          f"缩写[中:{ca} 英:{ea} w:{wa}]")
print("=" * 78)
print("✅ 读法规则全部正确，可用于生成数据" if n_bad == 0 else "⚠️ 存在错误，修好再用！")