# -*- coding: utf-8 -*-
"""readnum_parse.py — 读法【解析器】（读法 -> 数值，逆运算）
用途：① check_readnum_tmp.py 的独立校验（生成 vs 解析交叉验证）
     ② 01c_make_math_v3.py 的读法层自检（数据里的读法解析回原值）

⚠️ 本模块只做"读法 -> 数值"，与 readnum.py（数值 -> 读法）完全独立实现——
两条路各自独立，才能交叉验证（同错同过不可能）。
"""
CN_D = {"零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
        "七": 7, "八": 8, "九": 9, "两": 2}
CN_U = {"十": 10, "百": 100, "千": 1000}
CN_S = {"万": 10 ** 4, "亿": 10 ** 8}

EN_ONES = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
           "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
           "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
           "nineteen": 19}
EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
           "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
EN_SCALE = {"thousand": 10 ** 3, "million": 10 ** 6, "billion": 10 ** 9}


def parse_cn(s):
    """中文读法 -> 数值"""
    total = 0
    seg_val = 0
    num = 0
    for ch in s:
        if ch in CN_D:
            num = CN_D[ch]
        elif ch in CN_U:
            seg_val += (num if num else 1) * CN_U[ch]
            num = 0
        elif ch in CN_S:
            seg_val += num
            num = 0
            total += (seg_val if seg_val else 1) * CN_S[ch]
            seg_val = 0
        elif ch == "零":
            num = 0
        else:
            raise ValueError(f"未知字符 {ch!r} in {s!r}")
    return total + seg_val + num


def parse_en(s):
    """英文读法 -> 数值（支持英式 and）"""
    total = 0
    cur = 0
    for w in s.replace("-", " ").split():
        if w == "and":
            continue
        if w in EN_ONES:
            cur += EN_ONES[w]
        elif w in EN_TENS:
            cur += EN_TENS[w]
        elif w == "hundred":
            cur *= 100
        elif w in EN_SCALE:
            total += cur * EN_SCALE[w]
            cur = 0
        else:
            raise ValueError(f"未知词 {w!r} in {s!r}")
    return total + cur


def parse_abbr_cn(s):
    """中文缩写 5千/1.5万/3w/1亿 -> 数值"""
    if s.endswith("亿"):
        return int(round(float(s[:-1]) * 10 ** 8))
    if s.endswith("万"):
        return int(round(float(s[:-1]) * 10 ** 4))
    if s.endswith("千"):
        return int(round(float(s[:-1]) * 10 ** 3))
    if s.endswith("w"):
        return int(round(float(s[:-1]) * 10 ** 4))
    raise ValueError(s)


def parse_abbr_en(s):
    """英文缩写 2k/4.5k/1M/3B -> 数值"""
    mult = {"k": 10 ** 3, "M": 10 ** 6, "B": 10 ** 9}
    for suf, m in mult.items():
        if s.endswith(suf):
            return int(round(float(s[:-1]) * m))
    raise ValueError(s)