# -*- coding: utf-8 -*-
"""readnum.py — 数值 ↔ 读法（中文/英文/缩写）枢轴对齐模块
用途：v3 数据 [N] 读法层。同一数值的多种人类表面形式，对齐到同一本体。

覆盖（用户要求 N1~N5 全做）：
  N1 中文段内：1~9999（含零处理 4003→四千零三、10→十、110→一百一十）
  N2/N3 中文分段：万/亿（10000→一万、4000300→四百万零三百、1亿零一）
  N4 缩写：5千=5000、1.5万=15000、3w=30000、2k=2000、1M=1000000
  N5 英文：four thousand four hundred eighty-three / million / billion

设计原则：中英文是"界面"，符号/数值是"本体"。读法层的样本都是
"数值 = 读法" 的等式（双向），训练潜空间把读法对齐到数值。
"""
DIGITS_CN = "零一二三四五六七八九"
UNITS_CN = ["", "十", "百", "千"]


def _seg_cn(n):
    """1~9999 段内中文读法（含零处理）"""
    assert 1 <= n <= 9999, n
    s = ""
    zero_pending = False
    for pos in (3, 2, 1, 0):
        d = (n // 10 ** pos) % 10
        if d == 0:
            if s:
                zero_pending = True
            continue
        if zero_pending:
            s += "零"
            zero_pending = False
        if pos == 1 and d == 1 and s == "":
            s += "十"          # 10→十（不是一十）
        else:
            s += DIGITS_CN[d] + UNITS_CN[pos]
    return s


def cn_read(n):
    """0~10^12 中文读法（含万/亿分段 + 跨段零）"""
    if n == 0:
        return "零"
    assert 0 <= n < 10 ** 12, n
    segs = [(n // 10 ** 8) % 10 ** 4, (n // 10 ** 4) % 10 ** 4, n % 10 ** 4]
    names = ["亿", "万", ""]
    out = ""
    for seg, name in zip(segs, names):
        if seg == 0:
            continue
        if out and seg < 1000:
            out += "零"        # 跨段补零（4000300→四百万零三百）
        out += _seg_cn(seg) + name
    return out


# ---------- 英文 ----------
_EN_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
            "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty",
            "seventy", "eighty", "ninety"]
_EN_SCALES = ["", " thousand", " million", " billion"]


def _en_below_100(n):
    if n < 20:
        return _EN_ONES[n]
    t, o = divmod(n, 10)
    return _EN_TENS[t] + (f"-{_EN_ONES[o]}" if o else "")


def _en_below_1000(n, use_and=True):
    """英式：100 后有余数加 and（one hundred and twenty-three）"""
    h, r = divmod(n, 100)
    parts = []
    if h:
        parts.append(f"{_EN_ONES[h]} hundred")
    if r:
        if h and use_and:
            parts.append("and")
        parts.append(_en_below_100(r))
    return " ".join(parts)


def en_read(n):
    """0~10^12 英文读法（英式：段内 hundred and + 段间末段<100 补 and）
    选择理由（2026-09-17）：and 是分段边界信号，与中文"零"对称；
    美式 "four thousand three" 口语歧义，英式 "four thousand and three" 清晰。
    """
    if n == 0:
        return "zero"
    assert 0 <= n < 10 ** 12, n
    chunks = []
    k = 0
    while n > 0:
        c = n % 1000
        if c:
            chunks.append((_en_below_1000(c) + _EN_SCALES[k], c))
        n //= 1000
        k += 1
    chunks.reverse()  # 高位在前
    # 段间 and：最小段 <100 且前面还有段时补 and
    out = []
    for i, (txt, c) in enumerate(chunks):
        out.append(txt)
        if i < len(chunks) - 1:
            nxt = chunks[i + 1][1]
            if nxt < 100:
                out.append("and")
    return " ".join(out)


# ---------- 缩写（N4）----------
CN_ABBR = [(10 ** 8, "亿"), (10 ** 4, "万"), (10 ** 3, "千")]
EN_ABBR = [(10 ** 9, "B"), (10 ** 6, "M"), (10 ** 3, "k")]


def _fmt_num(x):
    """小数格式化（精度安全）：
    ⚠️ 首版用 f"{x:.4f}" 会截断精度（1.00000001 → "1"），配合宽 unit 会输出
       错误缩写（100000001 → "1亿"）。改用浮点精度的 repr 路径 + 去尾零。
    """
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    # 用 repr 保留真实精度（Python float repr 是最短可往返表示）
    s = repr(round(x, 10))
    if "e" in s or "E" in s:
        return s
    return s.rstrip("0").rstrip(".")


def _abbr_ok(n, unit, x, s):
    """缩写合法性（两道关，防错误数据）：
    1. 精度回验：float(s) * unit 必须精确等于 n（杜绝 1亿≠100000001）
    2. 自然度：小数位 ≤1（5千/1.5万 自然；4.483千 不自然）
    ⚠️ 首版 bug：_fmt_num 用 .4f 把 1.00000001 截成 "1" → 输出"1亿"（错误数据）。
    """
    if "." in s:
        if len(s.split(".")[1]) > 1:
            return False
    try:
        if int(round(float(s) * unit)) != n:
            return False
    except ValueError:
        return False
    return True


def cn_abbr(n):
    """中文缩写：5千=5000、1.5万=15000（小数位≤1 + 精度回验）"""
    for unit, suf in CN_ABBR:
        if n >= unit:
            s = _fmt_num(n / unit)
            if _abbr_ok(n, unit, n / unit, s):
                return s + suf
    return None


def en_abbr(n):
    """英文缩写：2k / 4.5k / 1M / 3B"""
    for unit, suf in EN_ABBR:
        if n >= unit:
            s = _fmt_num(n / unit)
            if _abbr_ok(n, unit, n / unit, s):
                return s + suf
    return None


def w_read(n):
    """中文互联网"w"（万）：3w=30000、1.5w=15000"""
    if n >= 10 ** 4:
        s = _fmt_num(n / 10 ** 4)
        if _abbr_ok(n, 10 ** 4, n / 10 ** 4, s):
            return s + "w"
    return None