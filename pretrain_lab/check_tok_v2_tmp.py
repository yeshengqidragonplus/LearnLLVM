# -*- coding: utf-8 -*-
"""check_tok_v2_tmp.py — v2（大纲渐进展开格式）BPE 预检（一次性脚本）
目的：v2 新格式的全部 token 组合在 Qwen2 tokenizer 下无陷阱。
铁律来源：v7.1 的 `=-` 合并毁过一整个版本。

v2 新格式（竖式行读位，探针 C 验证可行）：
  问: 112+522222 = ?
  理解: 算式 F, F=a+b, a=112, b=522222
  算法: 按位对齐，从个位起相加，满十进一
  个位: a=2, b=2, 2+2+0 = 4, 写4进0
  十位: a=1, b=2, 1+2+0 = 3, 写3进0
  ...
  合成: 5,2,2,3,3,4
  答: 522334

后继格式（同构）：
  问: s(4483) = ?
  理解: 算式 F, F=s(x), x=4483
  算法: 后继公理 s(x)=x+1，个位加一，满十进一
  个位: x=3, 3+1+0 = 4, 写4进0
  十位: x=8, 8+0+0 = 8, 写8进0
  ...
  合成: 4,4,8,4
  答: 4484

陷阱判据（继承 v1 预检 + 新增）：
  `=-`、`=×`、`×=`、`s(`与数字粘连、`∈`粘连、字母数字粘连（ASCII）、
  `?`/`:`与 CJK 粘连、`Ġ-`（空格负号，v2 仍无负数）
  新增：`写4进0` 的"写X进Y"模式、`a=2,` 的变量绑定模式、
        `F=a+b` 的算式绑定、`5,2,2,3,3,4` 逗号列表
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer")


def has_cjk(s):
    return any(0x4E00 <= ord(c) <= 0x9FFF for c in s)


samples = [
    # ---- 加法完整样本 ----
    ("问: 112+522222 = ?\n理解: 算式 F, F=a+b, a=112, b=522222\n算法: 按位对齐，从个位起相加，满十进一\n个位: a=2, b=2, 2+2+0 = 4, 写4进0\n十位: a=1, b=2, 1+2+0 = 3, 写3进0\n百位: a=1, b=2, 1+2+0 = 3, 写3进0\n千位: a=0, b=2, 0+2+0 = 2, 写2进0\n万位: a=0, b=2, 0+2+0 = 2, 写2进0\n十万位: a=0, b=2, 0+2+0 = 2, 写2进0\n合成: 5,2,2,3,3,4\n答: 522334", "加法完整样本（6位）"),
    ("问: 4483+579 = ?\n理解: 算式 F, F=a+b, a=4483, b=579\n算法: 按位对齐，从个位起相加，满十进一\n个位: a=3, b=9, 3+9+0 = 12, 写2进1\n十位: a=8, b=7, 8+7+1 = 16, 写6进1\n百位: a=4, b=5, 4+5+1 = 10, 写0进1\n千位: a=4, b=0, 4+0+1 = 5, 写5进0\n合成: 5,0,6,2\n答: 5062", "加法完整样本（4位含进位）"),
    # ---- 后继完整样本 ----
    ("问: s(4483) = ?\n理解: 算式 F, F=s(x), x=4483\n算法: 后继公理 s(x)=x+1，个位加一，满十进一\n个位: x=3, 3+1+0 = 4, 写4进0\n十位: x=8, 8+0+0 = 8, 写8进0\n百位: x=4, 4+0+0 = 4, 写4进0\n千位: x=4, 4+0+0 = 4, 写4进0\n合成: 4,4,8,4\n答: 4484", "后继完整样本"),
    ("问: s(4999) = ?\n理解: 算式 F, F=s(x), x=4999\n算法: 后继公理 s(x)=x+1，个位加一，满十进一\n个位: x=9, 9+1+0 = 10, 写0进1\n十位: x=9, 9+0+1 = 10, 写0进1\n百位: x=9, 9+0+1 = 10, 写0进1\n千位: x=4, 4+0+1 = 5, 写5进0\n合成: 5,0,0,0\n答: 5000", "后继连续进位样本"),
    # ---- 各行单独预检 ----
    ("问: 112+522222 = ?", "加法问题行", ),
    ("理解: 算式 F, F=a+b, a=112, b=522222", "理解行（绑定）"),
    ("算法: 按位对齐，从个位起相加，满十进一", "算法行"),
    ("个位: a=2, b=2, 2+2+0 = 4, 写4进0", "竖式行（个位）"),
    ("十万位: a=0, b=2, 0+2+0 = 2, 写2进0", "竖式行（十万位）"),
    ("合成: 5,2,2,3,3,4", "合成行（逗号列表）"),
    ("答: 522334", "答案行"),
    ("理解: 算式 F, F=s(x), x=4483", "后继理解行"),
    ("算法: 后继公理 s(x)=x+1，个位加一，满十进一", "后继算法行"),
    ("个位: x=3, 3+1+0 = 4, 写4进0", "后继竖式行"),
    # ---- 对齐层新形态 ----
    ("4483 的后继是 4484。", "中文对齐（保留 v1）"),
    ("The successor of 4483 is 4484.", "英文对齐（保留 v1）"),
    ("s(4483) = 4484", "符号对齐（保留 v1）"),
    # ---- 序关系（保留 v1 格式）----
    ("问: 3<4?\n依据序公理: 3<4 ⟺ ∃c∈ℕ, 3+c=4\n验证: 3+1=4\n答: 成立", "序推导（保留 v1）"),
]

print("=" * 78)
all_ok = True
for text, label in samples:
    ids = tok.encode(text)
    toks = tok.convert_ids_to_tokens(ids)
    bad = []
    for t in toks:
        if "=-" in t or "=×" in t or "×=" in t:
            bad.append(t)
        if t.startswith("s(") and len(t) > 2 and t[2].isdigit():
            bad.append(t)
        if "∈" in t and any(c.isalnum() for c in t.replace("∈", "")):
            bad.append(t)
        if t.isascii() and any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and "Ġ" not in t:
            bad.append(t)
        if "?" in t and has_cjk(t):
            bad.append(t)
        if ":" in t and has_cjk(t):
            bad.append(t)
        if "Ġ-" in t:
            bad.append(t)
    status = "OK " if not bad else "BAD"
    if bad:
        all_ok = False
    print(f"[{status}] {label}")
    print(f"       tokens({len(toks)}): {toks[:14]}{'...' if len(toks) > 14 else ''}")
    if bad:
        print(f"       ⚠️ 陷阱: {bad}")

print("=" * 78)
# 关键模式切分参考
print("\n关键模式切分参考：")
for pat in ["写4进0", "写0进1", "a=2,", "F=a+b", "F=s(x)", "x=4483", "5,2,2,3,3,4",
            "个位: a=2", "十万位: a=0", "算式 F", "满十进一"]:
    toks = tok.convert_ids_to_tokens(tok.encode(pat))
    print(f"  {pat!r:<16} -> {toks}")

print("=" * 78)
print("全部通过 ✅" if all_ok else "存在陷阱 ⚠️ —— 修格式再生成数据！")
