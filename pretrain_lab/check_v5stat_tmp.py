# -*- coding: utf-8 -*-
"""check_v5stat_tmp.py — v5 数据统计 + BPE 预检（一次性脚本）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re

n = 0
L = []
layers = {}
for line in open("data/math_v5.jsonl", encoding="utf-8"):
    t = json.loads(line)["text"]
    n += 1
    L.append(len(t))
    if t.startswith("问: ") and "是几位数" in t.split("\n")[0]:
        layers["数位数"] = layers.get("数位数", 0) + 1
    elif re.match(r"^问: \d+ 的\S+?是几\?", t):
        layers["取位"] = layers.get("取位", 0) + 1
    elif re.match(r"^问: s\(", t):
        layers["后继"] = layers.get("后继", 0) + 1
    elif re.match(r"^问: \d+\+\d+", t):
        layers["加法"] = layers.get("加法", 0) + 1
    elif re.match(r"^问: \d+<\d+", t):
        layers["序关系"] = layers.get("序关系", 0) + 1
    elif re.match(r"^问: 抄写", t):
        layers["抄写(不应有)"] = layers.get("抄写(不应有)", 0) + 1
    else:
        layers["其他"] = layers.get("其他", 0) + 1

print(f"math_v5.jsonl = {n} 条 | 最长 {max(L)} | 平均 {round(sum(L)/n)}")
print("层分布：")
for k in sorted(layers):
    print(f"  {k:<14} {layers[k]:>7}")

print("\n[BPE 预检] v5 新符号")
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("tokenizer")
pats = [
    "问: 94692 的百位是几?",
    "理解: 数 v := 94692",
    "算法: 百位 = (v div 100) mod 10 = 946 mod 10",
    "答: 6",
    "取位 Digit: n 的第 k 位 = (n div 10^k) mod 10（k 从 0 起）",
    "问: 抄写 94692",
]
for p in pats:
    print(f"  {p!r} -> {tok.convert_ids_to_tokens(tok.encode(p))}")