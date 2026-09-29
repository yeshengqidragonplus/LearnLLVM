# -*- coding: utf-8 -*-
"""check_len_tmp.py — v3 数据长度分组统计（一次性脚本）
目的：用户质疑"8/9 位锚点太长"——用数字说话，分组看长度分布。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
import statistics
from collections import defaultdict

groups = defaultdict(list)
for line in open("data/math_v3.jsonl", encoding="utf-8"):
    t = json.loads(line)["text"]
    n = len(t)
    m = re.match(r"^问: (\d+)\+(\d+)", t)
    if m:
        w = max(len(m.group(1)), len(m.group(2)))
        groups[f"加法{w}位"].append(n)
    elif re.match(r"^问: s\(", t) and "循环" in t:
        m2 = re.match(r"^问: s\((\d+)\)", t)
        groups[f"后继{len(m2.group(1))}位"].append(n)
    elif re.match(r"^问: \d+<\d+", t):
        groups["序关系"].append(n)
    elif re.match(r"^\d+ = ", t) or re.match(r"^[^ ]+ = \d+$", t):
        groups["读法"].append(n)
    else:
        groups["其他"].append(n)

print(f"{'分组':<12} {'条数':>7} {'最长':>6} {'平均':>6} {'中位':>6}")
for k in sorted(groups):
    v = groups[k]
    print(f"{k:<12} {len(v):>7} {max(v):>6} {round(sum(v)/len(v)):>6} {statistics.median(v):>6.0f}")

all_len = [len(json.loads(l)["text"]) for l in open("data/math_v3.jsonl", encoding="utf-8")]
print(f"\n全局: 条数 {len(all_len)}, 最长 {max(all_len)}, 平均 {round(sum(all_len)/len(all_len))}, 中位 {statistics.median(all_len):.0f}")
# 长度分布（关键：有多少超过 300 字符）
for th in (150, 200, 250, 300, 400):
    c = sum(1 for x in all_len if x > th)
    print(f"  > {th} 字符: {c} 条")