# -*- coding: utf-8 -*-
"""_merge_v5_tmp.py — 合并两批 v5 数据 + 去重（一次性脚本）"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import random
import os

parts = ["data/_v5_a.jsonl", "data/_v5_b.jsonl"]
seen = set()
out = []
for p in parts:
    if not os.path.exists(p):
        print(f"  ⚠️ 缺失 {p}")
        continue
    cnt = 0
    for line in open(p, encoding="utf-8"):
        t = json.loads(line)["text"]
        cnt += 1
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    print(f"  {p}: {cnt} 条")

print(f"合并去重后 = {len(out)}")
random.seed(2026)
random.shuffle(out)
target = 120000
final = out[:target] if len(out) > target else out
with open("data/math_v5.jsonl", "w", encoding="utf-8") as f:
    for t in final:
        f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
print(f"写入 data/math_v5.jsonl = {len(final)}")