# -*- coding: utf-8 -*-
"""check_mem_tmp.py — OOM 根因定位（一次性脚本）
假设：显存杀手是【词表 151646 × 序列长度 × batch】的 logits 张量
      （v2 序列 ~130 能用 batch=16；v3 序列 ~300 就爆）
验证：实测 v2/v3 的 token 长度，算 logits 显存
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
import statistics
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer")
VOCAB = 151646


def scan(path, label):
    add4 = []
    all_len = []
    for line in open(path, encoding="utf-8"):
        t = json.loads(line)["text"]
        n = len(tok.encode(t))
        all_len.append(n)
        m = re.match(r"^问: (\d+)\+(\d+)", t)
        if m and max(len(m.group(1)), len(m.group(2))) == 4:
            add4.append(n)
    print(f"\n【{label}】")
    print(f"  全部样本: 条数 {len(all_len)}, token 最长 {max(all_len)}, "
          f"平均 {round(sum(all_len)/len(all_len))}, 中位 {statistics.median(all_len):.0f}")
    if add4:
        print(f"  4位加法: 条数 {len(add4)}, token 最长 {max(add4)}, "
              f"平均 {round(sum(add4)/len(add4))}, 中位 {statistics.median(add4):.0f}")
    return max(all_len), (statistics.median(add4) if add4 else 0)


v2_max, v2_add = scan("data/math_v2.jsonl", "v2")
v3_max, v3_add = scan("data/math_v3.jsonl", "v3")


def logits_mem(batch, seq):
    """logits 张量显存：batch × seq × vocab × 2 bytes (bf16)"""
    bf16 = batch * seq * VOCAB * 2 / 1e9
    fp32 = batch * seq * VOCAB * 4 / 1e9   # cross_entropy 若升 fp32
    return bf16, fp32


print("\n" + "=" * 70)
print("logits 显存估算（词表 %d）" % VOCAB)
print("=" * 70)
print(f"{'场景':<28} {'logits(bf16)':>14} {'+fp32升位':>12}")
for label, seq in [("v2 中位序列 (batch=16)", v2_add),
                   ("v3 中位序列 (batch=16)", v3_add),
                   ("v3 最长序列 (batch=16)", v3_max),
                   ("v3 中位序列 (batch=8)", v3_add),
                   ("v3 最长序列 (batch=8)", v3_max),
                   ("v3 最长序列 (batch=4)", v3_max)]:
    batch = 16 if "16" in label else (8 if "8" in label else 4)
    bf, fp = logits_mem(batch, seq)
    print(f"{label:<28} {bf:>11.2f}GB {fp:>9.2f}GB")

print("\n其他固定开销（4层 65.32M 模型）:")
print(f"  权重 bf16:         {65.32e6 * 2 / 1e9:.2f}GB")
print(f"  优化器 AdamW fp32: {65.32e6 * 2 * 4 / 1e9:.2f}GB")
print(f"  梯度 fp32:         {65.32e6 * 4 / 1e9:.2f}GB")
print(f"  合计:              {65.32e6 * 10 / 1e9:.2f}GB")
print("\n注：8GB 卡实际可用约 7.2GB（系统/显示占用 ~0.8GB）")