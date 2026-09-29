# -*- coding: utf-8 -*-
"""check_v4_fail_tmp.py — v4-A 失败根因诊断（一次性脚本）
关键疑点：
  1. 数位数 5 位 0%：失败样例显示 `数列 L := 9,4,6,2`（5 位只抄了 4 个）
     —— 与 v3 的 `x := 9,7,4,4` 是**同一个病**！绑定行本身截断。
  2. 序 ⊤ 从 v3 的 100% 掉到 33%（退化）
诊断目标：确认"读取/复制行"是否能承载 5 位
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("tokenizer")
model = Qwen2ForCausalLM.from_pretrained("output_v4a/final_model").to(torch.bfloat16).to(device)
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.eval()
eos = tok.eos_token_id


def gen(prompt, max_new=300):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


CASES = [
    # 【核心疑点】数位数的读取行（4 位 vs 5 位 vs 6 位）
    ("数位数 4位（对照，应 100%）", "问: 4483 是几位数?\n"),
    ("数位数 5位（关键）", "问: 94692 是几位数?\n"),
    ("数位数 6位（关键）", "问: 645898 是几位数?\n"),
    # 【纯复制探针】直接让模型抄一遍（无任何推理）
    ("纯复制 4位", "问: 抄写 4483\n答: "),
    ("纯复制 5位", "问: 抄写 94692\n答: "),
    # 【退化疑点】序 ⊤（v3 是 100%）
    ("序 ⊤ 小差值", "问: 1619<1629?\n"),
    ("序 ⊤ 差1", "问: 500<501?\n"),
    ("序 ⊤ 对照（v3 常见形态）", "问: 100<105?\n"),
    # 读法万级（需要读 5 位数）
    ("万级读法（需读5位）", "16653 = "),
]

print("=" * 78)
print("v4-A 失败根因诊断")
print("=" * 78)
for label, prompt in CASES:
    txt = gen(prompt)
    print(f"\n{'─' * 78}")
    print(f"【{label}】prompt = {prompt!r}")
    print(f"{'─' * 78}")
    print(txt)
print("\n" + "=" * 78)