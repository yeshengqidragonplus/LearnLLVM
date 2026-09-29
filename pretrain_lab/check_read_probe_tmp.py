# -*- coding: utf-8 -*-
"""check_read_probe_tmp.py — 查读探针（一次性脚本）
验证用户的"整体表达 + 逐位查读"设计前提：
  · 数字**留在输入里**（prompt 中有 94692）
  · 模型能否"读取第 i 位"而不搬运整个数字串？

测什么：
  A. 从低位到高位读（用户设计的"渐进展开"方向）
  B. 从高位到低位读
  C. 指定位置读（"第 3 位"）
  D. 4 位（训练内）vs 5/6 位（外推）——定位能力是否定宽

⚠️ 这些格式 v4-A 训练数据里**没有**（零样本），分数低是正常的。
   我们看的是：**有没有信号**（4 位对、5 位错 = 位置定位定宽；4/5 位都对 = 可跨长度）。
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


def gen(prompt, max_new=120):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


CASES = [
    # A. 低位→高位（用户设计的展开方向）
    ("A 低位起 4位", "问: 4483 的各位（从低位到高位）\n答: "),
    ("A 低位起 5位", "问: 94692 的各位（从低位到高位）\n答: "),
    # B. 高位→低位
    ("B 高位起 4位", "问: 4483 的各位（从高位到低位）\n答: "),
    ("B 高位起 5位", "问: 94692 的各位（从高位到低位）\n答: "),
    # C. 指定位置
    ("C 个位 4位", "问: 4483 的个位是几?\n答: "),
    ("C 个位 5位", "问: 94692 的个位是几?\n答: "),
    ("C 第3位 4位", "问: 4483 的百位是几?\n答: "),
    ("C 第3位 5位", "问: 94692 的百位是几?\n答: "),
    # D. 用户原设计形态：整体绑定 + 逐位展开（不搬运）
    ("D 绑定+读位 4位", "问: 4483+579 = ?\n理解: 算式 F, F := a+b（a 为输入第 1 串，b 为第 2 串）\n算法: i 从 1 起，从右往左读 a_i 与 b_i\ni=1: a_1="),
    ("D 绑定+读位 5位", "问: 10561+914 = ?\n理解: 算式 F, F := a+b（a 为输入第 1 串，b 为第 2 串）\n算法: i 从 1 起，从右往左读 a_i 与 b_i\ni=1: a_1="),
]

print("=" * 78)
print("查读探针（v4-A 模型，零样本——看是否有位置定位信号）")
print("=" * 78)
for label, prompt in CASES:
    txt = gen(prompt)
    print(f"\n{'─' * 78}")
    print(f"【{label}】")
    print(f"prompt: {prompt!r}")
    print(f"输出: {txt!r}")
print("\n" + "=" * 78)