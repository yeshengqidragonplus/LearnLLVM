# -*- coding: utf-8 -*-
"""check_icl_probe_tmp.py — 上下文学习（ICL）探针（一次性脚本）
目的：零样本失败不能证明"位置查读"不可学。
      给几个示例（in-context），看能力是否"潜在可激发"。

判据：
  · ICL 4 位示例 → 新 4 位对（说明能学格式）
  · ICL 示例后 → 新 5/6 位对（说明位置定位可跨长度）
  · 全部乱 → 需要专门训练（转 v5-mini 实验）
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


def gen(prompt, max_new=80):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


# 4 位示例（教格式），然后问新的
ICL4 = ("问: 1234 的个位是几?\n答: 4\n"
        "问: 5678 的个位是几?\n答: 8\n"
        "问: 9012 的个位是几?\n答: 2\n")

CASES = [
    ("ICL 后 → 新4位", ICL4 + "问: 4483 的个位是几?\n答: "),
    ("ICL 后 → 新5位", ICL4 + "问: 94692 的个位是几?\n答: "),
    ("ICL 后 → 新6位", ICL4 + "问: 645898 的个位是几?\n答: "),
    # 高位示例（另一种问法）
    ("ICL 高位 → 新5位",
     "问: 1234 的最高位是几?\n答: 1\n"
     "问: 5678 的最高位是几?\n答: 5\n"
     "问: 94692 的最高位是几?\n答: "),
    # 逐位展开（用户设计的形态）
    ("ICL 逐位 → 新5位",
     "问: 1234 的各位（低位到高位）\n答: 4,3,2,1\n"
     "问: 5678 的各位（低位到高位）\n答: 8,7,6,5\n"
     "问: 94692 的各位（低位到高位）\n答: "),
]

print("=" * 78)
print("上下文学习（ICL）探针")
print("=" * 78)
for label, prompt in CASES:
    txt = gen(prompt)
    print(f"\n【{label}】")
    print(f"  输出: {txt!r}")
print("\n" + "=" * 78)