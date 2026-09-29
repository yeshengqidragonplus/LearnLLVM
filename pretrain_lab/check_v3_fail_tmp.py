# -*- coding: utf-8 -*-
"""check_v3_fail_tmp.py — v3 失败样例完整诊断（一次性脚本）
目的：日志把输出截断到 90 字符，无法判断是"模型错"还是"生成被截断"。
本脚本打印关键失败用例的【完整输出】，逐例人工判读。

诊断清单：
  1. 5位后继（怀疑绑定行丢位 vs 生成截断）
  2. 3位加法（训练内却 26.7%，反常）
  3. 序 a>b（⊥）、a=a（⊥）（怀疑模型只会 ⊤ 路径）
  4. 读法 英文/万级/反向（怀疑 prompt 歧义 + 丢位）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("tokenizer")
model = Qwen2ForCausalLM.from_pretrained("output_v3/final_model").to(torch.bfloat16).to(device)
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.eval()
eos = tok.eos_token_id


def gen(prompt, max_new=400):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    txt = tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)
    return txt, len(out[0]) - x.shape[1]


CASES = [
    # 1. 5位后继（绑定行是否丢位）
    ("5位后继", "问: s(97443) = ?\n"),
    ("5位后继2", "问: s(28635) = ?\n"),
    # 2. 3位加法（训练内反常）
    ("3位加法(进位到4位)", "问: 686+897 = ?\n"),
    ("3位加法(无最终进位)", "问: 293+700 = ?\n"),
    # 对照：4位加法（100% 通过）
    ("4位加法(对照)", "问: 4483+579 = ?\n"),
    # 3. 序 a>b、a=a
    ("序 a>b (⊥)", "问: 5503<5497?\n"),
    ("序 a=a (⊥)", "问: 490<490?\n"),
    ("序 a<b (⊤, 对照)", "问: 500<505?\n"),
    # 4. 读法
    ("数值→英文读法", "1583 = "),
    ("数值→中文读法(万级)", "38310 = "),
    ("中文读法→数值", "七万九千九百二十八 = "),
    # 对照：数值→中文4位（100%）
    ("数值→中文读法(4位,对照)", "1583 = "),
]

print("=" * 78)
print("v3 失败样例完整输出诊断（每一例打印全文）")
print("=" * 78)
for label, prompt in CASES:
    txt, ntok = gen(prompt)
    print(f"\n{'─' * 78}")
    print(f"【{label}】prompt = {prompt!r}")
    print(f"{'─' * 78}")
    print(txt)
    print(f"[生成 {ntok} token]")
print("\n" + "=" * 78)