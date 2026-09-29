# -*- coding: utf-8 -*-
"""check_probes_tmp.py — 三探针诊断（一次性脚本，v2 格式定稿依据）
目的：用 v1 现成模型测"哪种输出格式能绕过宽度绑定"。

v1 已证：推理在符号上 → 宽度无关（100%）；搬运字面量 → 宽度绑定（0%）。
v2 大纲格式的两个残留字面量点，对应三探针：

  探针 A（字面量复制）: 读: 11962          → 期望 11962
     测：裸复制（v1 失败模式，预期 4 位好 5 位崩——作基线）
  探针 B（逗号列表）  : 列出 11962 的各位:  → 期望 1,1,9,6,2
     测：合成行/读位行的发射格式（每个数字后跟逗号，2-token 循环）
  探针 C（每行一位）  : 逐位读出 11962:     → 期望 个位: 2 / 十位: 6 / ...
     测：竖式行的读取格式（每行一个数字，与运算融合）

⚠️ 注意：v1 训练数据里没有这三种任务——这是零样本探针，测的是
"格式本身的先验可行性"（模型没学过但哪种格式它天然做得更好）。
分数低是正常的，我们看的是【三种格式的相对差异】+【宽度塌陷曲线】。

判读：
  - A 塌 B/C 不塌 → 逗号列表/每行一位格式可行，v2 按设计走
  - A/B/C 全塌   → 读取通道本身宽度绑定，转 NAVE（表示层）
  - B 塌 C 不塌  → 竖式行格式优先（读取与运算融合优于独立列表）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random

parser = argparse.ArgumentParser(description="三探针诊断")
parser.add_argument("--model", default="output_v1/final_model")
parser.add_argument("--seed", type=int, default=777)
args = parser.parse_args()

rng = random.Random(args.seed)

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("tokenizer")
model = Qwen2ForCausalLM.from_pretrained(args.model).to(torch.bfloat16).to(device)
model.generation_config.pad_token_id = tok.pad_token_id
model.generation_config.eos_token_id = tok.eos_token_id
model.eval()
eos = tok.eos_token_id


def gen(prompt, max_new=60):
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id,
                             eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


def digits_of(n):
    return [int(c) for c in str(n)]


# ================= 探针定义 =================
# 每个探针: (名称, prompt构造, 期望输出构造, 校验函数)
PROBES = [
    ("A 裸复制", lambda n: f"读: {n}\n",
     lambda n: str(n),
     lambda out, n: out.strip().startswith(str(n))),

    ("B 逗号列表", lambda n: f"列出 {n} 的各位:\n",
     lambda n: ",".join(str(d) for d in digits_of(n)),
     lambda out, n: ",".join(str(d) for d in digits_of(n)) in out.replace(" ", "")),

    ("C 每行一位", lambda n: f"逐位读出 {n}:\n",
     lambda n: "\n".join(f"{'个十百千万'[i]}位: {digits_of(n)[::-1][i]}" for i in range(len(str(n)))),
     lambda out, n: all(f"{digits_of(n)[::-1][i]}" in out for i in range(len(str(n))))),
]

# ================= 宽度扫描 =================
WIDTHS = [2, 3, 4, 5, 6, 7]   # 4 位是训练边界，5~7 位是外推
N_PER = 10                     # 每宽度每探针样本数

print("=" * 78)
print("三探针宽度扫描（v1 模型，零样本——看相对差异与塌陷曲线）")
print("=" * 78)

summary = {}
for pname, mk_prompt, mk_expect, check in PROBES:
    print(f"\n【探针 {pname}】")
    summary[pname] = {}
    for w in WIDTHS:
        ok = 0
        fails = []
        for _ in range(N_PER):
            # 生成 w 位数（首位非零）
            n = rng.randint(10 ** (w - 1), 10 ** w - 1)
            out = gen(mk_prompt(n))
            if check(out, n):
                ok += 1
            else:
                fails.append((n, out))
        pct = 100.0 * ok / N_PER
        summary[pname][w] = pct
        mark = "✅" if pct >= 60 else ("⚠️" if pct >= 30 else "❌")
        print(f"  {w}位: {ok}/{N_PER} = {pct:5.1f}% {mark}")
        for n, o in fails[:2]:
            print(f"       FAIL {n} -> {o[:60]!r}")

# ================= 汇总表 =================
print("\n" + "=" * 78)
print("汇总（行=探针，列=位数，单位 %）")
print(f"{'探针':<12}", end="")
for w in WIDTHS:
    print(f"{w:>7}位", end="")
print()
for pname in summary:
    print(f"{pname:<12}", end="")
    for w in WIDTHS:
        v = summary[pname][w]
        print(f"{v:>7.0f}", end="")
    print()

print("\n判读：")
print("  A 塌 B/C 不塌 → v2 按大纲格式走（逗号列表/竖式行可行）")
print("  A/B/C 全塌   → 读取通道宽度绑定，转 NAVE 表示层方案")
print("  B 塌 C 不塌  → 竖式行格式优先（读取与运算融合）")
