# -*- coding: utf-8 -*-
"""check_tok_v1_tmp.py — 新 v1（公理化数系）BPE 预检（一次性脚本）
目的：新系列所有符号/格式在 Qwen2 tokenizer 下无合并陷阱。
这是新系列的第一个脚本——v7.1 的 `=-` 合并毁过一整个版本，预检是铁律。

检查范围（四层数据格式的全部 token 组合）：
  【定义层】ℕ={0,1,2,...} / 自然数 Natural numbers
  【公理层】a∈ℕ. s(a)=a+1 / ∀a∈ℕ / a+0=a / a+s(b)=s(a+b)
  【推导层】问: s(4483)=? / 依据后继公理: s(4483)=4483+1=4484 / 答: 4484
  【对齐层】4483 的后继是 4484。 / The successor of 4483 is 4484.
  【加法 CoT】数位: 4483=4×1000+4×100+8×10+3 / 满10进1写2 / 合成: ...
  【序公理】a<b ⟺ ∃c∈ℕ, a+c=b
陷阱判据（真分布外合并才 BAD）：
  `=-`（v7.1 根因）、`=×`、`×=`、`s(`与数字粘连、`∈`与符号粘连、
  `+0`/`+1`与字母粘连、`?`与中文粘连、`:`与中文粘连
  `Ġ-`（空格负号）在 v1 无负数，出现即报（提前防 v2 的雷）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer")

samples = [
    # (文本, 说明, 是否验前缀)
    # ---- 定义层 ----
    ("自然数 Natural numbers: ℕ = {0, 1, 2, 3, ...}", "定义：中英+符号", False),
    ("后继 Successor: s(a) = a+1", "定义：后继函数", False),
    ("加法 Addition: a+0 = a, a+s(b) = s(a+b)", "定义：递归加法", False),
    ("序 Order: a<b ⟺ ∃c∈ℕ, a+c=b", "定义：序关系", False),
    # ---- 公理层（代数形式）----
    ("公理: ∀a∈ℕ. s(a) = a+1", "公理：全称量词", False),
    ("公理: ∀a∈ℕ. a+0 = a", "公理：加法单位元", False),
    ("公理: ∀a,b∈ℕ. a+s(b) = s(a+b)", "公理：加法递归", False),
    ("公理: ∀a∈ℕ. s(a) ≠ 0", "公理：0 不是后继", False),
    ("公理: s(a)=s(b) ⟹ a=b", "公理：单射", False),
    # ---- 推导层（思维链套公理）----
    ("问: s(4483) = ?", "推导：问题", True),
    ("依据后继公理 s(a)=a+1: s(4483) = 4483+1 = 4484", "推导：套公理", True),
    ("答: 4484", "推导：答案", True),
    ("问: 4483+579 = ?", "加法问题", True),
    ("数位: 4483=4×1000+4×100+8×10+3, 579=5×100+7×10+9", "加法CoT：数位", True),
    ("对齐: 个位3+9, 十位8+7, 百位4+5, 千位4", "加法CoT：对齐", True),
    ("低位加起: 个位3+9=12, 满10进1写2; 十位8+7+1=16, 满10进1写6", "加法CoT：进位", True),
    ("合成: 5×1000+0×100+6×10+2=5062", "加法CoT：合成", True),
    ("答: 5062", "加法答案", True),
    # ---- 对齐层（三形式）----
    ("4483 的后继是 4484。", "对齐：中文", False),
    ("The successor of 4483 is 4484.", "对齐：英文", False),
    ("s(4483) = 4484", "对齐：符号", False),
    ("4483 的下一个数是 4484。", "对齐：中文变体2", False),
    ("4483 加一是 4484。", "对齐：中文变体3", False),
    ("4483 plus one is 4484.", "对齐：英文变体2", False),
    # ---- 序关系 ----
    ("3 < 4, 因为 3+1 = 4", "序：中文推导", False),
    ("3 小于 4。", "序：中文", False),
    ("3 is less than 4.", "序：英文", False),
    ("3 < 4", "序：符号", False),
    # ---- 挖洞协议涉及的形态 ----
    ("问: s(7777) = ?", "挖洞测试问题（训练中删除其实例）", True),
    # ---- 组合泛化形态 ----
    ("问: s(s(3)) = ?", "复合后继", True),
    ("依据后继公理两次: s(s(3)) = s(4) = 5", "复合推导", True),
]

print("=" * 78)
all_ok = True
for text, label, check_prefix in samples:
    ids = tok.encode(text)
    toks = tok.convert_ids_to_tokens(ids)
    # 陷阱扫描
    bad = []
    for t in toks:
        # v7.1 根因：= 与 - 合并
        if "=-" in t:
            bad.append(t)
        # 合成公式粘连
        if "=×" in t or "×=" in t:
            bad.append(t)
        # s( 与数字粘连（s(4 变单 token 会破坏位值对齐）
        if t.startswith("s(") and len(t) > 2 and t[2].isdigit():
            bad.append(t)
        # ∈ 与字母/数字粘连
        if "∈" in t and any(c.isalnum() for c in t.replace("∈", "")):
            bad.append(t)
        # 字母与数字粘连（a0, b1 之类会破坏代数形式）
        if any(c.isalpha() for c in t) and any(c.isdigit() for c in t) and "Ġ" not in t and "ã" not in t and "å" not in t and "ç" not in t:
            # 排除中文 token（bytes 形态含字母数字混合是正常的）
            if t.isascii():
                bad.append(t)
        # ? : 与中文粘连（真陷阱：问号/冒号被并进中文 token）
        # ⚠️ `Ġ?`（空格+问号）是正常组合 token——`Ġ` 是 U+0120（>127），
        #    首版判据 `ord(c)>127` 把它误判为中文粘连。判据改为：
        #    只查 CJK 区段（U+4E00~U+9FFF）与 ?/: 的合并。
        def has_cjk(s):
            return any(0x4E00 <= ord(c) <= 0x9FFF for c in s)
        if "?" in t and has_cjk(t):
            bad.append(t)
        if ":" in t and has_cjk(t):
            bad.append(t)
        # 空格负号（v1 无负数，出现即报）
        if "Ġ-" in t:
            bad.append(t)
    status = "OK " if not bad else "BAD"
    if bad:
        all_ok = False
    print(f"[{status}] {label}")
    print(f"       {text}")
    print(f"       tokens({len(toks)}): {toks[:16]}{'...' if len(toks) > 16 else ''}")
    if bad:
        print(f"       ⚠️ 陷阱: {bad}")

print("=" * 78)

# 额外：关键符号的独立切分参考
print("\n关键符号切分参考：")
for sym in ["ℕ", "∈", "∀", "⟺", "∃", "≠", "⟹", "s(", "s(a)", "a+1", "a+0", "{0,", "...}"]:
    ids = tok.encode(sym)
    toks = tok.convert_ids_to_tokens(ids)
    print(f"  {sym!r:<12} -> {toks}")

print("=" * 78)
print("全部通过 ✅" if all_ok else "存在陷阱 ⚠️ —— 修格式再生成数据！")
