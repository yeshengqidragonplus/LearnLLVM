# -*- coding: utf-8 -*-
"""04_eval_v1.py — 新系列 v1 评估：公理化数系的泛化与对齐
目的：训练前备好评估（旧系列教训：训完没判据 = 白训）。
评估维度（对照 01_make_math_v1.py 的四层数据设计）：

  [E1] 公理套用（挖洞协议——真泛化判据）
       L1 训练内实例: s(4483)=? 应会
       L2 挖洞实例: s(7777)=? 训练中删除（末位 7）——过 = 套公理非查表
       L3 宽度外推: s(54321)=? 5 位数（训练只到 4 位）——过 = 规则长度无关
  [E2] 加法 CoT（竖式算法执行）
       L1 训练内 / L2 挖洞 / L3 宽度外推（5 位）
       判据：最终答案对 + 合成行存在（token 级因果链）
  [E3] 跨形式一致性（潜空间对齐判据）
       中文问 → 符号答: "4483 的后继是?" → "s(4483) = 4484"
       符号问 → 中文答: "s(4483) = ?" → "4483 的后继是 4484。"
       英文问 → 符号答 / 符号问 → 英文答
  [E4] 组合泛化（防表面捷径判据——最严）
       训练只见 s(s(a)) 少量 + 中文从未配过复合形式
       测试: "s(s(4483)) 的结果用中文怎么说?" → "4485"
       过 = 潜空间真有逻辑本体（表面模板互译做不到）
  [E5] 往返测试（编码→逻辑→解码全链）
       "把 4483 的后继用符号表示" → "s(4483) = 4484" → "用中文说" → "4483 的后继是 4484。"
  [E6] 序关系（公理套用的第二场景）
       L1/L2/L3 同 E1 结构

输出：各维度 L1/L2/L3 分数 + 失败样例（前 4 条）+ 总判定。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random
import re

parser = argparse.ArgumentParser(description="v1 评估（公理化数系）")
parser.add_argument("--model", required=True, help="模型目录")
parser.add_argument("--n", type=int, default=15, help="每项测试样本数")
parser.add_argument("--hole-digit", type=int, default=7, help="挖洞末位（与训练一致）")
parser.add_argument("--seed", type=int, default=123)
args = parser.parse_args()

rng = random.Random(args.seed)
HO = args.hole_digit

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

# ⚠️ MAX_NEW：加法 CoT 完整跑完需要 ~200+ token（数位+对齐+低位加起+合成+答案）。
#    首版设 160 导致 E2 全部中途截断，报 0%——那是脚本 bug 不是模型能力
#    （与旧系列 04h [A] 项假测试同类错误）。4 位 CoT 实测需要 ~220，留余量到 260。
MAX_NEW = 260


def gen(prompt, max_new=MAX_NEW):
    """贪心生成。输入 = [eos] + prompt（v4+ 惯例：eos 前缀对齐训练分布）"""
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                              pad_token_id=tok.pad_token_id,
                              eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


# ================= [S0] prompt 前缀一致性自检（铁律）=================
print("[S0] prompt 前缀一致性自检（不过 = 脚本 bug，直接退出）")
SELF_CHECKS = [
    # (完整训练样本形态, prompt)
    # ⚠️ prompt 必须以 `\n` 结尾：训练样本 `?` 后是换行，BPE 会把 `?\n` 合并成
    #    `Ġ?Ċ`——prompt 停在 `?` 会切出裸 `Ġ?`，token 边界与训练分布不一致
    #    （[S0] 自检首跑就抓到这个 bug——铁律有效）。
    ("问: s(4483) = ?\n依据后继公理 s(a)=a+1: s(4483) = 4483+1 = 4484\n答: 4484", "问: s(4483) = ?\n"),
    ("问: 4483+579 = ?\n数位: 4483=4×1000+4×100+8×10+3, 579=5×100+7×10+9\n对齐: 个位3+9, 十位8+7, 百位4+5, 千位4\n低位加起: 个位3+9=12, 满10进1写2; 十位8+7+1=16, 满10进1写6; 百位4+5+1=10, 满10进1写0; 千位4+1=5\n合成: 5×1000+0×100+6×10+2=5062\n答: 5062", "问: 4483+579 = ?\n"),
]
for full, prefix in SELF_CHECKS:
    fids = tok.encode(full)
    pids = tok.encode(prefix)
    if fids[:len(pids)] != pids:
        print(f"  FAIL: {prefix!r} 不是训练样本的 token 前缀！")
        print(f"  full[:{len(pids)}]:  {tok.convert_ids_to_tokens(fids[:len(pids)])}")
        print(f"  prefix tokens:      {tok.convert_ids_to_tokens(pids)}")
        sys.exit(1)
print("  OK（2/2）\n")


# ================= 测试框架 =================
results = []


def run(dim, level, label, cases, check, show=4):
    """cases: [(prompt, expect)]；check(out, expect) -> bool"""
    ok = 0
    fails = []
    for prompt, expect in cases:
        out = gen(prompt)
        if check(out, expect):
            ok += 1
        else:
            fails.append((prompt, out, expect))
    pct = 100.0 * ok / len(cases) if cases else 0.0
    results.append((dim, level, label, ok, len(cases)))
    print(f"  [{level}] {label:<44} {ok}/{len(cases)} = {pct:5.1f}%")
    for p, o, e in fails[:show]:
        print(f"       FAIL {p[:44]!r}")
        print(f"            -> {o[:76]!r}")
    return pct


def pick(lo, hi, n, exclude_hole=True):
    """采样 [lo,hi] 的数，可选排除挖洞数"""
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if exclude_hole and v % 10 == HO:
            continue
        out.append(v)
    return out


def pick_hole(lo, hi, n):
    """采样挖洞数（末位 HO）"""
    out = []
    while len(out) < n:
        v = rng.randint(lo, hi)
        if v % 10 == HO:
            out.append(v)
    return out


# ================= [E1] 公理套用（后继）=================
print("[E1] 公理套用：后继（挖洞协议——真泛化判据）")

def succ_check(out, expect):
    """答案行 `答: X` 的 X 正确"""
    m = re.search(r"答: (\d+)", out)
    return bool(m) and int(m.group(1)) == expect

run("E1", "L1", "训练内实例 s(a)=?",
    [(f"问: s({a}) = ?\n", a + 1) for a in pick(100, 9999, args.n)], succ_check)
run("E1", "L2", f"挖洞实例 s(X{HO})=?（训练已删）",
    [(f"问: s({a}) = ?\n", a + 1) for a in pick_hole(100, 9999, args.n)], succ_check)
run("E1", "L3", "宽度外推 s(5位数)=?（训练只到4位）",
    [(f"问: s({a}) = ?\n", a + 1) for a in pick(10000, 99999, args.n)], succ_check)

# ================= [E2] 加法 CoT ==================
print("\n[E2] 加法 CoT（竖式算法执行）")

def add_check(out, expect):
    """答案对 + 合成行存在（token 级因果链）"""
    if "合成" not in out:
        return False
    m = re.search(r"答: (\d+)", out)
    return bool(m) and int(m.group(1)) == expect

add_cases_l1 = []
for _ in range(args.n):
    a = rng.randint(100, 4999)
    b = rng.randint(100, 4999)
    if a % 10 == HO or b % 10 == HO or (a + b) % 10 == HO:
        continue
    add_cases_l1.append((f"问: {a}+{b} = ?\n", a + b))
run("E2", "L1", "训练内加法（含进位）", add_cases_l1, add_check)

add_cases_l2 = []
for _ in range(args.n):
    a = pick_hole(100, 4999, 1)[0]
    b = rng.randint(1, 99)
    add_cases_l2.append((f"问: {a}+{b} = ?\n", a + b))
run("E2", "L2", f"挖洞加法（a 末位 {HO}）", add_cases_l2, add_check)

add_cases_l3 = []
for _ in range(args.n):
    a = rng.randint(10000, 59999)
    b = rng.randint(1000, 9999)
    add_cases_l3.append((f"问: {a}+{b} = ?\n", a + b))
run("E2", "L3", "宽度外推加法（5位+4位）", add_cases_l3, add_check)

# ================= [E3] 跨形式一致性 ==================
print("\n[E3] 跨形式一致性（潜空间对齐判据）")

def cn_succ_check(out, expect):
    """中文答：`4483 的后继是 4484。` 或含 4484 的中文句"""
    return str(expect) in out and ("后继" in out or "下一个" in out or "加一" in out)

def sym_succ_check(out, expect):
    """符号答：含 `s(4483) = 4484` 或 `= 4484`"""
    return f"= {expect}" in out or f"={expect}" in out

def en_succ_check(out, expect):
    """英文答：含 `is 4484`"""
    return f"is {expect}" in out

run("E3", "X1", "中文问 → 符号答",
    [(f"4483 的后继用符号表示: ", 4484)], sym_succ_check, show=1)
run("E3", "X1", "符号问 → 中文答",
    [(f"s(4483) = ? 用中文回答: ", 4484)], cn_succ_check, show=1)
run("E3", "X1", "英文问 → 符号答",
    [(f"Express the successor of 4483 in symbols: ", 4484)], sym_succ_check, show=1)
run("E3", "X1", "符号问 → 英文答",
    [(f"s(4483) = ? Answer in English: ", 4484)], en_succ_check, show=1)
# 批量跨形式（随机数）
run("E3", "X2", "中文问 → 符号答（批量）",
    [(f"{a} 的后继用符号表示: ", a + 1) for a in pick(100, 9999, args.n)], sym_succ_check)
run("E3", "X2", "符号问 → 中文答（批量）",
    [(f"s({a}) = ? 用中文回答: ", a + 1) for a in pick(100, 9999, args.n)], cn_succ_check)

# ================= [E4] 组合泛化（防表面捷径）=================
print("\n[E4] 组合泛化（最严判据：复合逻辑跨形式）")

def combo_check(out, expect):
    return str(expect) in out

run("E4", "C1", "s(s(a)) 符号复合（训练少量）",
    [(f"问: s(s({a})) = ?\n", a + 2) for a in pick(100, 2000, args.n)], succ_check)
run("E4", "C2", "s(s(a)) 中文问（训练从未配过）",
    [(f"{a} 的后继的后继是? ", a + 2) for a in pick(100, 2000, args.n)], combo_check)
run("E4", "C2", "s(s(a)) 英文问（训练从未配过）",
    [(f"The successor of the successor of {a} is? ", a + 2) for a in pick(100, 2000, args.n)], combo_check)

# ================= [E5] 往返测试 ==================
print("\n[E5] 往返测试（编码→逻辑→解码全链）")

def roundtrip_check(out, expect):
    """两步往返：符号化 + 中文回述，最终中文句含正确结果"""
    return str(expect) in out and ("后继" in out or "下一个" in out)

run("E5", "R1", "中文→符号→中文（单样本演示）",
    [(f"把 {a} 的后继先用符号表示，再用中文说: ", a + 1) for a in pick(100, 9999, 3)],
    roundtrip_check, show=1)

# ================= [E6] 序关系 ==================
print("\n[E6] 序关系（公理套用第二场景）")

def lt_check(out, expect):
    """`答: 成立` 或 `答: 不成立`"""
    if expect:
        return "成立" in out
    return "不成立" in out

lt_cases = []
for _ in range(args.n):
    a = rng.randint(100, 9998)
    b = rng.randint(a + 1, a + 20)
    lt_cases.append((f"问: {a}<{b}?\n", True))
run("E6", "L1", "训练内序关系", lt_cases, lt_check)

lt_hole = []
for _ in range(args.n):
    a = pick_hole(100, 9998, 1)[0]
    b = a + rng.randint(1, 9)
    lt_hole.append((f"问: {a}<{b}?\n", True))
run("E6", "L2", f"挖洞序关系（a 末位 {HO}）", lt_hole, lt_check)

lt_l3 = []
for _ in range(args.n):
    a = rng.randint(10000, 99998)
    b = rng.randint(a + 1, a + 20)
    lt_l3.append((f"问: {a}<{b}?\n", True))
run("E6", "L3", "宽度外推序关系（5位）", lt_l3, lt_check)

# ================= 汇总 ==================
print("\n" + "=" * 70)
print("汇总（按维度）")
dims = {}
for dim, level, label, ok, total in results:
    dims.setdefault(dim, [0, 0])
    dims[dim][0] += ok
    dims[dim][1] += total
for dim in sorted(dims):
    ok, total = dims[dim]
    pct = 100.0 * ok / total if total else 0
    names = {"E1": "公理套用（后继）", "E2": "加法 CoT", "E3": "跨形式一致性",
             "E4": "组合泛化", "E5": "往返测试", "E6": "序关系"}
    print(f"  {dim} {names[dim]:<24} {ok}/{total} = {pct:5.1f}%")

print("\n判定标准：")
print("  · E1-L2（挖洞）高 = 真泛化（套公理非查表）")
print("  · E1-L3（宽度）高 = 规则长度无关（突破旧系列宽度墙）")
print("  · E3/E4 高 = 潜空间对齐形成（枢轴语言方案成功）")
print("  · E4 低但 E3 高 = 表面捷径（对齐是假的，只有模板互译）")
print("  · E2-L3 高 = 竖式算法外推成功（用户设计的 CoT 有效）")
