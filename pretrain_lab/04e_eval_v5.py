# -*- coding: utf-8 -*-
"""04e_eval_v5.py — v5 (NAVE) 评估：核心验证"取位运算能否替代复制"

⚠️ 关键：NAVE 模型**不能用 model.generate**（新 token 走内建 embed_tokens，
   不会注入 NAVE → 与训练不一致）。必须手写贪心解码。

核心判据（DESIGN_nave_v5 §7.3）：
  · copy probe 5 位 ≥60% → 破墙成功（v4-A 是 0%）
  · 取位 probe 5/6 位 → 位置定位是否可用
  · 宽度扫描 5 位后继/加法 ≥60%
  · 数轴几何 cos(e_val(v),e_val(v+1)) 远高于随机基线

用法：
  .\.venv\Scripts\python 04e_eval_v5.py --model output_v5c1/final_model --nave --n
  （--nave 必须与训练时一致；不带则按普通模型走 model.generate）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import random
import re

parser = argparse.ArgumentParser(description="v5 NAVE 评估")
parser.add_argument("--model", required=True)
parser.add_argument("--nave", action="store_true", help="模型训练时启用了 NAVE")
parser.add_argument("--nave-theta", type=int, default=8)
parser.add_argument("--n", type=int, default=10)
parser.add_argument("--seed", type=int, default=123)
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

# ================= NAVE 加载（与训练一致） =================
NAVE = None
if args.nave:
    from nave import NaveWrapper
    digit_ids = []
    for d in "0123456789":
        ids = tok.encode(d, add_special_tokens=False)
        assert len(ids) == 1, f"数字 {d} 非单 token"
        digit_ids.append(ids[0])
    NAVE = NaveWrapper(model, model.config.hidden_size, digit_ids,
                       n_theta=args.nave_theta)
    # 从 nave.pt 恢复编码器（from_pretrained 不加载非 config 子模块）
    import os as _os
    _nave_path = _os.path.join(args.model, "nave.pt")
    if _os.path.exists(_nave_path):
        NAVE.enc.load_state_dict(torch.load(_nave_path, map_location="cpu"))
        print(f"[NAVE] 从 {_nave_path} 恢复编码器")
    else:
        # 兜底：尝试从 state_dict 里找 nave.* （某些保存路径会包含）
        sd = model.state_dict()
        nave_sd = {k[len("nave."):]: v for k, v in sd.items() if k.startswith("nave.")}
        if nave_sd:
            NAVE.enc.load_state_dict(nave_sd)
            print(f"[NAVE] 从 state_dict 恢复编码器（{len(nave_sd)} 张量）")
        else:
            print(f"[NAVE] ⚠️ 未找到 {_nave_path}，编码器为随机初始化（结果不可信！）")
    NAVE.enc = NAVE.enc.to(torch.bfloat16).to(device)
    NAVE.enc.eval()


def greedy_decode(prompt, max_new=400):
    """手写贪心解码（NAVE 一致）

    ⚠️ 关键 bug 修复（首版）：增量步**不能用 x[:,-1:]**——
    NAVE 的前缀值需要"完整历史数字串"才能算对
    （如生成 '94692' 时，第 3 步必须知道前面已有 '94'）。
    首版只传最后一个 token → prefix_values 看不到历史 → 注入错误值。
    修正：NAVE 模式下每步都传**完整序列**（重新编码），
    但只对"最后一个位置"取 logits；KV cache 仍可用于主干
    （前缀值只影响 embeds，不改变已算的 KV 的正确性需保证——
     故 NAVE 模式**不用 cache**，每步全序列前向，保证正确优先）。
    """
    ids = [eos] + tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    out_ids = []
    with torch.no_grad():
        for _ in range(max_new):
            if NAVE is not None:
                # 完整序列重算（保证前缀值正确）；不用 KV cache
                embeds = NAVE.embed(model, x)
                o = model(inputs_embeds=embeds, use_cache=False)
            else:
                o = model(input_ids=x, use_cache=False)
            nxt = int(o.logits[0, -1].argmax())
            if nxt == eos:
                break
            out_ids.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out_ids, skip_special_tokens=True)


def gen(prompt, max_new=400):
    if NAVE is not None:
        return greedy_decode(prompt, max_new)
    x = torch.tensor([[eos] + tok.encode(prompt)], device=device)
    with torch.no_grad():
        out = model.generate(x, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    return tok.decode(out[0][x.shape[1]:], skip_special_tokens=True)


results = []


def run(dim, level, label, cases, check, show=3):
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
    print(f"  [{level}] {label:<34} {ok}/{len(cases)} = {pct:5.1f}%")
    for p, o, e in fails[:show]:
        print(f"       FAIL {p[:36]!r} -> {o[:88]!r}")
    return pct


def num_field(field):
    def chk(out, expect):
        m = re.search(rf"{field}: (\d+)", out)
        return bool(m) and int(m.group(1)) == expect
    return chk


def ans_check(out, expect):
    if "合成" not in out:
        return False
    return num_field("答")(out, expect)


def contains_num(out, expect):
    return re.search(rf"(?<!\d){expect}(?!\d)", out) is not None


def pick(lo, hi, n):
    return [rng.randint(lo, hi) for _ in range(n)]


# ================= [P0] 手写解码 vs generate 对拍（非 NAVE 时才可对拍） =================
if NAVE is None:
    print("[P0] 手写解码 vs generate 对拍（普通模型，应一致）")
    import importlib.util
    # 临时用 generate 对照
    probe = "问: 4483+579 = ?\n"
    x = torch.tensor([[eos] + tok.encode(probe)], device=device)
    with torch.no_grad():
        g_out = model.generate(x, max_new_tokens=200, do_sample=False,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    g_txt = tok.decode(g_out[0][x.shape[1]:], skip_special_tokens=True)
    # 手写版（临时把 NAVE 关掉走 greedy_decode 的 else 分支）
    h_txt = greedy_decode(probe, 200)
    same = g_txt.strip() == h_txt.strip()
    print(f"   {'OK' if same else 'FAIL'} 一致性：{'一致' if same else '不一致'}")
    if not same:
        print(f"     generate: {g_txt[:80]!r}")
        print(f"     手写:     {h_txt[:80]!r}")
else:
    print("[P0] NAVE 模式：跳过对拍（generate 不支持 NAVE 注入，属预期）")
print()

# ================= [CP] 复制探针（破墙核心） =================
print("[CP] 复制探针（v4-A: 4位对/5位丢位）")
# 判据：输出以期望数字串开头（容忍后续标点/换行）
run("CP", "4位", "抄写 4 位",
    [(f"问: 抄写 {a}\n", str(a)) for a in pick(1000, 9999, args.n)],
    lambda o, e: o.strip().startswith(e))
run("CP", "5位", "抄写 5 位 ★破墙判据",
    [(f"问: 抄写 {a}\n", str(a)) for a in pick(10000, 99999, args.n)],
    lambda o, e: o.strip().startswith(e))
run("CP", "6位", "抄写 6 位",
    [(f"问: 抄写 {a}\n", str(a)) for a in pick(100000, 999999, args.n)],
    lambda o, e: o.strip().startswith(e))

# ================= [RP] 取位探针（位置定位） =================
print("\n[RP] 取位探针（验证位置定位是否跨长度）")
for w, lo, hi in [(4, 1000, 9999), (5, 10000, 99999), (6, 100000, 999999)]:
    # ⚠️ 首版 bug：prompt 末尾多加了 `答: `，但训练格式是 `问: ...\n理解: 数 v := ...`，
    #    导致模型收到分布外 prompt（结果不可信）。修正：只给 `问: ...\n`。
    cases = [(f"问: {a} 的个位是几?\n", a % 10) for a in pick(lo, hi, max(4, args.n // 2))]
    run("RP", f"{w}位", f"个位（{w} 位）", cases, num_field("答"))

# ================= [W] 宽度扫描（后继/加法） =================
print("\n[W] 宽度扫描（v4-A: 5 位 0%）")
for w, lo, hi in [(3, 100, 999), (4, 1000, 9999), (5, 10000, 99999), (6, 100000, 599999)]:
    run("W", f"{w}位", f"后继 s({w}位数)",
        [(f"问: s({a}) = ?\n", a + 1) for a in pick(lo, hi, args.n)], ans_check)
for w, lo, hi in [(4, 1000, 4999), (5, 10000, 59999), (6, 100000, 599999)]:
    cases = []
    for _ in range(args.n):
        a, b = rng.randint(lo, hi), rng.randint(100, 999)
        cases.append((f"问: {a}+{b} = ?\n", a + b))
    run("W", f"{w}位", f"加法（{w}位+3位）", cases, ans_check)

# ================= [C] 数位数（v4-A: 5 位 0%） =================
print("\n[C] 数位数")
for w, lo, hi in [(4, 1000, 9999), (5, 10000, 99999), (6, 100000, 999999)]:
    run("C", f"{w}位", f"数位数（{w} 位）",
        [(f"问: {a} 是几位数?\n", w) for a in pick(lo, hi, max(4, args.n // 2))],
        num_field("答"))

# ================= [G] 数轴几何（验证"潜空间认识数"） =================
if NAVE is not None:
    print("\n[G] 数轴几何（NAVE 值表示的关系结构）")
    from nave import prefix_values
    with torch.no_grad():
        def e_val(v):
            x = torch.tensor([[int(c) for c in str(v)]], device=device)
            # 直接编码"完整值"（取最后一位的前缀值 = 完整值）
            pv = torch.tensor([[v]], device=device)
            return NAVE.enc.encode(pv)[0, 0]
        # 相邻数相似度
        sims_adj, sims_far = [], []
        for _ in range(50):
            v = rng.randint(1000, 99999)
            a = e_val(v)
            b = e_val(v + 1)
            c = e_val(v + rng.randint(100, 999))
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            cos_far = torch.nn.functional.cosine_similarity(a.unsqueeze(0), c.unsqueeze(0)).item()
            sims_adj.append(cos)
            sims_far.append(cos_far)
        import statistics
        print(f"   cos(相邻数 v,v+1)   = {statistics.mean(sims_adj):.4f}（应高）")
        print(f"   cos(远数 v,v+Δ)     = {statistics.mean(sims_far):.4f}（应低）")
        # 位权结构：e_val(1000)-e_val(999) 是否 ≈ e_val(10)-e_val(9)
        with torch.no_grad():
            d1 = e_val(1000) - e_val(999)
            d2 = e_val(10) - e_val(9)
            cs = torch.nn.functional.cosine_similarity(d1.unsqueeze(0), d2.unsqueeze(0)).item()
        print(f"   cos(Δ1000, Δ10)     = {cs:.4f}（若高 → 位权结构一致）")
        # proj 激活度
        pw = NAVE.enc.proj.weight
        print(f"   proj.weight 范数     = {pw.norm().item():.4f}（>0 说明 NAVE 已激活）")

# ================= 汇总 =================
print("\n" + "=" * 72)
dims = {}
for dim, level, label, ok, total in results:
    dims.setdefault(dim, []).append((level, label, ok, total))
names = {"CP": "复制探针（破墙）", "RP": "取位探针", "W": "宽度扫描",
         "C": "数位数"}
for dim in ["CP", "RP", "W", "C"]:
    if dim not in dims:
        continue
    print(f"\n  {dim} {names[dim]}")
    for level, label, ok, total in dims[dim]:
        pct = 100.0 * ok / total if total else 0
        print(f"    {level:<6} {label:<26} {ok}/{total} = {pct:5.1f}%")

print("\n判据（DESIGN_nave_v5 §7.3）：")
print("  CP-5位 ≥60% → ★破墙成功（v4-A 是 0%）")
print("  RP 5/6 位高 → 位置定位可用")
print("  W-5位 ≥60% → 宽度外推成功")
print("  G 相邻数相似度高 + proj 范数 >0 → NAVE 真正激活")