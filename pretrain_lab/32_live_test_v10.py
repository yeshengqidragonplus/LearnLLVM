# -*- coding: utf-8 -*-
"""v10 现场测试：5 组分布外用例，检验「类型标记 + 整体编号」机制是否真成立。

留出集 468 条全 1.00 只证明「同分布泛化」；本脚本探的是**训练分布之外**：
模型没见过的问法、没被监督过的类型组合、超出训练规模的包表与投影。
如果机制真成立（模型懂「类型标记=类型、编号=地址」），这些测试应该过；
如果只是背熟了训练形态，会在这里露馅。

================= 5 组测试 =================
T1 未见问法：训练问句只有「把那个数值交给输出」等 3 种代词模板。
   轻改：换代词（数值包/数字包）；重改：换句式（数值包是哪一个？请交给输出）。
T2 跨类型指代：semantic_pick 训练**只在 NUM 上监督过**，「那个名字」「那段代码」
   等代词从未与「选包」任务组合。测类型前缀定位是否跨类型泛化。
T3 包表容量：训练最多 20 干扰（21 项）。测 25/30 干扰（26/31 项）。
   编号 gap 放宽到 8（400/8=50 格点才放得下 31 项）。
T4 投影长度：训练最长 813 token。测 1000 token（仍在 max_seq_len=1024 内）。
T5 同表多类型切换：一张表 NUM/NAME/CODE 各 1 个 + 3 个异类型干扰，
   同一张表分别问「那个数值/那个名字/那段代码」→ 三个不同答案。
   这是协议里多科室会诊的真实形态：一张病历多种包，按类型词取不同的包。

判分与 30 号评估器同口径：引用整体比较（类型+编号都对才算对）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from tokenizers import AddedToken, Tokenizer

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
ACT_OUTPUT = "<|OUTPUT|>"

PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_COUNT = 401
ID_TOKEN_BASE = 151660

REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]

# 与 31 号训练数据同源的代词表（T2/T5 用；T1 故意用表外的说法）
TYPE_WORD = {
    "NUM": ["那个数值", "刚才那个数值", "用户给的那个数值"],
    "NAME": ["那个名字", "刚才那个名字", "用户给的那个名字"],
    "TEXT": ["那段文本", "刚才那段文本"],
    "CODE": ["那段代码", "刚才那段代码"],
    "ARITH": ["那个算式", "刚才那个算式"],
    "TASK": ["那个任务", "刚才那个任务"],
}

SEED = 20261026


def id_tokens() -> list[str]:
    return [f"#{n}" for n in range(ID_COUNT)]


def load_model(model_dir: Path, device: str):
    """与 30 号评估器一致：注册协议标记 + 401 个编号 token 后加载。"""
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    ids = id_tokens()
    tok.add_special_tokens([
        AddedToken(k, special=True, normalized=False) for k in ids
    ])
    mis = []
    for off, k in enumerate(ids):
        got = tok.encode(k, add_special_tokens=False).ids
        if len(got) != 1 or got[0] != ID_TOKEN_BASE + off:
            mis.append(f"{k} -> {got}")
    if mis:
        raise SystemExit("[错误] 编号注册异常：" + "；".join(mis[:5]))

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    model.to(device).eval()
    return model, tok


def greedy_decode(model, tok, prompt: str, max_new: int, device: str) -> str:
    """贪心解码 + KV cache，遇 eos 即停（与 30 号一致）。"""
    ids = tok.encode(prompt, add_special_tokens=False).ids
    eos_id = tok.token_to_id("<|endoftext|>")
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    out_ids: list[int] = []
    with torch.no_grad():
        for _ in range(max_new):
            if past is None:
                out = model(input_ids=input_ids, use_cache=True)
            else:
                out = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tid = int(next_id.item())
            if tid == eos_id:
                break
            out_ids.append(tid)
    return tok.decode(out_ids, skip_special_tokens=False)


def score_ref(pred: str, expected_ref: str) -> dict:
    """引用整体判分：类型+编号都对才算对（30 号口径）。"""
    e = REF_RE.findall(expected_ref)
    p = REF_RE.findall(pred)
    if not e:
        raise ValueError(f"期望里没有引用：{expected_ref}")
    got = (p[0][0], p[0][1]) if p else (None, None)
    exp = (e[0][0], e[0][1])
    return {"ref_ok": got == exp,
            "type_ok": got[0] == exp[0],
            "got": f"<|PKG_{got[0]}|>#{got[1]}" if got[0] else "<无引用>",
            "expected": expected_ref}


def _ctx(tok: Tokenizer, text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def _grid(rng: random.Random, target: int, count: int, gap: int) -> list[int]:
    """等距网格采样 count 个干扰编号：与 target 差 ≥ gap，彼此差 ≥ gap。"""
    phase = rng.randrange(gap)
    grid = [phase + k * gap for k in range(400 // gap)]
    pool = [g for g in grid if abs(g - target) >= gap]
    if len(pool) < count:
        raise RuntimeError(f"网格池不足：{len(pool)} < {count}")
    return rng.sample(pool, count)


def _make_prompt(task_line: str, package_line: str, question: str) -> str:
    return f"[任务] {task_line}\n[包] {package_line}\n[动作]\n用户说：{question}\n"


def _fill_to(tok: Tokenizer, task: str, package_line: str, question: str,
             min_tokens: int) -> str:
    """填充 [任务] 行直到 prompt ≥ min_tokens（与 31 号的距离控制同构）。"""
    filler_units = ["已确认条目%d。" % i for i in range(1, 1600)]
    parts: list[str] = []
    while True:
        prompt = _make_prompt(f"{task} {' '.join(parts)}".rstrip(),
                              package_line, question)
        if _ctx(tok, prompt) >= min_tokens \
                or len(parts) >= len(filler_units):
            return prompt
        parts.append(filler_units[len(parts)])


# ================= 5 组测试 =================


def t1_unseen_phrasing(tok: Tokenizer) -> list[dict]:
    """T1 未见问法：训练只有「把{那个数值/刚才那个数值/用户给的那个数值}交给输出」。
    轻改换代词，重改换句式。对照档用训练内问法。"""
    rng = random.Random(f"{SEED}-t1")
    cases = []
    # (档名, 问句) —— 对照 2 条 + 轻改 4 条 + 重改 4 条
    questions = [
        ("对照(训练内)", "把那个数值交给输出。"),
        ("对照(训练内)", "把用户给的那个数值交给输出。"),
        ("轻改代词", "把数值包交给输出。"),
        ("轻改代词", "把数字包交给输出。"),
        ("轻改代词", "把那个数字交给输出。"),
        ("轻改代词", "把数交给输出。"),
        ("重改句式", "数值包是哪一个？请交给输出。"),
        ("重改句式", "请找出数值包并交给输出。"),
        ("重改句式", "哪个包装的是数值？把它交给输出。"),
        ("重改句式", "我要数值，把对应的包交给输出。"),
    ]
    for i, (kind, q) in enumerate(questions):
        target = rng.randrange(400)
        distract = _grid(rng, target, 5, 16)
        other = [t for t in PKG_TYPES if t != "NUM"]
        types = [rng.choice(other) for _ in range(5)] + ["NUM"]
        rng.shuffle(types)
        entries, di = [], 0
        for t in types:
            if t == "NUM":
                entries.append((t, target))
            else:
                entries.append((t, distract[di]))
                di += 1
        pkg_line = "  ".join(f"<|PKG_{t}|>#{n}" for t, n in entries)
        prompt = _make_prompt("把指代的对象交给输出", pkg_line, q)
        cases.append({"test": "T1_未见问法", "kind": kind, "prompt": prompt,
                      "expected": f"{ACT_OUTPUT} <|PKG_NUM|>#{target}"})
    return cases


def t2_cross_type(tok: Tokenizer) -> list[dict]:
    """T2 跨类型指代：semantic_pick 训练只在 NUM 上监督过。
    目标类型换成 NAME/CODE/TEXT/ARITH/TASK，代词用 TYPE_WORD 表内词
    （这些词从未与「选包」任务组合过）。每类型 6 条，干扰 5 个。"""
    rng = random.Random(f"{SEED}-t2")
    cases = []
    for goal in ("NAME", "CODE", "TEXT", "ARITH", "TASK"):
        for i in range(6):
            target = rng.randrange(400)
            distract = _grid(rng, target, 5, 16)
            other = [t for t in PKG_TYPES if t != goal]
            types = [rng.choice(other) for _ in range(5)] + [goal]
            rng.shuffle(types)
            entries, di = [], 0
            for t in types:
                if t == goal:
                    entries.append((t, target))
                else:
                    entries.append((t, distract[di]))
                    di += 1
            pkg_line = "  ".join(f"<|PKG_{t}|>#{n}" for t, n in entries)
            pronoun = rng.choice(TYPE_WORD[goal])
            prompt = _make_prompt("把指代的对象交给输出", pkg_line,
                                   f"把{pronoun}交给输出。")
            cases.append({"test": "T2_跨类型指代", "kind": goal, "prompt": prompt,
                          "expected": f"{ACT_OUTPUT} <|PKG_{goal}|>#{target}"})
    return cases


def t3_table_capacity(tok: Tokenizer) -> list[dict]:
    """T3 包表容量：训练最多 20 干扰（21 项）。测 25/30 干扰。
    gap 放宽到 8（400/8=50 格点才放得下 31 项）。"""
    rng = random.Random(f"{SEED}-t3")
    cases = []
    for n_distract in (25, 30):
        for i in range(6):
            target = rng.randrange(400)
            distract = _grid(rng, target, n_distract, 8)
            other = [t for t in PKG_TYPES if t != "NUM"]
            types = [rng.choice(other) for _ in range(n_distract)] + ["NUM"]
            rng.shuffle(types)
            entries, di = [], 0
            for t in types:
                if t == "NUM":
                    entries.append((t, target))
                else:
                    entries.append((t, distract[di]))
                    di += 1
            pkg_line = "  ".join(f"<|PKG_{t}|>#{n}" for t, n in entries)
            prompt = _make_prompt("把指代的对象交给输出", pkg_line,
                                   "把那个数值交给输出。")
            cases.append({"test": "T3_包表容量",
                          "kind": f"{n_distract + 1}项", "prompt": prompt,
                          "expected": f"{ACT_OUTPUT} <|PKG_NUM|>#{target}"})
    return cases


def t4_projection_length(tok: Tokenizer) -> list[dict]:
    """T4 投影长度：训练最长 813 token。测 1000 token（max_seq_len=1024 内）。"""
    rng = random.Random(f"{SEED}-t4")
    cases = []
    for i in range(8):
        target = rng.randrange(400)
        distract = _grid(rng, target, 5, 16)
        other = [t for t in PKG_TYPES if t != "NUM"]
        types = [rng.choice(other) for _ in range(5)] + ["NUM"]
        rng.shuffle(types)
        entries, di = [], 0
        for t in types:
            if t == "NUM":
                entries.append((t, target))
            else:
                entries.append((t, distract[di]))
                di += 1
        pkg_line = "  ".join(f"<|PKG_{t}|>#{n}" for t, n in entries)
        prompt = _fill_to(tok, "把指代的对象交给输出", pkg_line,
                          "把那个数值交给输出。", 1000)
        cases.append({"test": "T4_投影长度", "kind": "1000token",
                      "prompt": prompt,
                      "expected": f"{ACT_OUTPUT} <|PKG_NUM|>#{target}",
                      "ctx_tokens": _ctx(tok, prompt)})
    return cases


def t5_multi_type_switch(tok: Tokenizer) -> list[dict]:
    """T5 同表多类型切换：一张表 NUM/NAME/CODE 各 1 + ARITH/TEXT/TASK 各 1，
    同一张表问三种类型词 → 三个不同答案。多科室会诊的真实形态。"""
    rng = random.Random(f"{SEED}-t5")
    cases = []
    for table_i in range(4):
        # 一张表：6 个编号两两差 ≥16
        ids = _grid(rng, -1, 6, 16)  # target=-1 表示无目标，先取 6 个互距编号
        goals = ["NUM", "NAME", "CODE"]
        fillers = ["ARITH", "TEXT", "TASK"]
        types = goals + fillers
        rng.shuffle(types)
        entries = list(zip(types, ids))
        pkg_line = "  ".join(f"<|PKG_{t}|>#{n}" for t, n in entries)
        for goal in goals:
            n = next(n for t, n in entries if t == goal)
            pronoun = TYPE_WORD[goal][0]
            prompt = _make_prompt("把指代的对象交给输出", pkg_line,
                                   f"把{pronoun}交给输出。")
            cases.append({"test": "T5_同表切换", "kind": f"表{table_i + 1}/{goal}",
                          "prompt": prompt,
                          "expected": f"{ACT_OUTPUT} <|PKG_{goal}|>#{n}"})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description="v10 现场测试：5 组分布外用例")
    ap.add_argument("--model", type=Path,
                    default=Path("output_probe_v10/final_model"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "probe_v10_live_test.json")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, tok = load_model(args.model, device)

    all_cases = (t1_unseen_phrasing(tok) + t2_cross_type(tok)
                 + t3_table_capacity(tok) + t4_projection_length(tok)
                 + t5_multi_type_switch(tok))

    results = []
    print(f"\n=== v10 现场测试：{len(all_cases)} 条（全部为训练分布外）===")
    for c in all_cases:
        pred = greedy_decode(model, tok, c["prompt"], args.max_new, device)
        s = score_ref(pred, c["expected"])
        row = {"test": c["test"], "kind": c["kind"],
               "expected": c["expected"], "pred": pred, **s}
        if "ctx_tokens" in c:
            row["ctx_tokens"] = c["ctx_tokens"]
        results.append(row)
        mark = "✓" if s["ref_ok"] else ("△类型对" if s["type_ok"] else "✗")
        print(f"  {mark} [{c['test']}/{c['kind']}] 期望 {c['expected']}  "
              f"实际 {pred!r}")

    # 汇总
    summary: dict[str, dict] = {}
    for r in results:
        d = summary.setdefault(r["test"], {"n": 0, "ok": 0, "type_only": 0})
        d["n"] += 1
        if r["ref_ok"]:
            d["ok"] += 1
        elif r["type_ok"]:
            d["type_only"] += 1

    print("\n=== 汇总 ===")
    for name, d in summary.items():
        rate = d["ok"] / d["n"]
        extra = f"（另有类型对编号错 {d['type_only']} 条）" if d["type_only"] else ""
        print(f"  {name}: {d['ok']}/{d['n']} = {rate:.2f} {extra}")

    args.out.write_text(
        json.dumps({"model": str(args.model), "device": device,
                    "summary": summary, "results": results},
                   ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(f"\n报告：{args.out}")


if __name__ == "__main__":
    main()
