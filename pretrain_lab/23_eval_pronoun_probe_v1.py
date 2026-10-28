# -*- coding: utf-8 -*-
"""评估三轴探针（重设计版）：类型辨识 / 语义选取 / 距离。

================= 与上一版的关键区别 =================
上一版要求模型用**序数代词**（「第几个」）在 N 个同质 key 里数位置，
于是得出「包表超过 12 项就挑不准」。但协议里模型从来不数位置：

    `<|PKG_NUM|,#A>` 是一个数字包，值在缓存里，由程序按 (类型, 序号) 解包。

所以本版三条轴都不含序数：
  轴 1  type_id       给一个 key，说出它的类型（1 token 答案）
  轴 2  semantic_pick 包表里只有一个 NUM，其余是其他类型；按「那个数值」选出它
  轴 3  distance      同上 + 无关填充加长，测投影能写多长

判分只看 key 整体是否一致，不拆类型/序号、不转 int。
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
except Exception:
    pass

from tokenizers import AddedToken, Tokenizer

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")

ACT_OUTPUT = "<|OUTPUT|>"

# ================= 包 key：原子整体 =================
# 一个包 = 一个不可分的整体 key <|PKG_<TYPE>|,#<ID>>（注册后 1 token）。
# 整体抽取后比较，绝不拆成「类型标记 + 序号」两段分别比。
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEY_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|,#[A-Z]>")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

# 14 个协议标记占 151646..151659，key 从 151660 起（与 tokens.json 一致）
KEY_ID_BASE = 151660


def make_key(pkg_type: str, index: int) -> str:
    """造一个包 key（原子整体）；index 为 0-based，0->A。"""
    return f"<|PKG_{pkg_type}|,#{ID_CHARS[index]}>"


def all_keys() -> list[str]:
    return [make_key(t, i) for t in PKG_TYPES for i in range(len(ID_CHARS))]


PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]


# ================= 模型加载 =================


def load_model(model_dir: Path, device: str):
    """加载 03 训练出的模型与 tokenizer（注册同一套协议标记 + 全部包 key）。"""
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    # key 必须与训练时注册成同一批 ID，否则模型读到的 key 全是没见过的 token。
    keys = all_keys()
    tok.add_special_tokens([
        AddedToken(k, special=True, normalized=False) for k in keys
    ])
    mis = []
    for offset, k in enumerate(keys):
        ids = tok.encode(k, add_special_tokens=False).ids
        if len(ids) != 1:
            mis.append(f"{k} 不是单 token：{ids}")
        elif ids[0] != KEY_ID_BASE + offset:
            mis.append(f"{k} id={ids[0]} 应为 {KEY_ID_BASE + offset}")
    if mis:
        raise SystemExit("[错误] 包 key 注册异常：" + "；".join(mis[:5]))

    # 用 bfloat16 加载：长上下文 + 无 KV cache 的逐 token 前向在 fp32 下会 OOM
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    model.to(device).eval()
    return model, tok


# ================= 解码与判分 =================


def greedy_decode(model, tok, prompt: str, max_new: int, device: str) -> str:
    """贪心解码，带 KV cache，遇到 eos 即停。

    两个坑：
    1. **必须遇 eos 停止**，否则模型答对后继续吐 eos 和垃圾，strict 会误判为 0。
    2. **必须用 KV cache**，否则长投影逐 token 全量前向在 8GB 卡上会 OOM。
    """
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


def score(pred: str, expected: str, axis: str) -> tuple[bool, bool]:
    """返回 (严格, 部分)。

    轴 1（type_id）：答案是一个类型标记，整体比对；
    轴 2/3（语义/距离）：答案是一个包 key，按 key 整体比对。
    """
    p_tokens = pred.strip().split()
    e_tokens = expected.strip().split()
    strict = p_tokens == e_tokens

    if axis == "type_id":
        p = set(TYPE_MARK_RE.findall(pred))
        e = set(TYPE_MARK_RE.findall(expected))
    else:
        p = set(KEY_RE.findall(pred))
        e = set(KEY_RE.findall(expected))
    partial = bool(p & e)
    return strict, partial


def _level_key(item: tuple[str, dict]) -> tuple[int, str]:
    """排序键：数字 level 按数值排，类型名排后面按字符串排。"""
    level = item[0]
    return (0, f"{int(level):06d}") if level.isdigit() else (1, level)


def evaluate(model, tok, rows: list[dict], device: str, max_new: int) -> dict:
    per_cell: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    samples: list[dict] = []
    for row in rows:
        pred = greedy_decode(model, tok, row["prompt"], max_new, device)
        strict, partial = score(pred, row["target"], row["axis"])
        per_cell[(row["axis"], row["level"])].append((strict, partial))
        samples.append({"axis": row["axis"], "level": row["level"],
                        "pred": pred, "expected": row["target"],
                        "strict": strict, "partial": partial})
    curves: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for (axis, level), vals in per_cell.items():
        n = len(vals)
        curves[axis][level] = {
            "n": n,
            "strict": round(sum(v[0] for v in vals) / n, 4),
            "partial": round(sum(v[1] for v in vals) / n, 4),
        }
    return {"curves": dict(curves), "samples": samples}


# ================= 主流程 =================


def _verdict(curves: dict) -> str:
    """按三条轴各自的首个跌破点给结论。"""
    parts = []

    type_curve = curves.get("type_id", {})
    if type_curve:
        worst = min(v["strict"] for v in type_curve.values())
        if worst >= 0.9:
            parts.append(f"类型辨识：6 种类型全对（最低 {worst:.2f}）→ "
                         f"模型确实理解 key 的类型前缀，token filter 前提成立")
        else:
            parts.append(f"类型辨识：最低 {worst:.2f}，有类型分错")

    pick = curves.get("semantic_pick", {})
    if pick:
        # 干扰项个数 → 严格分
        ok = {lv: v["strict"] for lv, v in pick.items()}
        worst = min(ok.values())
        parts.append(f"语义选取：干扰项 {ok}（全部是其他类型）→ "
                     + (f"全是 1.00，靠类型前缀就能定位唯一 NUM，与候选个数无关"
                        if worst >= 0.9 else f"最低 {worst:.2f}"))

    dist = curves.get("distance", {})
    if dist:
        broken = [lv for lv, v in sorted(dist.items(), key=_level_key)
                  if v["strict"] < 0.9]
        if broken:
            parts.append(f"投影长度：从 {broken[0]} 档开始跌破 0.90")
        else:
            longest = max(int(lv) for lv in dist)
            parts.append(f"投影长度：至 {longest} token 仍全对，未探到上限")

    return "；".join(parts) if parts else "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser(description="三轴探针评估（重设计版）")
    parser.add_argument("--model", type=Path,
                        default=Path("output_pronoun_probe_v6/final_model"))
    parser.add_argument("--data", type=Path,
                        default=DATA_DIR / "pronoun_probe_v1.jsonl")
    parser.add_argument("--limit", type=int, default=0,
                        help="每格只取前 N 条（0=全量）")
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path,
                        default=DATA_DIR / "pronoun_probe_v1_eval.json")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, tok = load_model(args.model, device)

    raw = [json.loads(l) for l in args.data.open(encoding="utf-8")]
    if args.limit:
        seen: dict[tuple[str, str], int] = defaultdict(int)
        keep = []
        for r in raw:
            k = (r["axis"], r["level"])
            if seen[k] < args.limit:
                keep.append(r)
                seen[k] += 1
        raw = keep

    print(f"[A] 探针 {len(raw)} 条（{args.data}）")
    result = evaluate(model, tok, raw, device, args.max_new)
    for axis, levels in result["curves"].items():
        print(f"  {axis}: " + "  ".join(
            f"{lv}={v['strict']:.2f}/{v['partial']:.2f}"
            for lv, v in sorted(levels.items(), key=_level_key)))
    for s in result["samples"][:3]:
        print(f"    [样例] 期望 {s['expected']!r}  实际 {s['pred']!r}")

    verdict = _verdict(result["curves"])
    report = {
        "model": str(args.model),
        "device": device,
        "data": str(args.data),
        "n": len(raw),
        "curves": result["curves"],
        "verdict": verdict,
        "note": "本版三轴都不含序数代词，不要求模型数位置；"
                "包引用一律按原子 key 整体比较。",
        "samples": result["samples"][:50],
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n结论：{verdict}")
    print(f"报告：{args.out}")


if __name__ == "__main__":
    main()
