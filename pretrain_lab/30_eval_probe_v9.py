# -*- coding: utf-8 -*-
"""评估包引用探针 v9：定稿形态 `类型标记 + 整体编号 token`。

================= 判分口径 =================
`answer_kind` 决定答案形态：

  "type" → 期望 `<|OUTPUT|> <|PKG_NUM|>`     答案是一个类型标记
  "ref"  → 期望 `<|OUTPUT|> <|PKG_NUM|>#123` 答案是一个包引用（2 token）

引用一律**整体**比较：类型标记与编号必须都对。只对类型不算对——
那正是 v8 犯过的错（把「编号读错」掩盖成「对」）。

三档分开报，绝不合并：

  strict        完整 token 序列逐字相同
  ref_ok        引用整体一致（类型 + 编号都对）
  type_ok_only  类型对、编号错 ← **必须单列**，这类意味着替换会填错位置
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter, defaultdict
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

PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_COUNT = 401

REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
BARE_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!#\d)")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

ID_TOKEN_BASE = 151660

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]


def id_tokens() -> list[str]:
    return [f"#{n}" for n in range(ID_COUNT)]


def load_model(model_dir: Path, device: str):
    """加载模型与 tokenizer，注册同一套协议标记 + 401 个编号 token。"""
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
        if len(got) != 1:
            mis.append(f"{k} 不是单 token：{got}")
        elif got[0] != ID_TOKEN_BASE + off:
            mis.append(f"{k} id={got[0]} 应为 {ID_TOKEN_BASE + off}")
    if mis:
        raise SystemExit("[错误] 编号注册异常：" + "；".join(mis[:5]))

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    model.to(device).eval()
    return model, tok


def greedy_decode(model, tok, prompt: str, max_new: int, device: str) -> str:
    """贪心解码 + KV cache，遇 eos 即停。

    必须用 KV cache：长投影逐 token 全量前向在 8GB 卡上会 OOM。
    必须遇 eos 停：否则答对后继续吐垃圾，strict 误判为 0。
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


def score(pred: str, expected: str, kind: str) -> dict:
    """返回 strict / ref_ok / type_ok_only。"""
    strict = pred.strip() == expected.strip()

    if kind == "type":
        p = set(TYPE_MARK_RE.findall(pred))
        e = set(TYPE_MARK_RE.findall(expected))
        type_ok = bool(p & e)
        ref_ok = type_ok
        type_only_wrong = False
    else:
        p_ref = REF_RE.findall(pred)
        e_ref = REF_RE.findall(expected)
        if not e_ref:
            # 期望里没有引用（理论上不会出现在 ref 轴）
            ref_ok = type_ok = False
            type_only_wrong = False
        else:
            e_type, e_id = e_ref[0]
            got_type = p_ref[0][0] if p_ref else None
            got_id = p_ref[0][1] if p_ref else None
            ref_ok = (got_type, got_id) == (e_type, e_id)
            type_ok = got_type == e_type
            # 类型对但编号错：这类必须单独报，绝不能算对
            type_only_wrong = type_ok and not ref_ok

    return {"strict": strict, "ref_ok": ref_ok, "type_ok": type_ok,
            "type_ok_only": type_only_wrong}


def _level_key(item: tuple[str, dict]) -> tuple[int, str]:
    lv = item[0]
    return (0, f"{int(lv):06d}") if lv.isdigit() else (1, lv)


def evaluate(model, tok, rows: list[dict], device: str, max_new: int) -> dict:
    per_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    samples: list[dict] = []
    for r in rows:
        pred = greedy_decode(model, tok, r["prompt"], max_new, device)
        s = score(pred, r["target"], r["answer_kind"])
        per_cell[(r["axis"], r["level"])].append(s)
        samples.append({"axis": r["axis"], "level": r["level"],
                        "kind": r["answer_kind"],
                        "expected": r["target"], "pred": pred, **s})

    curves: dict[str, dict[str, dict]] = defaultdict(dict)
    for (axis, level), vals in per_cell.items():
        n = len(vals)
        curves[axis][level] = {
            "n": n,
            "strict": round(sum(v["strict"] for v in vals) / n, 4),
            "ref_ok": round(sum(v["ref_ok"] for v in vals) / n, 4),
            "type_ok": round(sum(v["type_ok"] for v in vals) / n, 4),
            "type_ok_only": round(sum(v["type_ok_only"] for v in vals) / n, 4),
        }
    return {"curves": dict(curves), "samples": samples}


def axis_mean(curves: dict, axis: str, key: str = "ref_ok") -> float | None:
    lv = curves.get(axis)
    if not lv:
        return None
    tot = sum(v["n"] for v in lv.values())
    return round(sum(v[key] * v["n"] for v in lv.values()) / tot, 4)


def _verdict(curves: dict, samples: list[dict]) -> dict:
    out: dict = {"statements": []}
    st = out["statements"]

    k2 = axis_mean(curves, "type_id_choice_k2")
    k6 = axis_mean(curves, "type_id_choice_k6")
    op = axis_mean(curves, "type_id_open")
    un = axis_mean(curves, "type_id_unseen")
    pick = axis_mean(curves, "semantic_pick")
    dist = curves.get("distance", {})

    out["numbers"] = {"type_id_choice_k2": k2, "type_id_choice_k6": k6,
                      "type_id_open": op, "type_id_unseen": un,
                      "semantic_pick": pick}

    # ---- 类型辨识四档 ----
    if k2 is not None:
        st.append(
            f"**类型辨识**（引用 = 类型标记 + 整体编号 token）："
            f"给 2 个候选 {k2:.2f}、给 6 个候选 {k6:.2f}、不给候选 {op:.2f}。"
            f"四档配比完全相同，唯一变量是那一行 [候选]。")
        if op is not None and op >= 0.85:
            st.append(
                "→ 开集输出类型标记**可行**，协议里不必为每个类型枚举候选。")

    # ---- 负对照 ----
    if un is not None:
        base = 1 / len(PKG_TYPES)
        st.append(
            f"**未训练编号的负对照**：{un:.2f}（随机基线 {base:.2f}）。"
            f"编号 #400 一次都没进过训练集 —— 这是「编号必须整体编码成 token」"
            f"的固有代价：新编号 token 的 embedding 是随机的。"
            f"→ 协议约束：编号池必须先声明再训练；扩容要补训练。")

    # ---- 语义选取 ----
    if pick is not None:
        st.append(
            f"**语义指代**：包表里唯一的 NUM 用「那个数值」选出，跨档 {pick:.2f}。"
            f"干扰项全是异类型，所以靠类型前缀定位，不数位置。")

    # ---- 距离 ----
    if dist:
        broken = [lv for lv, v in sorted(dist.items(), key=_level_key)
                  if v["ref_ok"] < 0.9]
        if broken:
            st.append(f"**投影长度**：从 {broken[0]} token 档开始跌破 0.90。")
        else:
            st.append(f"**投影长度**：至 {max(int(l) for l in dist)} token "
                      f"仍全对，未探到上限。")

    # ---- 关键：把「类型对编号错」显式报出来 ----
    bad = [(a, lv, v["type_ok_only"]) for a, levels in curves.items()
           for lv, v in levels.items() if v["type_ok_only"] > 0.05]
    if bad:
        out["type_ok_only_cells"] = bad
        st.append(
            "注意：以下格子存在「类型对、编号错」"
            + "、".join(f"{a}/{lv}={v:.2f}" for a, lv, v in bad[:5])
            + " —— 这类**不能算对**，它意味着替换时会把载荷填到错误位置。")

    # ---- 错例形态 ----
    wrong = [s for s in samples if s["kind"] == "ref" and not s["ref_ok"]]
    if wrong:
        c = Counter()
        for s in wrong:
            ids = [n for _t, n in REF_RE.findall(s["pred"])]
            c[ids[0] if ids else "<无引用输出>"] += 1
        out["wrong_shapes"] = c.most_common(8)

    return out


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser(description="包引用探针 v9 评估")
    ap.add_argument("--model", type=Path,
                    default=Path("output_probe_v9/final_model"))
    ap.add_argument("--data", type=Path, action="append", default=None)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, tok = load_model(args.model, device)

    targets = args.data or [DATA_DIR / "probe_v9_holdout.jsonl"]
    rep: dict = {"model": str(args.model), "device": device,
                 "package_form": "<|PKG_<TYPE>|>#<ID>（编号为整体编码 token）",
                 "reference_tokens": 2, "splits": {}}

    for path in targets:
        rows = _load(path)
        print(f"\n=== {path.name}：{len(rows)} 条 ===")
        res = evaluate(model, tok, rows, device, args.max_new)
        for axis, levels in res["curves"].items():
            print(f"  {axis}: " + "  ".join(
                f"{lv}=strict{v['strict']:.2f}/ref{v['ref_ok']:.2f}"
                f"/类型{v['type_ok']:.2f}"
                for lv, v in sorted(levels.items(), key=_level_key)))
        v = _verdict(res["curves"], res["samples"])
        for line in v["statements"]:
            print("  · " + line)
        rep["splits"][path.name] = {"n": len(rows), "curves": res["curves"],
                                    "verdict": v, "samples": res["samples"]}

    hold = next((k for k in rep["splits"] if "holdout" in k), None)
    if hold:
        rep["conclusion_from"] = hold

    out = args.out or DATA_DIR / "probe_v9_eval.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
