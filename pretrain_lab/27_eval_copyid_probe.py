# -*- coding: utf-8 -*-
"""评估编号复制探针：模型能否准确复现投影里出现过的 `#N`。

================= 判分必须严到编号 =================
`score()` 分三档，刻意拆开，因为**混淆它们正是 v8 犯的错**：

  strict   完整 token 序列逐字相同（`<|OUTPUT|> <|PKG_NUM|>#1247`）
  id_ok    编号数字完全一致（1247 == 1247），类型对不对另算
  type_ok  类型对了、编号错了 —— **这类必须单独报出来**

如果只报「类型对了」，那「编号读错」会被掩盖成「对」，看起来 1.00 实际无效。
所以本脚本把 `type_ok_id_wrong` 也当作**失败**，绝不并进正确率。

================= 结论怎么读 =================
  no_id_baseline ≈ 1.00      零复制的下限（模型能输出裸类型标记）
  copy_id/d1..d4 都是 ≈1.00  → 复制上文出现过的短串可行，编号方案成立
  copy_id 随位数下降          → 仍有复制墙，必须限制位数或改单 token 编号
  特别是 d4（1001 个值）      → 无法靠记忆，只能靠复制，是**真正的墙**
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

REF_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>#(\d+)")
BARE_RE = re.compile(r"<\|PKG_(NUM|NAME|TEXT|CODE|ARITH|TASK)\|>(?!#?\d)")

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]


def load_model(model_dir: Path, device: str):
    from transformers import AutoModelForCausalLM

    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.add_special_tokens([
        AddedToken(t, special=True, normalized=False) for t in PROTOCOL_TOKENS
    ])
    bad = [t for t in PROTOCOL_TOKENS
           if len(tok.encode(t, add_special_tokens=False).ids) != 1]
    if bad:
        raise SystemExit(f"[错误] 标记未成为单 token：{bad[:3]}")

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=dtype)
    model.to(device).eval()
    return model, tok


def greedy_decode(model, tok, prompt: str, max_new: int, device: str) -> str:
    """贪心解码 + KV cache，遇 eos 即停。"""
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


def score(pred: str, expected: str) -> dict:
    """返回四档判分，刻意不让「类型对但编号错」混进正确率。"""
    strict = pred.strip() == expected.strip()

    p_refs = REF_RE.findall(pred)
    e_refs = REF_RE.findall(expected)
    p_bare = BARE_RE.findall(pred)
    e_bare = BARE_RE.findall(expected)

    if e_refs:
        e_type, e_id = e_refs[0]
        got_id = p_refs[0][1] if p_refs else None
        got_type = p_refs[0][0] if p_refs else (
            p_bare[0] if p_bare else None)
        id_ok = got_id == e_id
        type_ok = got_type == e_type
        ref_ok = bool(p_refs) and p_refs[0] == (e_type, e_id)
    else:
        e_type = e_bare[0]
        ref_ok = bool(p_bare) and p_bare[0] == e_type
        id_ok = ref_ok
        type_ok = ref_ok
        got_id = None

    return {
        "strict": strict,
        "ref_ok": ref_ok,               # 引用整体一致（类型+编号，或无编号时只看类型）
        "id_ok": id_ok,
        "type_ok": type_ok,
        "type_ok_id_wrong": type_ok and not id_ok,   # ← 必须单列，绝不算对
    }


def _level_key(item: tuple[str, dict]) -> tuple[int, str]:
    lv = item[0]
    return (0, f"{int(lv):06d}") if lv.isdigit() else (1, lv)


def evaluate(model, tok, rows: list[dict], device: str, max_new: int) -> dict:
    per_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    samples: list[dict] = []
    for r in rows:
        pred = greedy_decode(model, tok, r["prompt"], max_new, device)
        s = score(pred, r["target"])
        per_cell[(r["axis"], r["level"])].append(s)
        samples.append({"axis": r["axis"], "level": r["level"],
                        "expected": r["target"], "pred": pred, **s})

    curves: dict[str, dict[str, dict]] = defaultdict(dict)
    for (axis, level), vals in per_cell.items():
        n = len(vals)
        curves[axis][level] = {
            "n": n,
            "strict": round(sum(v["strict"] for v in vals) / n, 4),
            "ref_ok": round(sum(v["ref_ok"] for v in vals) / n, 4),
            "type_ok": round(sum(v["type_ok"] for v in vals) / n, 4),
            "type_ok_id_wrong": round(
                sum(v["type_ok_id_wrong"] for v in vals) / n, 4),
        }
    return {"curves": dict(curves), "samples": samples}


def _verdict(curves: dict, samples: list[dict]) -> dict:
    out: dict = {"statements": []}
    st = out["statements"]

    base = curves.get("no_id_baseline", {})
    base_mean = (round(sum(v["ref_ok"] * v["n"] for v in base.values())
                       / sum(v["n"] for v in base.values()), 4)
                 if base else None)

    cid = curves.get("copy_id", {})
    by_d = {lv: v["ref_ok"] for lv, v in sorted(cid.items(), key=_level_key)}
    out["copy_id_ref_ok"] = by_d
    out["baseline_ref_ok"] = base_mean

    if base_mean is not None:
        st.append(
            f"零复制基线（投影里只有 1 个包、target 不带编号）：ref_ok {base_mean:.2f}。"
            f"这是「不需要复制」时的上限，后面所有档都要和它比。")

    if by_d:
        worst_lv = min(by_d, key=lambda k: by_d[k])
        best_lv = max(by_d, key=lambda k: by_d[k])
        st.append(
            "编号复制四档（ref_ok）："
            + "  ".join(f"{lv}位={v:.2f}" for lv, v in by_d.items()))
        if base_mean is not None and by_d[best_lv] >= base_mean - 0.10 \
                and by_d[worst_lv] >= 0.80:
            st.append(
                f"**复制上文出现过的编号可行**：即使 4 位档（#1000..#2000，"
                f"共 1001 个值，无法靠记忆）也有 {by_d.get('4', float('nan')):.2f}。"
                f"说明模型做的是「从上下文指回去」（copy）而不是「从参数生成」，"
                f"与载荷复制墙不是同一个问题。编号方案成立。")
        elif by_d[worst_lv] < 0.60:
            st.append(
                f"**仍存在复制墙**：{worst_lv} 位档只有 {by_d[worst_lv]:.2f}。"
                f"编号越长越差，说明多位数字串无法稳定复现。"
                f"→ 必须限制编号位数，或把编号改成单 token。")
        else:
            st.append(
                f"**部分可行**：{worst_lv} 位档 {by_d[worst_lv]:.2f}，"
                f"未达基线 {base_mean:.2f}。需看错例判断是位数问题还是样本量问题。")

        # 逐位退化趋势
        lvs = sorted(by_d, key=int)
        if len(lvs) >= 2:
            trend = [by_d[lv] for lv in lvs]
            if all(a >= b - 0.02 for a, b in zip(trend, trend[1:])):
                st.append("趋势：编号变长后分数**单调不降**，位数不是瓶颈。")
            else:
                st.append(f"趋势：随位数下降 {trend}，位数是影响因素。")

    # 关键：把「类型对但编号错」显式报出来，防止假阳性
    for lv, v in sorted(cid.items(), key=_level_key):
        if v["type_ok_id_wrong"] > 0.05:
            st.append(
                f"注意 {lv} 位档有 {v['type_ok_id_wrong'] * 100:.0f}% 的样本"
                f"**类型对、编号错**。这类绝不能算对——它意味着替换时会把载荷"
                f"填到错误位置。")

    # 错例形态
    wrong = [s for s in samples
             if s["axis"] == "copy_id" and not s["ref_ok"]]
    if wrong:
        c = Counter()
        for s in wrong:
            ids = [n for _t, n in REF_RE.findall(s["pred"])]
            c[ids[0] if ids else "<无编号输出>"] += 1
        out["wrong_shapes"] = c.most_common(8)

    return out


def _load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser(description="编号复制探针评估")
    ap.add_argument("--model", type=Path,
                    default=Path("output_copyid_probe_v1/final_model"))
    ap.add_argument("--data", type=Path, action="append", default=None)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, tok = load_model(args.model, device)

    targets = args.data or [DATA_DIR / "copyid_probe_holdout.jsonl"]
    report: dict = {"model": str(args.model), "device": device, "splits": {}}

    for path in targets:
        rows = _load(path)
        print(f"\n=== {path.name}：{len(rows)} 条 ===")
        res = evaluate(model, tok, rows, device, args.max_new)
        for axis, levels in res["curves"].items():
            print(f"  {axis}: " + "  ".join(
                f"{lv}=strict{v['strict']:.2f}/ref{v['ref_ok']:.2f}"
                f"/类型{v['type_ok']:.2f}"
                for lv, v in sorted(levels.items(), key=_level_key)))
        verdict = _verdict(res["curves"], res["samples"])
        for line in verdict["statements"]:
            print("  · " + line)
        report["splits"][path.name] = {
            "n": len(rows), "curves": res["curves"],
            "verdict": verdict, "samples": res["samples"],
        }

    out = args.out or DATA_DIR / "copyid_probe_eval.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\n报告：{out}")


if __name__ == "__main__":
    main()
