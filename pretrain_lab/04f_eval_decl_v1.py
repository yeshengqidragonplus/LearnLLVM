# -*- coding: utf-8 -*-
"""04f_eval_decl_v1.py：C/C++ 声明双向数据（decl_bilingual_v1）的评估脚本。

配套 20_prepare_decl_bilingual_v1.py 生成的评估集，按方向分三组打分：

  · code_to_comment（代码→中文注释）：严格=整句答案一致；
    部分=答案里同时出现变量名与初始值（或"没有给出初始值"）。
  · comment_to_code（注释→代码）：严格=生成的声明与标准答案逐字一致；
    部分=以分号结束、含正确变量名、含正确初始值。
  · semicolon（分号语义）：从输出里抽出"是 N 句"，N 必须等于代码分号数。

为什么单独写脚本而不用 04e：04e 是数学 v5 专用（复制/取位/宽度探针），
判据和输入格式完全不同；声明数据的 prompt 带 <|PKG_CODE|>/<|PKG_NUM|>/<|PKG_NAME|>
三个特殊 token，只有用训练时同一份 tokenizer 才能对齐。

用法（必须在 pretrain_lab/ 下运行）：
  .venv\\Scripts\\python 04f_eval_decl_v1.py --model output_decl_v1/final_model
  .venv\\Scripts\\python 04f_eval_decl_v1.py --model output_decl_v1/final_model --show-fails 5
  .venv\\Scripts\\python 04f_eval_decl_v1.py --model output_decl_v1/final_model --limit 6 --max-new-tokens 32

解码约定与训练一致：输入是 [eos] + prompt，贪心解码到 eos 或 max-new-tokens。
评估集路径默认 data/decl_bilingual_v1_decl_eval.jsonl（由 20 号脚本生成）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from runtime import add_runtime_args, resolve_runtime

# 原子包标记：解包时把标记替换回本条载荷，与 20 号数据生成器同一约定
PKG_NUM = "<|PKG_NUM|>"
PKG_NAME = "<|PKG_NAME|>"

DATA_DEFAULT = "data/decl_bilingual_v1_decl_eval.jsonl"


def normalize(text: str) -> str:
    """归一化比较用文本：压缩空白、去掉首尾空白。"""
    return re.sub(r"\s+", " ", (text or "").strip())


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ================= 打分函数（每个方向一组） =================

def unpack(text: str, row: dict) -> str:
    """把模型输出里的原子包标记替换回本条载荷 —— 与训练数据的 target 约定一一对应。

    模型学的是"在哪个位置放哪个标记"（零字符复制）；评估时必须先解包，
    再与明文期望比对，否则正确输出也会被判错。
    标记残留说明模型没用标记（可能直接抄了明文，或输出乱码），原样保留以便观察。
    """
    name = row.get("name")
    lit = row.get("lit")
    out = text or ""
    if name:
        out = out.replace(PKG_NAME, name)
    if lit:
        out = out.replace(PKG_NUM, lit)
    return out


def expected_plain(row: dict) -> str:
    """明文期望：优先用数据里存的 expected_unpacked，旧数据回退到 target。"""
    return row.get("expected_unpacked") or row["target"]


def score_code_to_comment(out: str, row: dict) -> dict:
    """代码→注释：严格整句一致；部分要求名字 + 初始值（或无初值时的那句说明）。"""
    unpacked = unpack(out, row)
    out_n, tgt_n = normalize(unpacked), normalize(expected_plain(row))
    strict = out_n.startswith(tgt_n) or tgt_n in out_n
    name = row.get("name") or ""
    lit = row.get("lit")
    has_name = bool(name) and name in unpacked
    if lit is None:
        has_value = "没有给出初始值" in unpacked or "不给初始值" in unpacked
    else:
        has_value = lit in unpacked
    return {"strict": strict, "partial": has_name and has_value,
            "has_name": has_name, "has_value": has_value,
            "unpacked": unpacked}


def score_comment_to_code(out: str, row: dict) -> dict:
    """注释→代码：严格逐字一致；部分要求以分号结尾 + 名字 + 初始值。"""
    unpacked = unpack(out, row)
    target = expected_plain(row)
    # 取第一行非空内容（生成可能带换行/后续解释）
    first = ""
    for line in unpacked.splitlines():
        if line.strip():
            first = line.strip()
            break
    strict = normalize(first) == normalize(target)
    name = row.get("name") or ""
    lit = row.get("lit")
    has_name = bool(name) and name in first
    has_lit = lit is None or lit in first
    return {"strict": strict,
            "partial": first.endswith(";") and has_name and has_lit,
            "has_name": has_name, "has_value": has_lit,
            "unpacked": unpacked}


SEMI_RE = re.compile(r"是\s*(\d+)\s*句")


def score_semicolon(out: str, row: dict) -> dict:
    """分号语义：输出里的"是 N 句"必须等于代码的分号数。"""
    expected = row.get("n_semicolons")
    if expected is None:
        expected = (row.get("code") or "").count(";")
    match = SEMI_RE.search(out or "")
    correct = bool(match) and int(match.group(1)) == expected
    return {"strict": correct, "partial": correct,
            "has_name": True, "has_value": True}


SCORERS = {
    "code_to_comment": score_code_to_comment,
    "comment_to_code": score_comment_to_code,
    "semicolon": score_semicolon,
}
DIR_LABELS = {
    "code_to_comment": "代码→中文注释",
    "comment_to_code": "中文注释→代码",
    "semicolon": "分号语义（几句话）",
}


# ================= 主流程 =================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估 C/C++ 声明双向数据（decl_bilingual_v1）")
    parser.add_argument("--model", required=True, help="训练输出的模型目录（含 tokenizer）")
    parser.add_argument("--data", type=Path, default=Path(DATA_DEFAULT),
                        help=f"评估 jsonl 路径（默认 {DATA_DEFAULT}）")
    parser.add_argument("--max-new-tokens", type=int, default=96,
                        help="每条最多生成 token 数（答案最长约 60 token，默认 96 足够）")
    parser.add_argument("--limit", type=int, default=0,
                        help="只评估前 N 条（0=全部；环境自检可用 6）")
    parser.add_argument("--show-fails", type=int, default=3,
                        help="每个方向打印几条失败样例（0=不打印）")
    parser.add_argument("--out", type=Path, default=None,
                        help="可选：把逐条结果写成 JSON（默认不写）")
    add_runtime_args(parser)
    args = parser.parse_args()

    device, dtype = resolve_runtime(args.device, args.dtype)

    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = Qwen2ForCausalLM.from_pretrained(args.model).to(dtype).to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 三个 Package 标记必须仍是单 token，否则说明模型与数据词表不对齐
    marker_ids = {}
    for marker in ("<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>"):
        ids = tokenizer.encode(marker, add_special_tokens=False)
        marker_ids[marker] = ids
    if any(len(ids) != 1 for ids in marker_ids.values()):
        raise SystemExit(
            f"[错误] Package 标记不是单 token：{marker_ids}。"
            f"训练时应加 --package-code-token --package-nested-tokens，"
            f"且 --model 目录里的 tokenizer 应包含这三个 token。")

    rows = load_rows(args.data)
    if args.limit:
        rows = rows[:args.limit]

    print(f"模型：{args.model}")
    print(f"设备：{device}，精度：{dtype}")
    print(f"评估集：{args.data}（{len(rows)} 条）")
    print(f"标记 id：{ {k: v[0] for k, v in marker_ids.items()} }\n")

    eos = tokenizer.eos_token_id
    by_dir = defaultdict(lambda: {"total": 0, "strict": 0, "partial": 0})
    details: list[dict] = []

    for row in rows:
        prompt = row["prompt"]
        ids = [eos] + tokenizer.encode(prompt, add_special_tokens=False)
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            out_ids = model.generate(
                x, max_new_tokens=args.max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=eos)
        # ⚠️ 必须 skip_special_tokens=False：
        # <|PKG_NAME|>/<|PKG_NUM|> 是 special token，跳过它们会把标记直接从输出里
        # 抹掉，模型明明正确输出了标记，却只看到 "声明一个整型变量 ，初始值是 "。
        # 这会制造"模型学会了但评估判 0"的假失败。
        out = tokenizer.decode(out_ids[0][len(ids):], skip_special_tokens=False)
        # 只清掉 eos/pad 这类真正的序列控制符；包标记必须保留到解包那一步
        for special in (tokenizer.eos_token, tokenizer.pad_token):
            if special:
                out = out.replace(special, "")
        out = out.strip()

        direction = row.get("direction", "unknown")
        scorer = SCORERS.get(direction)
        if scorer is None:
            continue
        marks = scorer(out, row)
        bucket = by_dir[direction]
        bucket["total"] += 1
        bucket["strict"] += int(marks["strict"])
        bucket["partial"] += int(marks["partial"])
        details.append({"direction": direction, "form": row.get("form"),
                        "code": row.get("code"), "expected": row["target"],
                        "output": out, **marks})

        tag = "OK " if marks["partial"] else "FAIL"
        print(f"  [{tag}] {direction:<16} {str(row.get('code'))[:44]:<46} -> {out[:60]!r}")

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("汇总（严格=整句/整行逐字一致；部分=关键字段正确）")
    total = strict = partial = 0
    for direction, label in DIR_LABELS.items():
        bucket = by_dir.get(direction)
        if not bucket or not bucket["total"]:
            continue
        n = bucket["total"]
        total += n
        strict += bucket["strict"]
        partial += bucket["partial"]
        print(f"  {label:<20} {n:>3} 条 | 严格 {bucket['strict']:>3}/{n:<3} "
              f"= {100.0 * bucket['strict'] / n:5.1f}% | "
              f"部分 {bucket['partial']:>3}/{n:<3} = {100.0 * bucket['partial'] / n:5.1f}%")
    if total:
        print(f"  {'合计':<20} {total:>3} 条 | 严格 {strict:>3}/{total:<3} "
              f"= {100.0 * strict / total:5.1f}% | "
              f"部分 {partial:>3}/{total:<3} = {100.0 * partial / total:5.1f}%")

    if args.show_fails:
        print("\n失败样例（按方向）")
        shown: dict[str, int] = defaultdict(int)
        for item in details:
            if item["partial"] or shown[item["direction"]] >= args.show_fails:
                continue
            shown[item["direction"]] += 1
            print(f"  [{item['direction']} / {item['form']}]")
            print(f"    期望：{item['expected'][:150]}")
            print(f"    实际：{item['output'][:150]!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "model": args.model, "data": str(args.data),
            "device": device, "dtype": str(dtype).removeprefix("torch."),
            "counts": {d: dict(v) for d, v in by_dir.items()},
            "details": details,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\n逐条结果已写入 {args.out}")


if __name__ == "__main__":
    main()