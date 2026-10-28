# -*- coding: utf-8 -*-
"""评估包引用探针 v8：按 answer_kind 判分，并给出四档类型辨识的对照结论。

================= 判分口径（只有两条） =================
`answer_kind` 决定「答案有几个 token、长什么样」：

  "type" → 期望 `<|OUTPUT|> <|PKG_NUM|>`     答案是一个类型标记
  "key"  → 期望 `<|OUTPUT|> <|PKG_NUM|,#B>`  答案是一个包 key

比对一律把 **key 整体** 用正则抽出来当集合比，绝不拆成「类型标记 + 序号」
两段分别比、也不转 int —— 那样会把「类型读错」和「序号读错」混成一个数，
而且原子 key 本来就不可分。

================= 为什么要区分 strict / partial =================
strict  = 生成的 token 序列与期望逐字相同
partial = 期望的那个引用对象出现在输出里（顺序/多余内容不计）

partial 存在的意义：模型偶尔会多吐一个 eos 或重复输出，strict 归零但
「它其实指向了正确对象」。协议里真正要紧的是**指向对不对**，所以两个数都报，
但结论以 strict 为准（更严格，不会自欺）。

================= 四档对照的读法 =================
type_id_choice_k2 / _k6 / _open / _unseen 四档**配比完全相同**（每种类型
同样条数、同样模型、同一次训练），唯一变量是「有没有给候选」和
「这个 key 有没有在训练里出现过」。所以：

  k2/k6 高、open 低   → 开集生成是瓶颈 → 协议里必须给候选枚举
  k2/k6/open 都高      → v7 的低分纯属配比不足，加量解决
  k2/k6/open 都低      → 不透明符号到类型标记的关联学不会（真实边界）
  unseen 贴近 1/6      → 符合预期：原子 key 不做零样本推广

最后一条是这个实验最重要的产物，它把「包必须是原子整体」的代价量化了：
key 与类型标记之间没有词形联系，没进过训练集的 key 只能瞎猜。
推论：包表必须先声明再训练，加新 key 必须补训练。
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

# 包 key：原子整体，注册后 1 token。整体抽取后比较。
PKG_TYPES = ("NUM", "NAME", "TEXT", "CODE", "ARITH", "TASK")
ID_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
KEY_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|,#[A-Z]>")
TYPE_MARK_RE = re.compile(r"<\|PKG_(?:NUM|NAME|TEXT|CODE|ARITH|TASK)\|>")

# 14 个协议标记占 151646..151659，key 从 151660 起（与 tokens.json 一致）
KEY_ID_BASE = 151660

# 只有留出集才有的轴（负对照）
HOLDOUT_ONLY_AXES = ("type_id_unseen",)

PROTOCOL_TOKENS = [
    "<|PKG_ARITH|>", "<|PKG_CODE|>", "<|PKG_NUM|>", "<|PKG_NAME|>",
    "<|PKG_TEXT|>", "<|PKG_TASK|>",
    "<|ASK|>", "<|ROUTE|>", "<|VERIFY|>", "<|OUTPUT|>",
    "<|DONE|>", "<|FAIL|>", "<|SPLIT|>", "<|MERGE|>",
]


def make_key(pkg_type: str, index: int) -> str:
    """造一个包 key（原子整体）；index 为 0-based，0->A。"""
    return f"<|PKG_{pkg_type}|,#{ID_CHARS[index]}>"


def all_keys() -> list[str]:
    return [make_key(t, i) for t in PKG_TYPES for i in range(len(ID_CHARS))]


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

    # 用 bfloat16 加载：长上下文逐 token 前向在 fp32 下会 OOM
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


def score(pred: str, expected: str, answer_kind: str) -> tuple[bool, bool]:
    """返回 (严格, 部分)。

    "type" → 只认类型标记；"key" → 只认包 key。两者都整体比较。
    """
    strict = pred.strip() == expected.strip()
    pattern = TYPE_MARK_RE if answer_kind == "type" else KEY_RE
    p = set(pattern.findall(pred))
    e = set(pattern.findall(expected))
    partial = bool(p & e)
    return strict, partial


def _level_key(item: tuple[str, dict]) -> tuple[int, str]:
    """排序键：数字 level 按数值排，类型名排后面按字符串排。"""
    level = item[0]
    return (0, f"{int(level):06d}") if level.isdigit() else (1, level)


# ================= 评估 =================


def evaluate(model, tok, rows: list[dict], device: str, max_new: int) -> dict:
    per_cell: dict[tuple[str, str], list[tuple[bool, bool]]] = defaultdict(list)
    samples: list[dict] = []
    for row in rows:
        pred = greedy_decode(model, tok, row["prompt"], max_new, device)
        strict, partial = score(pred, row["target"], row["answer_kind"])
        per_cell[(row["axis"], row["level"])].append((strict, partial))
        samples.append({"axis": row["axis"], "level": row["level"],
                        "kind": row["answer_kind"],
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


def axis_mean(curves: dict, axis: str) -> float | None:
    """一个轴的（跨取值）平均严格分，权重按 n。"""
    levels = curves.get(axis)
    if not levels:
        return None
    total = sum(v["n"] for v in levels.values())
    return round(sum(v["strict"] * v["n"] for v in levels.values()) / total, 4)


def wrong_answers(samples: list[dict], axis: str) -> Counter:
    """统计某轴答错的样本都吐了什么，用来确认是「坍缩」还是「随机错」。"""
    c: Counter = Counter()
    for s in samples:
        if s["axis"] == axis and not s["strict"]:
            pattern = TYPE_MARK_RE if s["kind"] == "type" else KEY_RE
            found = pattern.findall(s["pred"])
            c[found[0] if found else repr(s["pred"][:16])] += 1
    return c


def _unseen_mapping(samples: list[dict]) -> dict[str, set[str]]:
    """负对照里「期望类型 → 实际输出的类型」的映射。

    用来分辨随机错与确定性坍缩：随机错会让同一个期望类型散到多个输出上，
    坍缩则是每种期望类型都稳定吐同一个错类。
    """
    m: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        if s["axis"] != "type_id_unseen":
            continue
        exp = TYPE_MARK_RE.findall(s["expected"])
        got = TYPE_MARK_RE.findall(s["pred"])
        if exp:
            m[exp[0]].add(got[0] if got else "<空>")
    return dict(m)


def _verdict(curves: dict, samples: list[dict]) -> dict:
    """按四档对照给结论。这是本脚本的核心产物。"""
    k2 = axis_mean(curves, "type_id_choice_k2")
    k6 = axis_mean(curves, "type_id_choice_k6")
    op = axis_mean(curves, "type_id_open")
    un = axis_mean(curves, "type_id_unseen")
    pick = axis_mean(curves, "semantic_pick")
    dist_levels = curves.get("distance", {})

    out: dict = {"numbers": {"type_id_choice_k2": k2, "type_id_choice_k6": k6,
                             "type_id_open": op, "type_id_unseen": un,
                             "semantic_pick": pick}, "statements": []}
    st = out["statements"]

    # ---- 类型辨识：四档对照 ----
    if k2 is not None and op is not None:
        if k2 >= 0.9 and op < 0.75:
            st.append(
                f"**候选枚举是必需的**：给 2 个候选 {k2:.2f}、给 6 个候选 "
                f"{k6:.2f}，而不给候选只有 {op:.2f}。同一个模型、同一份配比，"
                f"唯一差别是那一行 [候选]。说明不是模型不懂类型，而是"
                f"「从 156 个不透明 key 里自由生成 1 个类型标记」这件事本身太难。"
                f"→ 协议里 type 判定必须给候选枚举，不能要求开集生成。")
        elif k2 >= 0.9 and (op is None or op >= 0.75):
            st.append(
                f"**类型标记确实被理解，v7 的低分是配比不足**：k2 {k2:.2f}、"
                f"k6 {k6:.2f}、开集 {op:.2f} 都达标。v7 里 type_id 只占 "
                f"192/512 条且答案是开集单 token，样本量不够导致坍缩到 "
                f"<|PKG_TASK|>。加量即可解决，机制本身没问题。")
        elif k2 < 0.6:
            st.append(
                f"**类型标记没学会（真实边界）**：连给 2 个候选都只有 {k2:.2f}，"
                f"6 个候选 {k6:.2f}。这不是候选多少的问题，是 `<|PKG_*|,#X>` "
                f"这个不透明符号与类型标记之间的关联在 65M/4 层容量下学不出来。"
                f"→ 需要改协议：给 key 加可读的类型线索，或换更可分的编码。")

    # ---- 负对照：零样本到底是多少 ----
    # 必须区分两种完全不同的失败，不能都叫「瞎猜」：
    #   接近 1/6        → 真瞎猜（有随机性，能碰运气）
    #   显著低于 1/6    → 系统性错误（贪心解码坍缩到一个固定错类，绝不碰运气）
    # 后者对协议更危险：模型遇到未声明 key 不会表示「不知道」，而是自信地答错。
    if un is not None:
        base = 1 / len(PKG_TYPES)
        mapping = _unseen_mapping(samples)
        deterministic = bool(mapping) and all(len(v) == 1 for v in mapping.values())
        if un >= base * 0.7:
            shape = (f"{un:.2f} 与随机基线 {base:.2f} 相当 → 确实是零信息、"
                     f"随机碰运气")
        else:
            shape = (f"{un:.2f} **显著低于**随机基线 {base:.2f}"
                     + ("，且每种类型都固定映射到同一个错类（确定性坍缩）"
                        if deterministic else "")
                     + " → 不是随机，是系统性错误")
        st.append(
            f"**未训练 key 的类型知识为零**：type_id_unseen {shape}。"
            f"这些 #Z key 一次都没进过训练集，而原子 key 与类型标记之间没有"
            f"词形联系（`<|PKG_NUM|,#Z>` 是单个不可分 token，拼写里没有 NUM），"
            f"所以没有任何可推广的结构。这是「包做成原子整体」的**固有代价**，"
            f"不是 bug。")
        if not un >= base * 0.7:
            st.append(
                f"**这条对协议的真正含义**：未声明的 key 不会让模型「表示不知道」，"
                f"它会稳定输出一个错误的类型标记。所以**「key 是否已注册」必须由"
                f"程序校验，不能指望模型自己发现**；包表必须先声明再训练，"
                f"给医生加新 key 必须补训练。")
        if mapping:
            out["unseen_mapping"] = {k: sorted(v) for k, v in mapping.items()}

    # ---- 语义选取 ----
    if pick is not None:
        st.append(
            f"**语义指代可用**：只给一个 NUM 包、其余全是异类型干扰时，"
            f"跨档平均 {pick:.2f}，靠的是类型前缀定位而非数位置。")

    # ---- 距离 ----
    if dist_levels:
        broken = [lv for lv, v in sorted(dist_levels.items(), key=_level_key)
                  if v["strict"] < 0.9]
        if broken:
            st.append(f"**投影长度上限**：从 {broken[0]} token 档开始跌破 0.90。")
        else:
            longest = max(int(lv) for lv in dist_levels)
            st.append(f"**投影可写到 {longest} token 仍全对**，本实验未探到上限。")

    # ---- 错误形态（避免把坍缩误读成能力） ----
    for axis in ("type_id_open", "type_id_unseen"):
        c = wrong_answers(samples, axis)
        if c:
            top = c.most_common(3)
            out.setdefault("wrong_answer_shapes", {})[axis] = top
            if len(top) == 1:
                st.append(
                    f"注意：{axis} 答错的样本**全部**吐成 {top[0][0]}，"
                    f"这是典型坍缩（押最高频类），不是随机错。"
                    f"随机错会散在多个类型上。")

    return out


# ================= 主流程 =================


def _load_rows(path: Path, limit: int) -> list[dict]:
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    if limit:
        seen: dict[tuple[str, str], int] = defaultdict(int)
        keep = []
        for r in rows:
            k = (r["axis"], r["level"])
            if seen[k] < limit:
                keep.append(r)
                seen[k] += 1
        rows = keep
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="包引用探针 v8 评估（四档类型辨识对照 + 语义选取 + 距离）")
    parser.add_argument("--model", type=Path,
                        default=Path("output_pronoun_probe_v8/final_model"))
    parser.add_argument("--data", type=Path, action="append", default=None,
                        help="可多次指定；默认同时评估 holdout 与 train")
    parser.add_argument("--limit", type=int, default=0, help="每格只取前 N 条")
    parser.add_argument("--max-new", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    model, tok = load_model(args.model, device)

    targets = args.data or [DATA_DIR / "pronoun_probe_v8_holdout.jsonl",
                            DATA_DIR / "pronoun_probe_v8_train.jsonl"]
    report: dict = {"model": str(args.model), "device": device,
                    "note": "判分按 answer_kind：type 只认类型标记，key 只认包 key，"
                            "两者都整体比较（不拆类型/序号）。", "splits": {}}

    for path in targets:
        rows = _load_rows(path, args.limit)
        print(f"\n=== {path.name}：{len(rows)} 条 ===")
        result = evaluate(model, tok, rows, device, args.max_new)
        for axis, levels in result["curves"].items():
            print(f"  {axis}: " + "  ".join(
                f"{lv}={v['strict']:.2f}/{v['partial']:.2f}"
                for lv, v in sorted(levels.items(), key=_level_key)))
        verdict = _verdict(result["curves"], result["samples"])
        for line in verdict["statements"]:
            print("  · " + line)
        # 逐条样本必须**全量**落盘：截断会把排在后面的轴（如 type_id_unseen）
        # 整个截掉，事后就无法检查错误形态，只能看一个聚合分。文件很小，不省这个。
        report["splits"][path.name] = {
            "n": len(rows),
            "curves": result["curves"],
            "verdict": verdict,
            "samples": result["samples"],
        }

    # 结论文本以 holdout 为准（同分布分数会虚高，见笔记）
    hold_key = next((k for k in report["splits"] if "holdout" in k), None)
    if hold_key:
        report["conclusion_from"] = hold_key

    out_path = args.out or DATA_DIR / "pronoun_probe_v8_eval.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"\n报告：{out_path}")


if __name__ == "__main__":
    main()
