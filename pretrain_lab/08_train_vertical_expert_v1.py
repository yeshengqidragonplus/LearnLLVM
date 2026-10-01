#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练独立的逐位竖式加法专家，并接入 arithmetic package。"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE = 256, 257, 258, 259
MAX_LENGTH = 1024
WIDTH, LAYERS, HEADS = 128, 4, 4


def encode(text: str) -> list[int]:
    """将专家输入与答案编码为 UTF-8 字节 ID。"""
    return list(text.encode("utf-8"))


def addition_trace(a: int, b: int) -> str:
    """为非负整数加法生成逐列进位监督答案，列序从个位向左。"""
    left, right = str(a)[::-1], str(b)[::-1]
    carry = 0
    rows = []
    for index in range(max(len(left), len(right))):
        da = int(left[index]) if index < len(left) else 0
        db = int(right[index]) if index < len(right) else 0
        total = da + db + carry
        digit, next_carry = total % 10, total // 10
        rows.append(
            f"COL {index + 1} {da} {db} {carry} {total} {digit} {next_carry}\n"
        )
        carry = next_carry
    if carry:
        rows.append(f"COL {max(len(left), len(right)) + 1} 0 0 {carry} {carry} {carry} 0\n")
    rows.append(f"ANS {a + b}\nEND\n")
    return "".join(rows)


def make_prompt(a: int, b: int) -> str:
    """建立专家可见的展开内容；路由器仍只看到通用 package 标记。"""
    return f"FADD {a}+{b}\nTRACE\n"


def random_number(rng: random.Random, digits: int) -> int:
    """生成指定十进制位数的无前导零整数。"""
    if digits == 1:
        return rng.randrange(0, 10)
    return rng.randrange(10 ** (digits - 1), 10**digits)


def sample_pair(rng: random.Random, max_digits: int) -> tuple[int, int]:
    """生成长度、进位模式都有变化的加数。"""
    digits_a = rng.randint(1, max_digits)
    digits_b = rng.randint(1, max_digits)
    a = random_number(rng, digits_a)
    b = random_number(rng, digits_b)
    # 注入部分高进位链样本，降低训练集里长进位链过少的问题。
    if rng.random() < 0.2:
        a = int("9" * rng.randint(1, max_digits))
        b = rng.randint(1, 9)
    return a, b


def make_data(train_count: int, valid_count: int, ood_per_width: int, seed: int):
    """训练/验证只含 1–4 位数；宽度外测试使用 5–8 位数。"""
    rng = random.Random(seed)
    train_pairs = [sample_pair(rng, 4) for _ in range(train_count)]
    valid_pairs = [sample_pair(rng, 4) for _ in range(valid_count)]
    ood_pairs = []
    for digits in range(5, 9):
        for _ in range(ood_per_width):
            a = random_number(rng, digits)
            b = random_number(rng, digits)
            if rng.random() < 0.25:
                a, b = int("9" * digits), 1
            ood_pairs.append((a, b))
    return train_pairs, valid_pairs, ood_pairs


def save_data(out_dir: Path, pairs_by_name: dict[str, list[tuple[int, int]]]) -> None:
    """保存本轮完整输入、监督输出与可复核标签。"""
    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, pairs in pairs_by_name.items():
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for a, b in pairs:
                row = {"prompt": make_prompt(a, b), "target": addition_trace(a, b), "a": a, "b": b}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(pairs)} 条 -> {data_dir / (name + '.jsonl')}")


def load_rows(path: Path) -> list[dict]:
    """读取并校验准备好的 JSONL。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for index, row in enumerate(rows, 1):
        if set(row) != {"prompt", "target", "a", "b"}:
            raise ValueError(f"{path}:{index} 字段不匹配")
        if not row["prompt"].startswith("FADD ") or not row["target"].endswith("END\n"):
            raise ValueError(f"{path}:{index} prompt/target 格式错误")
    return rows


class TraceDataset(Dataset):
    """prompt+trace 因果语言建模样本，只在 trace 上计算 loss。"""

    def __init__(self, rows: list[dict]) -> None:
        self.samples = []
        for row in rows:
            prefix = encode(row["prompt"])
            target = encode(row["target"])
            ids = [BOS_ID] + prefix + target + [EOS_ID]
            if len(ids) > MAX_LENGTH:
                raise ValueError(f"样本超出最大长度：{row}")
            self.samples.append((ids, 1 + len(prefix)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate(items):
    """变长补齐，并屏蔽 prompt/padding 的语言模型损失。"""
    width = max(len(ids) for ids, _ in items)
    x = torch.full((len(items), width - 1), PAD_ID, dtype=torch.long)
    y = torch.full((len(items), width - 1), -100, dtype=torch.long)
    for row, (ids, prefix_end) in enumerate(items):
        x[row, : len(ids) - 1] = torch.tensor(ids[:-1])
        shifted = ids[1:]
        start = prefix_end - 1
        y[row, start : len(shifted)] = torch.tensor(shifted[start:])
    return x, y, x.eq(PAD_ID)


class VerticalExpert(nn.Module):
    """小型独立因果 Transformer 数学专家。"""

    def __init__(self) -> None:
        super().__init__()
        self.token = nn.Embedding(VOCAB_SIZE, WIDTH, padding_idx=PAD_ID)
        self.position = nn.Embedding(MAX_LENGTH, WIDTH)
        layer = nn.TransformerEncoderLayer(
            d_model=WIDTH, nhead=HEADS, dim_feedforward=512,
            dropout=0.05, batch_first=True, norm_first=True, activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, VOCAB_SIZE, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        causal = torch.triu(torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1)
        hidden = self.token(ids) + self.position(torch.arange(length, device=ids.device))[None]
        hidden = self.layers(hidden, mask=causal, src_key_padding_mask=padding)
        return self.head(self.norm(hidden))


def evaluate_loss(model, loader, device) -> float:
    """计算监督 trace 的 token 交叉熵。"""
    model.eval()
    total_loss = total_tokens = 0
    with torch.inference_mode():
        for x, y, padding in loader:
            logits = model(x.to(device), padding.to(device))
            labels = y.to(device)
            loss_sum = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            total_loss += float(loss_sum)
            total_tokens += int(labels.ne(-100).sum())
    return total_loss / max(total_tokens, 1)


def generate(model, prompt: str, device, max_new_tokens: int = 600) -> str:
    """贪心生成逐位 trace，遇 EOS 停止。"""
    ids = [BOS_ID] + encode(prompt)
    model.eval()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            current = torch.tensor([ids], dtype=torch.long, device=device)
            padding = torch.zeros_like(current, dtype=torch.bool)
            token = int(model(current, padding)[0, -1].argmax())
            if token == EOS_ID:
                break
            if token > 255:
                break
            ids.append(token)
    return bytes(value for value in ids[1 + len(encode(prompt)):] if value < 256).decode("utf-8", errors="replace")


def score_ood(model, rows: list[dict], device, examples_per_width: int, seed: int) -> dict:
    """按 5–8 位分组统计 trace 完全匹配率和答案正确率。"""
    rng = random.Random(seed)
    results = {}
    for width in range(5, 9):
        pool = [row for row in rows if max(len(str(row["a"])), len(str(row["b"]))) == width]
        selected = rng.sample(pool, min(examples_per_width, len(pool)))
        exact = answer_ok = 0
        for row in selected:
            output = generate(model, row["prompt"], device)
            exact += output == row["target"]
            match = re.search(r"^ANS (\d+)$", output, re.M)
            answer_ok += bool(match and int(match.group(1)) == row["a"] + row["b"])
        results[str(width)] = {"n": len(selected), "trace_exact": exact, "answer_correct": answer_ok}
    return results


def train(args) -> None:
    """训练数学专家，逐轮打印 train/valid loss，并做未见位宽评估。"""
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out_dir = Path(args.out)
    train_rows = load_rows(out_dir / "data" / "train.jsonl")
    valid_rows = load_rows(out_dir / "data" / "valid.jsonl")
    ood_rows = load_rows(out_dir / "data" / "test_ood.jsonl")
    train_loader = DataLoader(TraceDataset(train_rows), batch_size=args.batch, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(TraceDataset(valid_rows), batch_size=args.batch, shuffle=False, collate_fn=collate)
    model = VerticalExpert().to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"独立数学专家参数量={params:,}，设备={device}，训练样本={len(train_rows)}，batch={args.batch}")
    print(f"训练样本最大位宽=4；宽度外测试为 5–8 位；每轮更新={len(train_loader)}", flush=True)
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_tokens = 0
        for x, y, padding in train_loader:
            x, y, padding = x.to(device), y.to(device), padding.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, padding)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=-100
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = int(y.ne(-100).sum())
            total_loss += float(loss.detach()) * count
            total_tokens += count
        train_loss = total_loss / max(total_tokens, 1)
        valid_loss = evaluate_loss(model, valid_loader, device)
        print(f"epoch {epoch:02d}/{args.epochs} train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} elapsed={time.perf_counter()-started:.1f}s", flush=True)

    metrics = score_ood(model, ood_rows, device, args.eval_per_width, args.seed + 1)
    print("未见位宽测试（每个宽度抽样）:", json.dumps(metrics, ensure_ascii=False))
    checkpoint = out_dir / "vertical_expert.pt"
    torch.save({"model": model.cpu().state_dict(), "width": WIDTH, "layers": LAYERS,
                "heads": HEADS, "max_length": MAX_LENGTH, "vocab_size": VOCAB_SIZE,
                "parameters": params, "metrics": metrics, "seed": args.seed}, checkpoint)
    print(f"数学专家权重已保存：{checkpoint}")


def load_expert(model_path: str, device_name: str):
    """加载数学专家；交互模式复用此实例。"""
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else device_name
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = VerticalExpert()
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, device


def run_request(text: str, model_path: str, device_name: str, loaded=None) -> None:
    """打包请求；显式竖式意图交给独立专家，其他请求由既有路由基座处理。"""
    package_module = importlib.import_module("06_package_router_v1")
    try:
        session = package_module.PackageSession(text)
    except ValueError as exc:
        print("最终路由：UNKNOWN", f"原因：{exc}", sep="\n")
        return
    kinds = [package.kind for package in session.packages.values()]
    if kinds != ["arithmetic"]:
        print("最终路由：UNKNOWN\n原因：没有唯一的加法 package")
        return
    print(f"基座路由器可见：{session.controller_view}")
    if not package_module.EXPLANATION_INTENT.search(session.controller_view):
        print("本专家只接收明确的竖式/步骤请求；本次不调用数学专家。")
        return

    model, device = loaded if loaded is not None else load_expert(model_path, device_name)
    payload = session.unpack("p0001")
    match = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", payload)
    if not match:
        print("最终路由：UNKNOWN\n原因：专家仅支持非负整数加法")
        return
    a, b = map(int, match.groups())
    result = generate(model, make_prompt(a, b), device)
    print("路由：VERTICAL_MATH_EXPERT（控制器意图规则选择；基座权重冻结）")
    print("数学专家逐位输出：")
    print(result, end="" if result.endswith("\n") else "\n")
    answer = re.search(r"^ANS (\d+)$", result, re.M)
    if not answer or int(answer.group(1)) != a + b:
        print("校验：失败；输出不作为正确答案")
        return
    total = answer.group(1)
    width = max(len(str(a)), len(str(b)), len(total))
    print("竖式：")
    print(f"  {a:>{width}}")
    print(f"+ {b:>{width-1}}")
    print("-" * (width + 2))
    print(f"  {total:>{width}}")
    print("校验：专家结果与整数校验一致")


def main() -> None:
    """准备训练数据、训练数学专家或运行 package→逐位专家闭环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="生成训练/验证/5–8 位外推数据，不训练。")
    parser.add_argument("--train", action="store_true", help="训练独立逐位数学专家。")
    parser.add_argument("--text", help="以文本请求测试 package→竖式专家闭环。")
    parser.add_argument("--interactive", action="store_true", help="启动连续测试；专家只加载一次。")
    parser.add_argument("--model", default="output_vertical_add_v1/vertical_expert.pt")
    parser.add_argument("--out", default="output_vertical_add_v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--valid-samples", type=int, default=512)
    parser.add_argument("--ood-per-width", type=int, default=64)
    parser.add_argument("--eval-per-width", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.prepare_data:
        pairs = make_data(args.samples, args.valid_samples, args.ood_per_width, args.seed)
        save_data(Path(args.out), dict(zip(("train", "valid", "test_ood"), pairs)))
        print("本次只准备数据，没有启动训练。")
        return
    if args.train:
        train(args)
        return
    if args.text:
        run_request(args.text, args.model, args.device)
        return
    if args.interactive:
        try:
            loaded = load_expert(args.model, args.device)
        except (OSError, RuntimeError, KeyError) as exc:
            parser.error(f"无法加载数学专家 {args.model}: {exc}")
        print("逐位数学专家交互测试已启动。输入 exit/quit/退出 结束。")
        print(f"权重：{args.model}；设备：{loaded[1]}")
        print("触发词：竖式、步骤、过程、推导、解释、怎么计算、如何计算")
        while True:
            try:
                text = input("请求> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n交互测试结束。")
                break
            if text.lower() in {"exit", "quit", "退出"}:
                print("交互测试结束。")
                break
            if not text:
                continue
            print("--- 专家结果 ---")
            run_request(text, args.model, args.device, loaded)
            print()
        return
    parser.error("请提供 --prepare-data、--train、--text 或 --interactive。")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
