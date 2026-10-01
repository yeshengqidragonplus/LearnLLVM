#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练约 1M 参数的小型 package 路由器，并用它调用整数加法专家。"""

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


# UTF-8 字节可表示任意语言字符；PACKAGE_ID 是唯一的专用结构 token。
PACKAGE_MARKER = "<|PKG_ARITH|>"
PACKAGE_ID = 256
BOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
MAX_LENGTH = 384
CALCULATOR = 0
UNKNOWN = 1
LABEL_NAMES = ("CALCULATOR", "UNKNOWN")

TRAIN_TEMPLATES = {
    CALCULATOR: [
        "请计算 {pkg}",
        "帮我计算一下 {pkg}",
        "请直接给出 {pkg} 的结果",
        "{pkg}，算出答案",
        "求和：{pkg}",
        "帮我算出 {pkg} 的结果",
        "calculate {pkg}",
        "compute the result of {pkg}",
        "计算{pkg}",
        "计算 {pkg}",
        "{pkg}",
    ],
    UNKNOWN: [
        "请对 {pkg} 做竖式计算",
        "展示 {pkg} 的逐位进位过程",
        "请详细拆解计算步骤：{pkg}",
        "解释如何得出结果：{pkg}",
        "explain {pkg} step by step",
        "show your work for {pkg}",
        "不要计算 {pkg}，只检查格式",
        "do not calculate {pkg}; just identify the operation",
        "这个算式是什么含义：{pkg}",
        "请计算 8*9",
        "今天是 2026-09-30",
    ],
}
TEST_TEMPLATES = {
    CALCULATOR: [
        "麻烦求一下 {pkg}",
        "请给我算出答案：{pkg}",
        "求 {pkg} 的值",
        "what is the answer for {pkg}?",
        "计算 {pkg}",
    ],
    UNKNOWN: [
        "请把 {pkg} 的运算过程展开",
        "不要只给数字，说明 {pkg} 如何计算",
        "详细展示竖式：{pkg}",
        "compute {pkg} and explain every step",
        "不要执行 {pkg}，只复述输入",
        "do not calculate {pkg}, only repeat it",
        "查询今天的日期",
        "请计算 123*456",
    ],
}
DECORATIONS = ["", "，谢谢", "。", "，不要近似", "，只处理这个请求"]
EXPRESSION_MARKERS = [
    "<|PKG_ARITH|>",
]


def normalize_controller_whitespace(text: str) -> str:
    """将路由视图中的连续水平空白折叠为一个空格。"""
    return re.sub(r"[^\S\r\n]+", " ", text).strip()


def encode_router_text(text: str) -> list[int]:
    """UTF-8 字节编码普通文本，并将 package 标记编码成一个专用 ID。"""
    text = normalize_controller_whitespace(text)
    ids = [BOS_ID]
    parts = text.split(PACKAGE_MARKER)
    for index, part in enumerate(parts):
        ids.extend(part.encode("utf-8"))
        if index < len(parts) - 1:
            ids.append(PACKAGE_ID)
    if len(ids) > MAX_LENGTH:
        raise ValueError(f"路由输入超过 v1 上限 {MAX_LENGTH - 1} 个字节 token")
    return ids


def make_examples(count_per_class: int, seed: int) -> list[tuple[str, int]]:
    """生成短路由任务样本；算式值不会进入路由器输入。"""
    rng = random.Random(seed)
    examples: list[tuple[str, int]] = []
    for label, templates in TRAIN_TEMPLATES.items():
        for _ in range(count_per_class):
            template = rng.choice(templates)
            marker = rng.choice(EXPRESSION_MARKERS)
            text = template.format(pkg=marker)
            if template != "{pkg}":
                text += rng.choice(DECORATIONS)
            text = normalize_controller_whitespace(text)
            examples.append((text, label))
    rng.shuffle(examples)
    return examples


def make_heldout_examples() -> list[tuple[str, int]]:
    """使用训练中未出现的请求措辞评估路由泛化。"""
    examples = []
    for label, templates in TEST_TEMPLATES.items():
        for template in templates:
            examples.append((normalize_controller_whitespace(template.format(pkg=PACKAGE_MARKER)), label))
    return examples


def save_examples(data_dir: str, examples: list[tuple[str, int]], heldout: list[tuple[str, int]]) -> None:
    """将固定训练集与留出集写成可人工检查的 UTF-8 JSONL。"""
    output = Path(data_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", examples), ("heldout.jsonl", heldout)):
        with (output / name).open("w", encoding="utf-8", newline="\n") as stream:
            for text, label in rows:
                row = {"text": text, "label": LABEL_NAMES[label]}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"已写入 {output / name}：{len(rows)} 条")


def load_examples(data_dir: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """读取已准备好的训练集和留出集。"""
    loaded = []
    for name in ("train.jsonl", "heldout.jsonl"):
        rows = []
        with (Path(data_dir) / name).open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                if set(row) != {"text", "label"} or row["label"] not in LABEL_NAMES:
                    raise ValueError(f"{name}:{line_no} 格式或 label 无效")
                rows.append((row["text"], LABEL_NAMES.index(row["label"])))
        loaded.append(rows)
    return loaded[0], loaded[1]


class RouterDataset(Dataset):
    """路由器样本集合。"""

    def __init__(self, examples: list[tuple[str, int]]) -> None:
        self.examples = [(encode_router_text(text), label) for text, label in examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_batch(items: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """补齐变长请求，并返回 padding mask。"""
    width = max(len(ids) for ids, _ in items)
    input_ids = torch.full((len(items), width), PAD_ID, dtype=torch.long)
    labels = torch.empty(len(items), dtype=torch.long)
    for row, (ids, label) in enumerate(items):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[row] = label
    padding_mask = input_ids.eq(PAD_ID)
    return input_ids, padding_mask, labels


class PackageRouter(nn.Module):
    """小型双向 Transformer 分类路由器。"""

    def __init__(self, width: int = 128, layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, width, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(MAX_LENGTH, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 2)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.classifier(self.norm(hidden[:, 0]))


def evaluate(model: PackageRouter, examples: list[tuple[str, int]], device: torch.device) -> dict[str, float]:
    """返回准确率与每类准确率。"""
    loader = DataLoader(RouterDataset(examples), batch_size=64, collate_fn=collate_batch)
    correct = total = 0
    class_correct = [0, 0]
    class_total = [0, 0]
    model.eval()
    with torch.inference_mode():
        for input_ids, padding_mask, labels in loader:
            predictions = model(input_ids.to(device), padding_mask.to(device)).argmax(-1).cpu()
            correct += int(predictions.eq(labels).sum())
            total += len(labels)
            for label in (CALCULATOR, UNKNOWN):
                selected = labels.eq(label)
                class_total[label] += int(selected.sum())
                class_correct[label] += int(predictions[selected].eq(labels[selected]).sum())
    return {
        "accuracy": correct / max(total, 1),
        "calculator_accuracy": class_correct[CALCULATOR] / max(class_total[CALCULATOR], 1),
        "unknown_accuracy": class_correct[UNKNOWN] / max(class_total[UNKNOWN], 1),
        "examples": float(total),
    }


def train(args: argparse.Namespace) -> None:
    """训练 1M 量级路由器并保存验证指标。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    examples, heldout = (
        load_examples(args.data_dir) if args.data_dir else
        (make_examples(args.samples_per_class, args.seed), make_heldout_examples())
    )
    loader = DataLoader(
        RouterDataset(examples), batch_size=args.batch, shuffle=True,
        collate_fn=collate_batch,
    )
    model = PackageRouter().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    print(f"路由器参数量: {parameter_count:,}，设备: {device}，训练样本: {len(examples)}")
    print(f"训练配置: batch={args.batch}，每轮更新={len(loader)}，计划总更新={len(loader) * args.epochs}", flush=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for input_ids, padding_mask, labels in loader:
            input_ids = input_ids.to(device)
            padding_mask = padding_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(input_ids, padding_mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
        heldout_metrics = evaluate(model, heldout, device)
        elapsed = time.perf_counter() - started
        memory = f"，峰值显存={torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB" if device.type == "cuda" else ""
        print(
            f"epoch {epoch:02d}/{args.epochs} loss={total_loss / len(loader):.4f} "
            f"heldout_acc={heldout_metrics['accuracy']:.3f} "
            f"(CALC={heldout_metrics['calculator_accuracy']:.3f}, UNKNOWN={heldout_metrics['unknown_accuracy']:.3f}) "
            f"elapsed={elapsed:.1f}s{memory}", flush=True,
        )

    metrics = evaluate(model, heldout, device)
    print("未见措辞测试:", metrics)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.cpu().state_dict(),
            "width": 128,
            "layers": 4,
            "heads": 4,
            "max_length": MAX_LENGTH,
            "vocab_size": VOCAB_SIZE,
            "package_token_id": PACKAGE_ID,
            "labels": LABEL_NAMES,
            "parameters": parameter_count,
            "metrics": metrics,
            "seed": args.seed,
        },
        output / "router.pt",
    )
    print(f"模型已保存: {output / 'router.pt'}")


def load_router(model_path: str, device_name: str) -> tuple[PackageRouter, torch.device]:
    """加载已训练路由器；交互模式复用同一实例。"""
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto" else device_name
    )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = PackageRouter(
        width=checkpoint["width"],
        layers=checkpoint["layers"],
        heads=checkpoint["heads"],
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, device


def run_request(
    text: str,
    model_path: str,
    device_name: str,
    loaded: tuple[PackageRouter, torch.device] | None = None,
) -> None:
    """将 package 交给学习路由器，再按路由结果调用专家。"""
    package_module = importlib.import_module("06_package_router_v1")
    try:
        session = package_module.PackageSession(text)
    except ValueError as exc:
        print("路由：UNKNOWN")
        print(f"原因：{exc}")
        return

    package_kinds = [package.kind for package in session.packages.values()]
    if package_kinds != ["arithmetic"]:
        print("路由：UNKNOWN")
        print("原因：没有唯一且受支持的算式 package")
        return
    model, device = loaded if loaded is not None else load_router(model_path, device_name)
    try:
        ids = encode_router_text(session.controller_view)
    except ValueError as exc:
        print("路由：UNKNOWN")
        print(f"原因：{exc}")
        return
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    with torch.inference_mode():
        probabilities = model(input_ids, padding_mask).softmax(-1)[0]
    label = int(probabilities.argmax())
    print(f"路由器可见输入：{session.controller_view}")
    print(f"package 类型：{package_kinds[0]}；payload 与句柄未输入路由器")
    print(f"学习路由器预测：{LABEL_NAMES[label]}（置信度 {float(probabilities[label]):.3f}）")
    if label != CALCULATOR:
        print("最终路由：UNKNOWN")
        print("结果：UNKNOWN")
        return
    guard = package_module.route_request(session.controller_view, package_kinds)
    if guard.route != "CALCULATOR":
        print(f"安全路由门：拒绝（{guard.reason}）")
        print("最终路由：UNKNOWN")
        print("结果：UNKNOWN")
        return
    print("安全路由门：通过")
    print("最终路由：CALCULATOR")
    try:
        value = package_module.calculator_expert(session.unpack("p0001"))
    except (KeyError, ValueError) as exc:
        print("路由：UNKNOWN")
        print(f"原因：计算专家拒绝输入：{exc}")
        return
    print("专家：整数加法计算器")
    print(f"结果：{value}")


def main() -> None:
    """训练路由器或运行 package→路由→计算器链路。"""
    parser = argparse.ArgumentParser(
        description="训练小型 package 路由器，或调用计算器专家；未路由请求返回 UNKNOWN。"
    )
    parser.add_argument("--train", action="store_true", help="训练小型路由模型。")
    parser.add_argument("--prepare-data", action="store_true", help="只生成训练集和留出集，不训练。")
    parser.add_argument("--data-dir", help="读取已准备好的 train.jsonl 与 heldout.jsonl。")
    parser.add_argument("--text", help="待路由的用户请求。")
    parser.add_argument("--interactive", action="store_true", help="启动交互测试，每轮复用已加载的路由模型。")
    parser.add_argument("--model", default="output_router_v1/router.pt", help="路由模型文件。")
    parser.add_argument("--out", default="output_router_v1", help="训练结果目录。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--samples-per-class", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train:
        train(args)
        return
    if args.prepare_data:
        examples = make_examples(args.samples_per_class, args.seed)
        heldout = make_heldout_examples()
        data_dir = Path(args.out) / "data"
        save_examples(str(data_dir), examples, heldout)
        print(f"类别计数：{sum(y == CALCULATOR for _, y in examples)} CALCULATOR；{sum(y == UNKNOWN for _, y in examples)} UNKNOWN")
        print("本次只生成数据，没有启动训练。")
        return
    if args.text:
        run_request(args.text, args.model, args.device)
        return
    if args.interactive:
        try:
            loaded = load_router(args.model, args.device)
        except (OSError, RuntimeError, KeyError) as exc:
            parser.error(f"无法加载路由模型 {args.model}: {exc}")
        print("交互测试已启动。输入请求进行测试；输入 exit 或 quit 退出。")
        print(f"路由模型：{args.model}；设备：{loaded[1]}")
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
            print("--- 路由结果 ---")
            run_request(text, args.model, args.device, loaded)
            print()
        return
    parser.error("请提供 --train、--text 或 --interactive。")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
