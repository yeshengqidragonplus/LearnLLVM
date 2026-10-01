#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩展现有 package 路由器为计算/竖式/UNKNOWN 三类并训练专家选择能力。"""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_LABEL_NAMES = ("CALCULATOR", "VERTICAL_MATH_EXPERT", "UNKNOWN")
CALCULATOR, VERTICAL, UNKNOWN = range(3)
PACKAGE_MARKER = "<|PKG_ARITH|>"

TRAIN_TEMPLATES = {
    CALCULATOR: [
        "请计算 {pkg}", "帮我计算一下 {pkg}", "请直接给出 {pkg} 的结果",
        "{pkg}，算出答案", "求和：{pkg}", "帮我算出 {pkg} 的结果",
        "calculate {pkg}", "compute the result of {pkg}", "计算{pkg}", "{pkg}",
        "帮我求一下 {pkg} 的数值", "给出算式 {pkg} 的结果",
        "{pkg}，请算出答案", "直接算 {pkg}", "我只要答案 {pkg}",
        "{pkg} 求结果", "请给出和：{pkg}", "solve {pkg}",
    ],
    VERTICAL: [
        "请对 {pkg} 做竖式计算", "展示 {pkg} 的逐位进位过程",
        "请详细拆解计算步骤：{pkg}", "解释如何得出结果：{pkg}",
        "请把 {pkg} 按位展开并写出每步进位", "请用竖式分列计算 {pkg}",
        "show {pkg} as column-wise addition", "add {pkg} digit by digit",
        "show the carry at each place for {pkg}", "请逐列列出 {pkg} 的计算过程",
        "把 {pkg} 从个位开始展开计算", "请用竖式展示 {pkg} 的每一列",
        "{pkg} 竖式计算", "{pkg} 做逐位步骤", "{pkg} 按位展开算",
        "请从右向左逐位计算 {pkg}", "列竖式，一步一步算 {pkg}",
        "先看个位进位，再继续计算 {pkg}", "给我看每一列怎么进位：{pkg}",
        "用竖式把 {pkg} 展开", "逐位演算这个算式 {pkg}",
    ],
    UNKNOWN: [
        "不要计算 {pkg}，只检查格式", "只复述 {pkg}，不要给答案",
        "{pkg} 是什么类型的运算", "请辨认 {pkg} 但不要求解",
        "do not calculate {pkg}; just identify the operation",
        "repeat {pkg} without solving", "保留 {pkg} 原样，不要计算",
        "检查 {pkg} 的写法，不要给结果", "只告诉我 {pkg} 有几个符号",
        "请把 {pkg} 原样抄写", "不要执行 {pkg}，只分析它的格式",
        "identify the expression type in {pkg}, do not solve it",
        "{pkg} 只复述，不要算", "{pkg} 是什么符号组合，只描述",
        "不要给 {pkg} 的答案，只说有几个数字", "请原样保留 {pkg}",
    ],
}

HELDOUT_TEMPLATES = {
    CALCULATOR: [
        "麻烦求一下 {pkg}", "请给我算出答案：{pkg}", "求 {pkg} 的值",
        "what is the answer for {pkg}?", "帮我把 {pkg} 求出来",
        "计算一下 {pkg} 的值", "告诉我 {pkg} 等于多少", "evaluate {pkg}",
        "how much is {pkg}?", "只要最终结果：{pkg}",
        "请算出 {pkg} 的和", "给我这个算式的数值 {pkg}",
        "{pkg} 给出答案", "帮我算一下这个：{pkg}",
        "直接求 {pkg} 的和", "只需要计算结果 {pkg}",
    ],
    VERTICAL: [
        "请把 {pkg} 写成竖式并算完", "从个位开始逐列演算 {pkg}",
        "列出 {pkg} 每位的和与进位", "按竖列展示算式 {pkg}",
        "show the vertical addition for {pkg}", "work through the carry for {pkg}",
        "把 {pkg} 拆成一列一列来算", "请逐位写出 {pkg} 的中间计算",
        "我想看 {pkg} 的竖式过程", "每一位如何相加？算式是 {pkg}",
        "逐列计算 {pkg} 并说明进位", "write out each column for {pkg}",
        "将 {pkg} 从右边开始逐位展开", "演示 {pkg} 的每次进位",
        "请列出加法 {pkg} 的各列计算", "从个位到最高位计算 {pkg}",
        "我想看加数 {pkg} 的竖式写法", "逐位展示 {pkg} 怎么进位",
        "按列拆解这个加法 {pkg}", "逐步写出每列结果：{pkg}",
    ],
    UNKNOWN: [
        "只检查表达式格式：{pkg}", "把 {pkg} 原样念出来",
        "我只想知道 {pkg} 的结构，不要算答案", "repeat only {pkg}",
        "repeat the expression {pkg} verbatim", "请比较这段文本和另一段文本：{pkg}",
        "不要给 {pkg} 的结果，只说它是不是表达式", "保留原文 {pkg}，不做运算",
        "这段输入包含什么字符：{pkg}", "标注 {pkg} 的操作符，但不要求值",
        "请原样输出 {pkg}，不要回答结果", "classify but do not solve {pkg}",
        "{pkg} 只判断格式，不用计算", "只告诉我里面有什么符号 {pkg}",
        "抄录 {pkg}，不要进行运算", "请描述 {pkg} 的文字形式",
    ],
}
DECORATIONS = ["", "，谢谢", "。", "，不要近似", "，只处理这个请求"]


def get_codec():
    """复用 v1 路由器的 UTF-8 字节词表和 package 特殊 ID。"""
    return importlib.import_module("07_train_package_router_v1")


def make_examples(count_per_class: int, seed: int) -> list[tuple[str, int]]:
    """生成只包含意图和 package 标记的三类训练输入。"""
    rng = random.Random(seed)
    rows = []
    for label, templates in TRAIN_TEMPLATES.items():
        for _ in range(count_per_class):
            text = rng.choice(templates).format(pkg=PACKAGE_MARKER)
            if text != PACKAGE_MARKER:
                text += rng.choice(DECORATIONS)
            rows.append((text, label))
    rng.shuffle(rows)
    return rows


def make_heldout() -> list[tuple[str, int]]:
    """用与训练模板不重叠的措辞测试学习路由。"""
    return [(template.format(pkg=PACKAGE_MARKER), label)
            for label, templates in HELDOUT_TEMPLATES.items() for template in templates]


def write_data(out_dir: Path, train_rows: list[tuple[str, int]], heldout: list[tuple[str, int]]) -> None:
    """保存可审核的 UTF-8 JSONL，不含数字 payload。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train_rows), ("heldout.jsonl", heldout)):
        with (out_dir / name).open("w", encoding="utf-8", newline="\n") as stream:
            for text, label in rows:
                stream.write(json.dumps({"text": text, "label": BASE_LABEL_NAMES[label]}, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} 条 -> {out_dir / name}")


def read_data(out_dir: Path, name: str) -> list[tuple[str, int]]:
    """读取路由样本并校验类别和 package 标记。"""
    rows = []
    with (out_dir / name).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            record = json.loads(line)
            if set(record) != {"text", "label"} or record["label"] not in BASE_LABEL_NAMES:
                raise ValueError(f"{name}:{line_no} 格式或类别错误")
            if PACKAGE_MARKER not in record["text"]:
                raise ValueError(f"{name}:{line_no} 缺少 package 标记")
            rows.append((record["text"], BASE_LABEL_NAMES.index(record["label"])))
    return rows


class RouterDataset(Dataset):
    """学习 package 路由意图的样本集合。"""

    def __init__(self, rows: list[tuple[str, int]]) -> None:
        codec = get_codec()
        self.rows = [(codec.encode_router_text(text), label) for text, label in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(items):
    """变长 UTF-8 字节序列批处理。"""
    codec = get_codec()
    width = max(len(ids) for ids, _ in items)
    ids = torch.full((len(items), width), codec.PAD_ID, dtype=torch.long)
    labels = torch.empty(len(items), dtype=torch.long)
    for row, (tokens, label) in enumerate(items):
        ids[row, :len(tokens)] = torch.tensor(tokens)
        labels[row] = label
    return ids, ids.eq(codec.PAD_ID), labels


class PackageRouterThreeWay(nn.Module):
    """与 v1 路由基座同结构、改为 CALCULATOR/VERTICAL/UNKNOWN 三分类。"""

    def __init__(self, width: int = 128, layers: int = 4, heads: int = 4) -> None:
        super().__init__()
        codec = get_codec()
        self.token_embedding = nn.Embedding(codec.VOCAB_SIZE, width, padding_idx=codec.PAD_ID)
        self.position_embedding = nn.Embedding(codec.MAX_LENGTH, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=512, dropout=0.1,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.classifier = nn.Linear(width, 3)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.classifier(self.norm(hidden[:, 0]))


def initialize_from_checkpoint(model: PackageRouterThreeWay, checkpoint: dict) -> None:
    """从既有二分类或三分类路由器迁移初始化。"""
    old = checkpoint["model"]
    if old["classifier.weight"].shape[0] == 3:
        model.load_state_dict(old)
        return
    current = model.state_dict()
    for name, value in old.items():
        if name.startswith("classifier."):
            continue
        current[name].copy_(value)
    current["classifier.weight"][CALCULATOR].copy_(old["classifier.weight"][0])
    current["classifier.weight"][UNKNOWN].copy_(old["classifier.weight"][1])
    current["classifier.bias"][CALCULATOR].copy_(old["classifier.bias"][0])
    current["classifier.bias"][UNKNOWN].copy_(old["classifier.bias"][1])
    model.load_state_dict(current)


def evaluate(model, rows, device) -> dict:
    """返回整体、分类准确率与混淆矩阵。"""
    loader = DataLoader(RouterDataset(rows), batch_size=64, collate_fn=collate)
    correct = [0, 0, 0]
    totals = [0, 0, 0]
    confusion = [[0] * 3 for _ in range(3)]
    model.eval()
    with torch.inference_mode():
        for ids, mask, labels in loader:
            pred = model(ids.to(device), mask.to(device)).argmax(-1).cpu()
            for y, p in zip(labels.tolist(), pred.tolist()):
                totals[y] += 1
                correct[y] += int(y == p)
                confusion[y][p] += 1
    return {"accuracy": sum(correct) / max(sum(totals), 1),
            "per_class": {BASE_LABEL_NAMES[i]: correct[i] / max(totals[i], 1) for i in range(3)},
            "counts": {BASE_LABEL_NAMES[i]: totals[i] for i in range(3)},
            "confusion_rows_true_cols_pred": confusion}


def load_router(model_path: str, device_name: str):
    """加载三分类路由器用于 package 请求选择。"""
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = PackageRouterThreeWay(width=checkpoint["width"], layers=checkpoint["layers"], heads=checkpoint["heads"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), device


def predict(text: str, loaded):
    """根据 package controller view 输出三类路由标签和置信度。"""
    model, device = loaded
    codec = get_codec()
    ids = torch.tensor([codec.encode_router_text(text)], dtype=torch.long, device=device)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    with torch.inference_mode():
        probabilities = model(ids, mask).softmax(-1)[0]
    label = int(probabilities.argmax())
    return BASE_LABEL_NAMES[label], float(probabilities[label])


def train(args) -> None:
    """从既有二分类路由器迁移初始化，学习第三类数学专家路由。"""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    out = Path(args.out)
    train_rows = read_data(out / "data", "train.jsonl")
    heldout_rows = read_data(out / "data", "heldout.jsonl")
    codec = get_codec()
    base = torch.load(args.init_from, map_location="cpu", weights_only=True)
    model = PackageRouterThreeWay(width=base["width"], layers=base["layers"], heads=base["heads"])
    initialize_from_checkpoint(model, base)
    model.to(device)
    train_loader = DataLoader(RouterDataset(train_rows), batch_size=args.batch, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"从二分类路由器迁移初始化；参数量={params:,}；设备={device}")
    print(f"训练={len(train_rows)}，留出={len(heldout_rows)}，每轮更新={len(train_loader)}，总更新={len(train_loader)*args.epochs}", flush=True)
    best_accuracy = -1.0
    best_state = None
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach())
        metrics = evaluate(model, heldout_rows, device)
        print(f"epoch {epoch:02d}/{args.epochs} loss={loss_sum/len(train_loader):.4f} "
              f"heldout={metrics['accuracy']:.3f} per_class={metrics['per_class']} "
              f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    metrics = evaluate(model, heldout_rows, device)
    print("最佳留出集指标:", json.dumps(metrics, ensure_ascii=False))
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.cpu().state_dict(), "width": base["width"],
                "layers": base["layers"], "heads": base["heads"],
                "max_length": codec.MAX_LENGTH, "vocab_size": codec.VOCAB_SIZE,
                "package_token_id": codec.PACKAGE_ID, "labels": BASE_LABEL_NAMES,
                "parameters": params, "metrics": metrics, "seed": args.seed,
                "initialized_from": args.init_from}, out / "router.pt")
    print(f"三分类路由权重已保存：{out / 'router.pt'}")


def main() -> None:
    """准备三类数据、训练路由器或审阅类别定义。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true", help="生成训练/留出数据，不训练。")
    parser.add_argument("--train", action="store_true", help="训练三分类专家选择路由。")
    parser.add_argument("--out", default="output_router_v3_three_way")
    parser.add_argument("--init-from", default="output_router_v2_three_way/router.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--samples-per-class", type=int, default=2400)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.prepare_data:
        rows = make_examples(args.samples_per_class, args.seed)
        heldout = make_heldout()
        write_data(Path(args.out) / "data", rows, heldout)
        print("类别计数:", {BASE_LABEL_NAMES[i]: sum(y == i for _, y in rows) for i in range(3)})
        print("本次只生成数据，没有启动训练。")
        return
    if args.train:
        train(args)
        return
    parser.error("请提供 --prepare-data 或 --train。")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
