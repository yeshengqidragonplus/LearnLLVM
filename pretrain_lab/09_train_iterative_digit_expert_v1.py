#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练单列加法专家，并通过消费数字的循环处理任意位宽。"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def make_domain() -> list[tuple[int, int, int, int, int]]:
    """枚举单列加法的完整有限输入域：(a_digit, b_digit, carry_in, digit, carry_out)。"""
    rows = []
    for a in range(10):
        for b in range(10):
            for carry in range(2):
                total = a + b + carry
                rows.append((a, b, carry, total % 10, total // 10))
    return rows


class StepDataset(Dataset):
    """单列进位规则的监督数据。"""

    def __init__(self, rows: list[tuple[int, int, int, int, int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        a, b, carry, digit, next_carry = self.rows[index]
        return torch.tensor([a, b, carry]), torch.tensor(next_carry * 10 + digit)


class DigitStepExpert(nn.Module):
    """接收两位数字和进位，输出本列写入数字与新进位。"""

    def __init__(self, embedding_size: int = 16, hidden_size: int = 64) -> None:
        super().__init__()
        self.digit_embedding = nn.Embedding(10, embedding_size)
        self.carry_embedding = nn.Embedding(2, 8)
        self.network = nn.Sequential(
            nn.Linear(embedding_size * 2 + 8, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 20),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [self.digit_embedding(state[:, :1]).squeeze(1),
             self.digit_embedding(state[:, 1:2]).squeeze(1),
             self.carry_embedding(state[:, 2])], dim=-1
        )
        return self.network(features)


def train(args) -> None:
    """训练完整的 200 种单列状态，并穷举核验输出。"""
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    rows = make_domain()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "digit_step_domain.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for a, b, carry, digit, next_carry in rows:
            stream.write(json.dumps({"a_digit": a, "b_digit": b, "carry_in": carry,
                                    "write_digit": digit, "carry_out": next_carry}) + "\n")
    loader = DataLoader(StepDataset(rows), batch_size=args.batch, shuffle=True)
    model = DigitStepExpert().to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"单列专家参数量={params:,}，设备={device}，输入状态={len(rows)}，batch={args.batch}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        for states, labels in loader:
            states, labels = states.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(states)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            correct += int(logits.argmax(-1).eq(labels).sum())
            count += len(labels)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:03d}/{args.epochs} loss={loss_sum/count:.6f} accuracy={correct/count:.3f}", flush=True)

    model.eval()
    with torch.inference_mode():
        inputs = torch.tensor([row[:3] for row in rows], dtype=torch.long, device=device)
        expected = torch.tensor([row[4] * 10 + row[3] for row in rows], dtype=torch.long, device=device)
        predicted = model(inputs).argmax(-1)
    exact = int(predicted.eq(expected).sum())
    metrics = {"domain_states": len(rows), "correct": exact, "accuracy": exact / len(rows)}
    print("穷举单列核验：", metrics)
    checkpoint = out / "digit_step_expert.pt"
    torch.save({"model": model.cpu().state_dict(), "embedding_size": 16,
                "hidden_size": 64, "parameters": params, "metrics": metrics,
                "seed": args.seed}, checkpoint)
    print(f"单列专家权重已保存：{checkpoint}")


def load_expert(model_path: str, device_name: str):
    """加载已训练的单列专家。"""
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = DigitStepExpert(checkpoint["embedding_size"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), device


def add_by_iteration(a_text: str, b_text: str, model: DigitStepExpert, device: torch.device) -> tuple[str, list[str]]:
    """每轮消费两个字符串当前最右数字，只将单列状态送入专家。"""
    a_pos, b_pos, carry = len(a_text) - 1, len(b_text) - 1, 0
    output_reversed: list[str] = []
    trace: list[str] = []
    step = 1
    while a_pos >= 0 or b_pos >= 0 or carry:
        da = int(a_text[a_pos]) if a_pos >= 0 else 0
        db = int(b_text[b_pos]) if b_pos >= 0 else 0
        carry_in = carry
        state = torch.tensor([[da, db, carry_in]], dtype=torch.long, device=device)
        with torch.inference_mode():
            prediction = int(model(state).argmax(-1).item())
        next_carry, digit = divmod(prediction, 10)
        # 外部校验器验证学习专家满足单列不变量；不替专家生成数字。
        if 10 * next_carry + digit != da + db + carry_in:
            raise ArithmeticError(
                f"单列专家违反不变量：{da}+{db}+{carry_in} -> {digit}, carry={next_carry}"
            )
        total = da + db + carry_in
        trace.append(
            f"COL {step} a={da} b={db} carry_in={carry_in} sum={total} write={digit} carry_out={next_carry}"
        )
        output_reversed.append(str(digit))
        carry = next_carry
        a_pos -= 1
        b_pos -= 1
        step += 1
    return "".join(reversed(output_reversed)) or "0", trace


def load_router(model_path: str, device_name: str):
    """加载新训练的三分类学习路由器。"""
    router_module = importlib.import_module("10_train_package_router_v2")
    return router_module.load_router(model_path, device_name)


def run_request(text: str, loaded, router_loaded, router_path: str, device_name: str) -> None:
    """由学习路由器选择计算器、迭代竖式专家或 UNKNOWN。"""
    package_module = importlib.import_module("06_package_router_v1")
    model, device = loaded
    try:
        session = package_module.PackageSession(text)
    except ValueError as exc:
        print(f"路由：UNKNOWN\n原因：{exc}")
        return
    kinds = [package.kind for package in session.packages.values()]
    if kinds != ["arithmetic"]:
        print("路由：UNKNOWN\n原因：没有唯一且受支持的加法 package")
        return
    print(f"基座路由器可见：{session.controller_view}")
    router_module = importlib.import_module("10_train_package_router_v2")
    label, confidence = router_module.predict(session.controller_view, router_loaded)
    print(f"学习路由器选择：{label}（置信度 {confidence:.3f}）")
    if label == "UNKNOWN":
        print("最终路由：UNKNOWN")
        return
    payload = session.unpack("p0001")
    match = re.fullmatch(r"\s*(\d+)\s*\+\s*(\d+)\s*", payload)
    if not match:
        print("路由：UNKNOWN\n原因：仅支持两个非负十进制整数相加")
        return
    a_text, b_text = match.groups()
    if label == "CALCULATOR":
        guard = package_module.route_request(session.controller_view, kinds)
        if guard.route != "CALCULATOR":
            print(f"安全路由门：拒绝（{guard.reason}）\n最终路由：UNKNOWN")
            return
        answer = package_module.calculator_expert(payload)
        print("最终路由：CALCULATOR\n专家：整数加法计算器")
        print(f"结果：{answer}")
        return
    answer, trace = add_by_iteration(a_text, b_text, model, device)
    print("最终路由：VERTICAL_MATH_EXPERT（逐列循环；专家权重固定）")
    for line in trace:
        print(line)
    width = max(len(a_text), len(b_text), len(answer))
    print("竖式：")
    print(f"  {a_text:>{width}}")
    print(f"+ {b_text:>{width-1}}")
    print("-" * (width + 2))
    print(f"  {answer:>{width}}")
    expected = str(int(a_text) + int(b_text))
    print(f"校验：{'通过' if answer == expected else '失败'}；迭代次数={len(trace)}；预测结果={answer}；参考结果={expected}")


def main() -> None:
    """训练单列数学专家或交互测试 package→逐位循环闭环。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="训练固定参数的单列加法专家。")
    parser.add_argument("--interactive", action="store_true", help="连续测试；专家只加载一次。")
    parser.add_argument("--text", help="待测试的竖式请求文本。")
    parser.add_argument("--out", default="output_iterative_digit_v1")
    parser.add_argument("--model", default="output_iterative_digit_v1/digit_step_expert.pt")
    parser.add_argument("--router-model", default="output_router_v3_three_way/router.pt",
                        help="三分类学习路由器权重，用于计算器/竖式专家/UNKNOWN 选择。")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train:
        train(args)
        return
    if args.text or args.interactive:
        try:
            loaded = load_expert(args.model, args.device)
            router_loaded = load_router(args.router_model, args.device)
        except (OSError, RuntimeError, KeyError) as exc:
            parser.error(f"无法加载数学专家或普通路由器：{exc}")
        if args.text:
            run_request(args.text, loaded, router_loaded, args.router_model, args.device)
            return
        print("三分类学习路由测试已启动：CALCULATOR / VERTICAL_MATH_EXPERT / UNKNOWN。输入 exit 退出。")
        while True:
            try:
                request = input("请求> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n测试结束。")
                break
            if request.lower() in {"exit", "quit", "退出"}:
                print("测试结束。")
                break
            if request:
                print("--- 路由与专家结果 ---")
                run_request(request, loaded, router_loaded, args.router_model, args.device)
                print()
        return
    parser.error("请提供 --train、--interactive 或 --text。")


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
