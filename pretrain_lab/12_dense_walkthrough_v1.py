#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐步打印一个教学用 Dense Decoder 的 token、RoPE、QKV、多头注意力与 logits。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer

from runtime import environment_info, resolve_runtime


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    """对形状 [batch, heads, time, head_dim] 的 Q/K 应用 RoPE。"""
    batch, heads, time, width = x.shape
    if width % 2:
        raise ValueError("RoPE 要求每头维度是偶数。")
    pair = x.reshape(batch, heads, time, width // 2, 2)
    position = torch.arange(time, device=x.device, dtype=torch.float32)
    frequency = torch.arange(0, width, 2, device=x.device, dtype=torch.float32)
    inverse = 1.0 / (10000.0 ** (frequency / width))
    angle = position[:, None] * inverse[None, :]
    cosine = angle.cos().to(dtype=x.dtype)[None, None, :, :]
    sine = angle.sin().to(dtype=x.dtype)[None, None, :, :]
    even, odd = pair[..., 0], pair[..., 1]
    rotated = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    )
    return rotated.flatten(-2)


class OneDecoderBlock(nn.Module):
    """暴露中间张量的单层 Pre-Norm Decoder block。"""

    def __init__(self, width: int, heads: int):
        super().__init__()
        if width % heads:
            raise ValueError("hidden size 必须能被注意力头数整除。")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, time, _ = x.shape
        normalized = self.norm1(x)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.view(batch, time, self.heads, self.head_dim).transpose(1, 2)

        q_raw = split_heads(self.q_proj(normalized))
        k_raw = split_heads(self.k_proj(normalized))
        v = split_heads(self.v_proj(normalized))
        q, k = apply_rope(q_raw), apply_rope(k_raw)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.full((time, time), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )
        weights = torch.softmax(scores + mask, dim=-1)
        attended = weights @ v
        merged = attended.transpose(1, 2).contiguous().view(batch, time, self.width)
        attention_output = self.o_proj(merged)
        after_attention = x + attention_output
        mlp_output = self.mlp(self.norm2(after_attention))
        result = after_attention + mlp_output
        trace = {
            "normalized": normalized,
            "q_raw": q_raw,
            "k_raw": k_raw,
            "q": q,
            "k": k,
            "v": v,
            "scores": scores,
            "mask": mask,
            "weights": weights,
            "head_output": attended,
            "merged": merged,
            "attention_output": attention_output,
            "mlp_output": mlp_output,
            "block_output": result,
        }
        return result, trace


class TinyDenseDecoder(nn.Module):
    """权重随机初始化的教学模型，不用于语言能力结论。"""

    def __init__(self, vocab_size: int, width: int, heads: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.block = OneDecoderBlock(width, heads)
        self.final_norm = nn.LayerNorm(width)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        embedded = self.token_embedding(ids)
        hidden, trace = self.block(embedded)
        hidden = self.final_norm(hidden)
        logits = F.linear(hidden, self.token_embedding.weight)
        trace["final_hidden"] = hidden
        return embedded, logits, trace


def shape(tensor: torch.Tensor) -> str:
    return str(tuple(tensor.shape))


def run_training_demo(args, model, tokenizer, token_ids, vocab_size, device, dtype):
    """用单条短句演示 next-token 标签错位、梯度更新与过拟合。"""
    if tokenizer.eos_token_id is None:
        raise RuntimeError("训练演示要求 tokenizer 提供 EOS token。")
    out_dir = args.out or Path(
        "output_dense_tiny_train_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖：{out_dir}")
    out_dir.mkdir(parents=True)

    full_ids = torch.tensor([token_ids + [tokenizer.eos_token_id]], dtype=torch.long, device=device)
    inputs, targets = full_ids[:, :-1], full_ids[:, 1:]
    if inputs.shape[1] == 0:
        raise ValueError("训练演示需要至少一个输入 token。")
    predictions_desc = "每个位置预测右移一格的 token，最后预测 EOS"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    metrics = []
    start = datetime.now(timezone.utc)
    clock_start = time.perf_counter()

    def loss_value():
        with torch.no_grad():
            _, out_logits, _ = model(inputs)
            return float(F.cross_entropy(out_logits.reshape(-1, vocab_size), targets.reshape(-1)))

    initial_loss = loss_value()
    log_path = out_dir / "train.log"
    run = {
        "status": "started", "started_at": start.isoformat(),
        "text": args.text, "token_ids": token_ids,
        "sequence_ids_with_eos": full_ids[0].tolist(), "input_ids": inputs[0].tolist(),
        "target_ids": targets[0].tolist(), "target_alignment": predictions_desc,
        "model": {"type": "one-layer teaching decoder", "hidden": args.hidden,
                  "heads": args.heads, "parameters": sum(p.numel() for p in model.parameters())},
        "training": {"steps": args.steps, "learning_rate": args.lr,
                     "optimizer": "AdamW", "single_repeated_example": True},
        "environment": environment_info(device, dtype),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_loss": initial_loss, "metrics": metrics,
        "scope": "单样本记忆演示，不是语言能力或泛化评估",
    }
    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"text={args.text!r}\ninput_ids={inputs[0].tolist()}\ntarget_ids={targets[0].tolist()}\n")
        log.write(f"initial_loss={initial_loss:.6f}\n")
        print("\n=== 单样本 next-token 训练演示 ===")
        print("输入 IDs：", inputs[0].tolist(), "；目标 IDs：", targets[0].tolist())
        print("目标说明：每个位置预测右移一格的 token，最后预测 EOS")
        print(f"初始 loss：{initial_loss:.6f}；训练 {args.steps} 步，lr={args.lr:g}")
        model.train()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            _, out_logits, _ = model(inputs)
            loss = F.cross_entropy(out_logits.reshape(-1, vocab_size), targets.reshape(-1))
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            item = {"step": step, "loss": float(loss.detach()), "grad_norm": grad_norm,
                    "elapsed_seconds": time.perf_counter() - clock_start}
            metrics.append(item)
            line = f"step={step:04d} loss={item['loss']:.6f} grad_norm={grad_norm:.5f} elapsed={item['elapsed_seconds']:.2f}s"
            print(line)
            log.write(line + "\n")
            log.flush()
        model.eval()
        final_loss = loss_value()
        run.update({"status": "completed", "completed_at": datetime.now(timezone.utc).isoformat(),
                    "final_loss": final_loss, "loss_reduction": initial_loss - final_loss,
                    "duration_seconds": time.perf_counter() - clock_start})
        torch.save({"model_state_dict": model.state_dict(), "config": run["model"]}, out_dir / "final_model.pt")
        (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
        log.write(f"final_loss={final_loss:.6f}\nloss_reduction={initial_loss - final_loss:.6f}\n")
        print(f"训练后 loss：{final_loss:.6f}（下降 {initial_loss - final_loss:.6f}）")
        print("运行记录：", out_dir.resolve())
        print("说明：模型反复看同一条样本，只证明可以记住这条例子；不证明新句子泛化。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="逐步检查 I love you 如何经过一个随机初始化的教学 Decoder。"
    )
    parser.add_argument("--tokenizer", default="tokenizer", help="本地 tokenizer 目录。")
    parser.add_argument("--text", default="I love you", help="输入示例句子。")
    parser.add_argument("--hidden", type=int, default=32, help="教学模型宽度，默认 32。")
    parser.add_argument("--heads", type=int, default=4, help="注意力头数，默认 4。")
    parser.add_argument("--top-k", type=int, default=5, help="显示末位置 top-k 候选。")
    parser.add_argument("--seed", type=int, default=7, help="随机初始化种子。")
    parser.add_argument("--train-demo", action="store_true", help="反复训练一条输入，演示标签错位、loss 与权重更新。")
    parser.add_argument("--steps", type=int, default=20, help="单样本训练更新步数，默认 20。")
    parser.add_argument("--lr", type=float, default=0.01, help="训练演示学习率，默认 0.01。")
    parser.add_argument("--out", type=Path, help="训练演示输出目录；目录必须不存在。")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                        help="auto 优先 CUDA，否则使用 CPU。此教学演示固定 FP32，便于观察数值。")
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    device, dtype = resolve_runtime(args.device, "float32")
    if args.heads < 1:
        parser.error("--heads 必须大于 0。")
    if args.top_k < 1:
        parser.error("--top-k 必须大于 0。")
    if args.steps < 1:
        parser.error("--steps 必须大于 0。")
    if args.lr <= 0:
        parser.error("--lr 必须大于 0。")
    if args.out and not args.train_demo:
        parser.error("--out 仅配合 --train-demo 使用。")
    if args.hidden % args.heads:
        parser.error("--hidden 必须能被 --heads 整除。")
    if (args.hidden // args.heads) % 2:
        parser.error("--hidden/--heads 必须为偶数，以便演示 RoPE。")

    torch.manual_seed(args.seed)
    torch.set_printoptions(precision=4, sci_mode=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_ids = tokenizer.encode(args.text, add_special_tokens=False)
    if not token_ids:
        parser.error("输入没有产生 token，请换一条非空文本。")
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    vocab_size = max(
        len(tokenizer),
        tokenizer.eos_token_id + 1 if tokenizer.eos_token_id is not None else 0,
        tokenizer.pad_token_id + 1 if tokenizer.pad_token_id is not None else 0,
    )

    model = TinyDenseDecoder(vocab_size, args.hidden, args.heads).to(
        device=device, dtype=dtype
    )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        embedded, logits, trace = model(ids)
        probabilities = torch.softmax(logits[0, -1].float(), dim=-1)
        top_values, top_ids = torch.topk(probabilities, k=min(args.top_k, vocab_size))

        print("模型：单层教学 Decoder；hidden=", args.hidden, "heads=", args.heads)
        print("参数量：", parameter_count, "（embedding 与输出权重共享）")
        print()
        print("=== 输入与 Tokenizer ===")
        print("文本：", args.text)
        token_pieces = [tokenizer.decode([token_id], skip_special_tokens=False)
                        for token_id in token_ids]
        print("token 文本片段：", token_pieces)
        print("IDs：", token_ids, "；ID 张量：", shape(ids))
        print()
        print("=== Embedding 与位置 ===")
        print("Embedding 查表形状：", shape(embedded), "=[batch, time, hidden]")
        print("位置索引：", list(range(len(token_ids))))
        print("第一个 token 向量前 6 维：", embedded[0, 0, :6].float().cpu().tolist())
        print("位置方法：本教学模型将 RoPE 应用于每层的 Q/K，不向 embedding 直接相加。")
        print()
        print("=== Q / K / V 与 RoPE ===")
        for name in ("q_raw", "k_raw", "q", "k", "v"):
            print(f"{name:>6} shape =", shape(trace[name]))
        print("位置 2 的 Q，RoPE 前：", trace["q_raw"][0, 0, -1, :4].float().cpu().tolist())
        print("位置 2 的 Q，RoPE 后：", trace["q"][0, 0, -1, :4].float().cpu().tolist())
        print("head 0 的 V，位置 0 前 4 维：", trace["v"][0, 0, 0, :4].float().cpu().tolist())
        print("说明：Q/K 经过 RoPE；V 保留为待加权汇总的信息。")
        print()
        print("=== Causal multi-head self-attention ===")
        print("每头 score shape：", shape(trace["scores"]), "=[batch, heads, query, key]")
        print("head 0 的原始 QK 分数（加 mask 前）：")
        print(trace["scores"][0, 0].float().cpu())
        print("未来位置 mask（-inf 表示禁止读取）：")
        print(trace["mask"].cpu())
        print("head 0 的注意力权重：")
        print(trace["weights"][0, 0].float().cpu())
        print("所有头输出：", shape(trace["head_output"]))
        print("拼接回 hidden width：", shape(trace["merged"]))
        print()
        print("=== Decoder block 与词表输出 ===")
        print("Attention 子层输出：", shape(trace["attention_output"]))
        print("MLP 子层输出：", shape(trace["mlp_output"]))
        print("Block 输出：", shape(trace["block_output"]))
        print("logits：", shape(logits), "=[batch, time, vocab]")
        print("末位置概率 top-k：")
        for rank, (prob, token_id) in enumerate(zip(top_values, top_ids), start=1):
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            print(f"  {rank}. id={int(token_id):6d} p={float(prob):.5f} token={piece!r}")

    if len(token_ids) >= 2:
        print()
        print("=== 下一个 token 训练目标（仅计算一次示例 loss，不更新权重）===")
        targets = ids[:, 1:]
        predictions = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(
            predictions.view(-1, vocab_size), targets.reshape(-1)
        )
        print("输入位置预测目标 IDs：", targets[0].tolist())
        print("用于 loss 的 logits：", shape(predictions))
        print(f"随机权重下的 next-token cross-entropy：{float(loss):.4f}")

    print()
    print("注意：这是随机初始化的单层教学模型（MLP 为 4×GELU）；top-k 和 loss 只展示计算路径。")
    print("它不代表模型学会了语言，也不等同于完整 Qwen2 的全部实现细节。")
    print("设备：", device, "；精度：", str(dtype).removeprefix("torch."))
    if args.train_demo:
        run_training_demo(args, model, tokenizer, token_ids, vocab_size, device, dtype)


if __name__ == "__main__":
    main()
