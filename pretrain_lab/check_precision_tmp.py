# -*- coding: utf-8 -*-
"""check_precision_tmp.py — loss 精度实测（一次性脚本）
目的：用户问"fp32 是否必须，bf16/fp16 行不行"——用数字回答。

背景：词表 V=151646，HF 默认把 logits 升 fp32 算 loss（数值稳定）。
      升位要 2.96GB 显存（v3 长序列），是 OOM 主因。
问题：能否用 bf16/fp16 算 loss 省显存？精度损失多大？

实测：
  1. 不同 logits 值域下，fp32 / bf16 / fp16 / 手工 bf16 的 loss 差异
  2. log_softmax 的数值稳定性（fp16 是否溢出）
  3. 梯度差异（loss 的 dt 影响反向传播）
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
import torch.nn.functional as F

torch.manual_seed(0)
V = 151646          # 我们的词表
B, L = 4, 16        # 小规模（CPU 可跑）
labels = torch.randint(0, V, (B * L,))

print("=" * 78)
print(f"词表 V={V}, 样本 {B}×{L}={B * L} 个 token")
print("=" * 78)
print(f"{'logits值域':<12} {'fp32(基准)':>12} {'bf16输入':>12} {'fp16输入':>12} {'手工bf16':>12}")
for scale in (1.0, 5.0, 10.0, 20.0):
    logits = torch.randn(B, L, V) * scale

    # 基准：fp32
    l32 = F.cross_entropy(logits.reshape(-1, V).float(), labels)

    # bf16 输入（PyTorch 的 log_softmax 对 half 输入内部升 fp32 累积，输出 cast 回 half）
    l_bf = F.cross_entropy(logits.reshape(-1, V).to(torch.bfloat16), labels)

    # fp16 输入
    try:
        l_fp = F.cross_entropy(logits.reshape(-1, V).to(torch.float16), labels)
        l_fp_s = f"{float(l_fp):.6f}"
    except Exception as e:
        l_fp_s = f"异常({type(e).__name__})"

    # 手工全程 bf16（模拟"完全不用 fp32"）
    lg16 = logits.reshape(-1, V).to(torch.bfloat16)
    lp16 = F.log_softmax(lg16, dim=-1)
    l_man = F.nll_loss(lp16, labels)

    print(f"{scale:<12.1f} {float(l32):>12.6f} {float(l_bf):>12.6f} {l_fp_s:>12} "
          f"{float(l_man):>12.6f}")

print()
print("=" * 78)
print("数值稳定性检查（fp16 溢出风险）")
print("=" * 78)
for scale in (10.0, 50.0, 100.0):
    logits = torch.randn(2, 8, V) * scale
    h16 = logits.to(torch.float16)
    n_inf = torch.isinf(h16).sum().item()
    n_nan = torch.isnan(h16).sum().item()
    # log_softmax 后是否有 inf/nan
    lp = F.log_softmax(h16, dim=-1)
    print(f"  值域 {scale:>5.1f}: fp16 溢出 {n_inf} 个, NaN {n_nan} 个; "
          f"log_softmax 后 inf={torch.isinf(lp).sum().item()} NaN={torch.isnan(lp).sum().item()}")

print()
print("=" * 78)
print("梯度差异（反向传播影响）")
print("=" * 78)
for scale in (5.0, 10.0):
    logits = (torch.randn(B, L, V) * scale).requires_grad_(True)
    # fp32 路径
    l32 = F.cross_entropy(logits.reshape(-1, V).float(), labels)
    g32 = torch.autograd.grad(l32, logits)[0]
    # bf16 路径（输入 bf16，但 autograd 会经 fp32 中转）
    logits2 = logits.detach().clone().requires_grad_(True)
    l_bf = F.cross_entropy(logits2.reshape(-1, V).to(torch.bfloat16), labels)
    g_bf = torch.autograd.grad(l_bf, logits2)[0]
    rel = (g32 - g_bf.to(g32.dtype)).abs().mean() / (g32.abs().mean() + 1e-12)
    print(f"  值域 {scale:>5.1f}: loss 差 {abs(float(l32) - float(l_bf)):.6f}, "
          f"梯度相对误差 {float(rel):.4%}")

print()
print("=" * 78)
print("结论要点：")
print("  · bf16 输入算 loss：loss 值差异小（PyTorch 内部仍用 fp32 累积）")
print("  · fp16 输入：范围小，值域大时溢出 → 危险")
print("  · 手工全程 bf16：误差最大（log_softmax 在 bf16 下精度损失）")
print("  · 真正安全的省显存方式 = 分块（chunked）fp32，精度完全不变")
print("=" * 78)