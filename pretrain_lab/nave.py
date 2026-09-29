# -*- coding: utf-8 -*-
"""nave.py — NAVE 数字值嵌入（Number-Aware Value Embedding）

目的（见 DESIGN_nave_v5.md）：让"数"在潜空间成为**整体/值**，
而非一串无关字符 token —— 以绕开 v4-A 发现的"复制通道定宽"墙。

核心设计：
  1. **因果安全前缀值**（继承 DESIGN_nave_v2）：自回归从左到右读，
     每个数字 token 注入的是"**已出现数字构成的前缀值**"（不泄漏未来位）。
     `11962` → 注入 `[1, 11, 119, 1196, 11962]`
  2. **组合编码（可外推）**：`v = Σ_i d_i·10^i` →
     `Σ_i (digit_emb[d_i] + place_encode(i))`，多一位只是多一项。
  3. **连续位权编码（RoPE 式，v5 新增）**：位权/指数用三角函数而非查表
     → 参数 0 个、范围**任意**（10^1000 也只需指数 1000）。
     这是用户"科学计数法"洞察的技术落地。
  4. **零初始化 proj**：训练起点等价于无 NAVE（不扰动基线），
     梯度让值信息渐进注入。

参数：digit_emb 10×H + proj H×H ≈ 0.15M（对 65M 可忽略，与"有多少个数"无关）
"""
import torch
import torch.nn as nn


def prefix_values(input_ids, digit_ids):
    """计算"因果安全的前缀值"（逐列因果扫描）

    Args:
        input_ids: (B, L) long
        digit_ids: list of token ids，**必须按 '0'..'9' 顺序**（id2val = 下标）
    Returns:
        (B, L) long，非数字位置为 -1；数字位置为"从该连续数字串起点到本位置"的值
    例：'11962' -> [1, 11, 119, 1196, 11962]
    """
    B, L = input_ids.shape
    device = input_ids.device
    digit_ids = list(digit_ids)
    id2val = {tid: i for i, tid in enumerate(digit_ids)}   # id -> 0..9

    run = torch.full((B, L), -1, dtype=torch.long, device=device)
    cur = torch.zeros(B, dtype=torch.long, device=device)   # (B,) 当前前缀值
    for t in range(L):
        col = input_ids[:, t]                               # (B,)
        # 本列数字值（非数字置 0，随后由 is_d 掩掉）
        col_dig = torch.zeros(B, dtype=torch.long, device=device)
        is_d = torch.zeros(B, dtype=torch.bool, device=device)
        for tid, val in id2val.items():
            hit = (col == tid)
            is_d = is_d | hit
            col_dig = torch.where(hit, torch.full_like(col_dig, val), col_dig)
        # 因果递推：数字则 cur=cur*10+d，否则重置
        cur = torch.where(is_d, cur * 10 + col_dig, torch.zeros_like(cur))
        run[:, t] = torch.where(is_d, cur, torch.full_like(cur, -1))
    return run


class NumberValueEncoder(nn.Module):
    """值编码器：值 -> 潜空间向量（组合编码 + 连续位权）"""

    def __init__(self, hidden, n_theta=8, max_place_slots=0):
        """n_theta: 连续位权编码的角度数（2*n_theta 维输出）
        max_place_slots: 保留参数（0 = 纯连续编码，无查表上限）
        """
        super().__init__()
        self.hidden = hidden
        self.n_theta = n_theta
        self.digit_emb = nn.Embedding(10, hidden)
        # 连续位权/指数编码：θ_j = 1/10000^(j/n_theta)（同 RoPE）
        theta = 1.0 / (10000.0 ** (torch.arange(n_theta).float() / n_theta))
        self.register_buffer("theta", theta)
        # 连续编码输出 2*n_theta 维 -> 投影回 hidden
        self.place_proj = nn.Linear(2 * n_theta, hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.proj.weight)        # 零初始化：起点等价于无 NAVE

    def place_encode(self, k):
        """连续位权/指数编码（k 可为任意大整数）
        Args:
            k: (...,) long tensor（位权索引或指数）
        Returns:
            (..., hidden)
        """
        ang = k.unsqueeze(-1).float() * self.theta        # (..., n_theta)
        feat = torch.cat([ang.sin(), ang.cos()], dim=-1)  # (..., 2*n_theta)
        return self.place_proj(feat.to(self.place_proj.weight.dtype))

    def encode(self, values):
        """values: (B, L) long，非数字位置 -1
        Returns: (B, L, hidden)，非数字位置为 0
        """
        B, L = values.shape
        valid = values >= 0
        v = values.clamp(min=0)

        acc = torch.zeros(B, L, self.hidden, device=values.device,
                          dtype=self.digit_emb.weight.dtype)
        place = 0
        remaining = v.clone()
        alive = valid.clone()
        # 逐位分解（最多到 values 的最大位数）
        while alive.any():
            d = remaining % 10
            # digit_emb[d]（只对 alive 位置有意义）
            de = self.digit_emb(d)
            # 连续位权编码
            pe = self.place_encode(torch.full_like(d, place))
            acc = acc + torch.where(alive.unsqueeze(-1), de + pe,
                                    torch.zeros_like(de))
            remaining = remaining // 10
            alive = alive & (remaining > 0)
            place += 1
            if place > 64:          # 安全上限（防止异常输入死循环）
                break

        out = self.proj(acc)
        out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
        return out


class NaveWrapper(nn.Module):
    """把 NAVE 挂到模型上，并统一"训练/推理"的嵌入入口"""

    def __init__(self, model, hidden, digit_ids, n_theta=8):
        super().__init__()
        self.enc = NumberValueEncoder(hidden, n_theta=n_theta)
        self.digit_ids = list(digit_ids)     # 必须按 0..9 顺序
        self.digit_set = set(digit_ids)
        # 注意：不把 model 设为子模块（避免参数重复注册）；
        # 调用方负责把 self.enc 挂到 model.nave 以进入 parameters()

    def embed(self, model, input_ids):
        """返回 inputs_embeds = 基础嵌入 + NAVE 值嵌入"""
        base = model.get_input_embeddings()(input_ids)
        pv = prefix_values(input_ids, self.digit_ids)
        ve = self.enc.encode(pv)
        return base + ve.to(base.dtype)


# ================= 防泄漏单元测试 =================
def _selftest():
    """DESIGN_nave_v5 §4 的四项单测（必须先过才允许训练）"""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 构造假 digit_ids：假设 '0'..'9' 的 id 是 100..109（测试用）
    digit_ids = list(range(100, 110))
    DIG = {d: 100 + d for d in range(10)}

    def ids_from_str(s):
        """把 '11962' -> [101,101,109,106,102]；非数字字符跳过（用 999 表示）"""
        return [DIG[int(c)] if c.isdigit() else 999 for c in s]

    print("[单测1] 范围：'11962' 的前缀值应为 [1,11,119,1196,11962]")
    x = torch.tensor([ids_from_str("11962")])
    pv = prefix_values(x, digit_ids)
    got = pv[0].tolist()
    expect = [1, 11, 119, 1196, 11962]
    assert got == expect, f"期望 {expect} 得 {got}"
    print(f"   OK {got}")

    print("[单测2] 因果性：改写 t 之后的 token，位置 ≤ t 的前缀值不变")
    x2 = torch.tensor([ids_from_str("11962")])
    x2_mod = x2.clone()
    x2_mod[0, 3:] = DIG[9]      # 把第 3 位之后全改成 9
    pv2 = prefix_values(x2, digit_ids)
    pv2_mod = prefix_values(x2_mod, digit_ids)
    assert pv2[0, :3].tolist() == pv2_mod[0, :3].tolist(), "因果性破坏！"
    print(f"   OK 前 3 位一致：{pv2[0, :3].tolist()}")

    print("[单测3] 边界：0 / 连续串 / 非数字夹断 / 串尾数字")
    cases = [
        ("0", [0]),
        ("007", [0, 0, 7]),               # 前导零（前缀值 0,0,7）
        ("12x34", [1, 12, -1, 3, 34]),    # x=999 非数字
        ("9", [9]),
    ]
    for s, exp in cases:
        xc = torch.tensor([ids_from_str(s)])
        pvc = prefix_values(xc, digit_ids)[0].tolist()
        assert pvc == exp, f"{s}: 期望 {exp} 得 {pvc}"
        print(f"   OK {s!r} -> {pvc}")

    print("[单测4] 编码器：可跑通 + 零初始化输出为 0 + 大数不崩")
    enc = NumberValueEncoder(hidden=16, n_theta=4)
    pv_big = torch.tensor([[1, 12, 123, 1234, 12345, 123456]])
    out = enc.encode(pv_big)
    assert out.shape == (1, 6, 16)
    assert torch.allclose(out, torch.zeros_like(out)), "零初始化下输出应全 0"
    print(f"   OK 形状 {tuple(out.shape)}，零初始化输出全 0")
    # 打乱 proj 后再跑（模拟训练后），确认大数/大指数不崩
    with torch.no_grad():
        enc.proj.weight.normal_(0, 0.02)
    out2 = enc.encode(pv_big)
    assert not torch.isnan(out2).any() and not torch.isinf(out2).any()
    # 指数编码极大 k
    pe = enc.place_encode(torch.tensor([0, 23, 1000, 10 ** 6]))
    assert not torch.isnan(pe).any() and not torch.isinf(pe).any()
    print(f"   OK 大数前缀 {pv_big[0].tolist()} 与指数 k=10^6 均无 NaN/Inf")

    print("\n✅ nave.py 全部单测通过")


if __name__ == "__main__":
    _selftest()