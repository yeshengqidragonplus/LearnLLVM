# NAVE 设计方案：数字值嵌入（Number-Aware Value Embedding）

> 版本：v2 预研（2026-09-15）｜产出：设计文档，未动代码
> 核心假设（用户）：一个数应在潜空间被表示为一个「整体/值」，而非一串无关字符 token。

---

## 1. 问题定位（v1 实测）

- **现象**：`问: 11962+1685 = ?` → `数位: 1192=...`（万位丢失）；`s(99779)` → `s(9979)`。
- **归因**：复制/读取通道**宽度绑定**（注意力学的是"取回 N 个数字 token"），不是算法不会。
- **反证**：训练宽度内加法 CoT 83~87% 且挖洞通过；序公理（无需逐位复制）五位数外推 100%。

### 1.1 排除的两条路

| 方案 | 为何不选 |
|---|---|
| 原子 token（给 0..99999 各加一个词表 token） | 词表 +10 万（6 位要 +100 万），**完全无法外推**到训练范围之外，违背"只训 3 位扩展到 N 位"的目标 |
| 纯数据铺宽（加 5/6 位样本） | 墙只会移动不会消失（旧系列 v8 已证：3→5→6 位墙逐级上移） |

### 1.2 选定的方向

**值嵌入（value embedding）**：在输入序列里，让每个数字所在位置携带"它代表的数值"的连续向量表示。
好处：(a) 数在潜空间里成为"整体"；(b) 相邻数值表示相近 → 天然数轴几何；(c) 用**数位×位权**的可加编码 → 宽度可外推。

---

## 2. 因果安全的关键约束（决定一切设计）

自回归模型**从左到右**读。读 `11962` 时：
- 读到第 1 位 `1` 时，**不可能知道**这个数是 11962（未来未知）→ 不能在每个位置注入"完整数值"。
- 读到第 5 位 `2` 时，"前缀值"= 11962 = 完整值。

因此设计规则（**NAVE-prefix**）：

> 对每一个数字 token，注入「当前连续数字串的**前缀值**」`v_prefix` 的值嵌入。

- 因果安全：`v_prefix` 只由**已出现**的数字决定（含自身），不泄漏未来的位。
- 训练与推理同一规则：prompt 与逐 token 生成都成立，无需"知道数字何时结束"。
- 数字串末位处，前缀值 = 完整值 → 后续生成（CoT/答案）可经注意力读到该位置的"整值"。

示例（`11962`）：

```
pos: 1    1    9    6    2
v:   1    11   119  1196 11962   <- 注入值
```

---

## 3. 值编码器（e_val）：为什么用「数位×位权」可加编码

直接 `MLP(标量值)` 在大数上外推差（训练 ≤4 位，5 位的数量级超出）。
改用**与 CoT 同构**的可加分解：`v = Σ_i d_i · 10^i`，
表示向量 = Σ_i (digit_emb[d_i] + place_emb[i])，再过一个零初始化的投影。

- **可加 → 宽度可外推**：多一位只是多一项，不改变已有项的语义。
- **与数据同构**：CoT 里 `数位: 4×1000+4×100+8×10+3` 就是这个分解，表示与算法共享结构。
- **几何**：高位相同的数共享项 → 表示相近，形成数轴邻近性。

参数（hidden=384）：`digit_emb 10×384 + place_emb 8×384 + proj 384×384 ≈ 0.30M`（可忽略）。

---

## 4. 实现细节（伪代码）

```python
# nave.py
class NumberValueEncoder(nn.Module):
    def __init__(self, hidden, max_place=8):
        super().__init__()
        self.digit_emb = nn.Embedding(10, hidden)
        self.place_emb = nn.Embedding(max_place, hidden)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        nn.init.zeros_(self.proj.weight)          # 零初始化：不扰动起点

    def encode(self, value):                       # value: (B,L) long，非数字位置为 -1
        d = torch.zeros_like(value)                # 逐位分解 value
        p = torch.zeros_like(value)
        valid = value >= 0
        v = value.clone()
        place = 0
        while True:
            live = valid & (v > 0) | (valid & (place == 0))   # place0 处理 value==0
            if not live.any(): break
            d[live] = v[live] % 10
            p[live] = place
            v[live] = v[live] // 10
            place += 1
        e = self.digit_emb(d) + self.place_emb(p.clamp(max=7))
        e = self.proj(e)
        e[~valid] = 0
        return e
```

前缀值计算（因果、向量化）：

```python
def prefix_values(input_ids, digit_ids):          # digit_ids: 10 个数字 token id 的集合
    is_digit = torch.isin(input_ids, digit_ids)
    run = torch.zeros_like(input_ids)
    cur = torch.zeros_like(input_ids)
    for t in range(input_ids.shape[1]):            # 顺序扫描（因果）
        d = (input_ids[:, t] - digit0_id)          # 该位置数字 0-9
        cur = torch.where(is_digit[:, t], cur * 10 + d, torch.zeros_like(cur))
        run = torch.where(is_digit[:, t], cur, torch.full_like(cur, -1))
    return run
```

注入（训练/推理统一入口）：

```python
def embed_with_nave(model, enc, input_ids):
    base = model.get_input_embeddings()(input_ids)
    pv = prefix_values(input_ids, DIGIT_IDS)
    return base + enc.encode(pv)
```

---

## 5. 训练脚本（03）改动

1. `import nave`；新增 `--nave` 开关（默认关 → 对照组完全不受影响）。
2. 建编码器并**挂到 model 上**（`model.nave = enc`），使其进入 `model.parameters()` → 优化器/`save_pretrained`/`from_pretrained` 全部自动覆盖。
3. 训练/评估循环把 `input_ids` 换成 `inputs_embeds = embed_with_nave(...)`，
   并在 shift 时同步 `labels`（保留 `-100` 逻辑）；其余超参/调度/warmup 一律不动。
4. **不改** `03` 的既有分支语义，`--nave` 关闭时行为与 v1 训练完全一致（保证可复现对照）。

### 5.1 注意

- `inputs_embeds` 与 `labels` 同时传入是 HF 支持的；`use_cache=False` 训练时无影响。
- 与 `tie_word_embeddings` 无冲突：NAVE 参数独立于共享词嵌入。

---

## 6. 评估脚本（04）改动

生成不能用 `model.generate`（新 token 走内建 embed_tokens，不会注入 NAVE → 与训练不一致）。
改为**手写贪心解码循环**（~25 行，带 KV cache），每步：`embeds=embed_with_nave(...)` → 取 argmax → 拼回 ids。
收敛判据/`[S0]` 前缀自检照旧。

---

## 7. 防泄漏单元测试（必须先过）

- **因果性**：对随机序列，把 `t` 之后的 token 全部改写，断言位置 `≤ t` 的 `v_prefix` 与前向结果逐元素相等。
- **范围**：`11962` 的 `v_prefix` 序列必须恰为 `[1,11,119,1196,11962]`。
- **边界**：`0`、连续数字串、非数字夹断、序列以数字结尾，均不报错且语义正确。
- 不过 → 不出训练（防止"注入的是未来值"的隐形泄漏）。

---

## 8. 对照实验设计（关键）

| 组 | 数据 | 结构 | 目的 |
|---|---|---|---|
| C0 | v1 原数据 | 无 NAVE | 基线（已有：E1-L3 0%、E2-L3 0%） |
| C1 | v1 + 读位行/6 位锚点 | 无 NAVE | 纯数据能否解决（治标验证） |
| C2 | v1 原数据 | **+NAVE** | **架构单独能否外推（核心假设验证）** |
| C3 | v1 + 数据补强 | +NAVE | 组合最优 |

### 8.1 新增探针

- **copy probe（读取探针）**：`读: 11962` → `11962`。直接测量"读取通道"宽度能力（与算法解耦）。
- **宽度扫描**：4/5/6/7 位分别统计后继、加法、copy 准确率（画出"墙"在哪里）。
- **数轴几何度量**：`cos(e_val(v), e_val(v+1))` 随 v 的变化；若远高于随机向量基线 → 值表示真的形成了邻近性。

### 8.2 判据

- C2 在 copy probe 与 E1-L3 上显著优于 C0 → 「值嵌入使数成为整体」成立。
- C2 ≈ C0 → 读的问题不在表示层而在注意力复制 → 转向数据侧（见 §9 预案）。

---

## 9. 风险与预案

| 风险 | 预案 |
|---|---|
| 零初始化导致 NAVE 长期不激活 | 监控 `proj.weight` 范数与 `e_val` 输出的 cos 相似度；必要时给 e_val 一个小的非零初值/`proj` 用 xavier |
| 注入干扰普通语言 token（`×1000` 里的 1000 也被注入值） | 可加"排除乘法上下文"的裁剪；先用统一注入观察，若有害再裁剪 |
| 手动解码循环有 bug | 先用 `prompt→greedy` 与关闭 NAVE 的 `generate` 对拍（应完全一致） |
| NAVE 无效（C2≈C0） | 数据侧预案：**数字倒序书写 / 定宽补零**（把"从个位对齐"变成左到右顺序复制，取消前瞻需求） |
| 8 层深度限制仍在 | 属已知边界；NAVE 只解"读/整体"，不承诺解"长进位链" |

---

## 10. 实施步骤与验收

1. 写 `nave.py`（编码器 + 前缀值 + 注入 + 单测），过 §7 全部单测。
2. `03` 加 `--nave`（默认关），跑通 1 个 epoch 冒烟（loss 正常下降、无 NaN）。
3. 生成 C1 数据（读位行 + 6 位锚点，改 `01`→`01b`，遵循"新版本新文件"规则）。
4. 训练 C2（v1 数据 + NAVE，8 层，8 epoch）；若时间允许并行跑 C1。
5. 用扩展后的 `04` 评估（copy probe + 宽度扫描 + 几何度量 + 原 E1~E6）。
6. 汇总对照表，判定核心假设；据此定 v2 的最终形态。

**验收标准（核心）**：C2 在 **copy probe 与 5 位数后继(s)** 上相对 C0 有**决定性提升**（如 0% → ≥60%），则"数字值嵌入 = 数作为整体"得到验证，进入 C3。
