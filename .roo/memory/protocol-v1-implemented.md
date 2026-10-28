# protocol_v1 已实现并跑通（2026-10-21）

`DESIGN_protocol_v1.md` 从设计稿变成可执行件。三个新文件：

| 件 | 作用 |
|---|---|
| [`tokens.json`](../../pretrain_lab/tokens.json) | 全局标记表，所有医生共用 ID（C2 硬约束） |
| [`21_rule_brain_v1.py`](../../pretrain_lab/21_rule_brain_v1.py) | 规则脑，证明协议可执行；导出轨迹作训练 teacher |
| [`22_pronoun_probe_v1.py`](../../pretrain_lab/22_pronoun_probe_v1.py) | 指示代词三轴探针（距离/干扰项/类型混淆） |

`03_train_pretrain.py` 新增 `--protocol-tokens`：按 tokens.json 顺序注册 14 个标记，
ID 不一致直接 SystemExit。

## 结果

L1 五轮轨迹实跑通过：`声明科 → 算数科 → 进制转换 → OUTPUT → DONE`，
结果 87A238H（1234567+7654321 = 8888888 = 0x87A238）。
投影全程零载荷泄漏（自测断言 1234567/7654321/8888888/87A238H 均不出现）。

探针数据 352 条 / 51,233 token，距离轴 63→414 token 单调递增，边界校验通过。

## 三个实现坑（写类似代码必须避开）

**1. 聚合科室被 BFS 剪掉。** `ArithDept` 是 `num→num` 的 kind 自环，
只按 kind 做 visited 会把它剪掉，于是规划成「声明科→进制转换」，
两个数根本没相加 —— 但轨迹仍能跑完 4 轮，**看起来成功、语义是错的**。
修法：改成逐轮动态规划（每轮只看下一步），聚合由执行时按 batch 大小发现。
只有执行后才知道某科室产出几个包（声明科产出 2 个 num）。

**2. tokenizer 未注册标记时长度统计全是假的。** `<|OUTPUT|>` 未注册时被切成
`['<','|','OUTPUT','|','>']` 5 个 token，注册后是 1 个。数据脚本必须先
`add_special_tokens` 再统计 token，否则距离轴和边界校验都不可信。

**3. `text` 必须严格等于 `prompt + target`。** 第一版 prompt 已以 `\n` 结尾，
生成 `text` 时又加了一个 `\n`，导致 88 条全部报 `--target-mask` 边界不一致。
这正是 20 号记忆里那条「prompt/target token 边界」约束的又一次实例。

## 下一步

跑 03 训练探针数据（`--protocol-tokens --target-mask --no-pack`），
得距离-准确率曲线，定投影长度上限，再回头定模板训练的投影格式。
