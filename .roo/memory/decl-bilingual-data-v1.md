# C/C++ 声明双向数据 v1（2026-10-19 修复后，20_prepare_decl_bilingual_v1.py）

用户方案 A：`int i = 3;` ⇄ "声明一个整型变量 i，初始值是 3"。语句打成 `<|PKG_CODE|>` 子包，语句内部的数值/标识符再打成嵌套原子包 `<|PKG_NUM|>`/`<|PKG_NAME|>`；反向（中文描述→代码）由 `package_description` 打包，路由器看不到载荷，专家解包后写代码。

- 前端扩展在 `code_package_frontend_v1.py`：`extract_atoms`、`expert_input_nested`、`package_description`、`expert_input_description`；`package_request(nested=False)` 默认保持旧行为，v2/v3 数据 sha256 不变。
- 数据：**920 条 / 76,442 token**（正向 420、反向 420、分号 80）；7 类型 × 4 形态各精确 15+15 条。评估 60 条（24/24/12），20 个未见名字子串级零泄漏。

## 三个缺口已全部补齐（2026-10-19）

1. **made 计数 bug 已修**：`add()` 改为返回 bool，正/反向分别计数，不收敛直接抛错并提示扩名字池。旧版 888 条 → 现 920 条，每组合精确 15+15。
2. **target-only loss 已实现**：`03_train_pretrain.py --target-mask`，序列 `[eos]+prompt+target+[eos]`，labels 前 `1+len(prompt_ids)` 位置 -100。**必须配 `--no-pack`**（否则直接报错）。实测监督 token 30.0%，`--verify-loss` 对拍相对差 0.0000%。
3. **评估脚本已建**：`04f_eval_decl_v1.py`，按方向分三组（代码→注释 / 注释→代码 / 分号），每档给严格+部分两个分数，加载时断言三个 Package 标记为单 token，`--out` 可存逐条结果。

## 关键坑：prompt/target token 边界

`--target-mask` 的正确性依赖 `encode(prompt+target) == encode(prompt)+encode(target)`。若 BPE 在拼接处合并 token，监督位置会整体错位且**不报错**，训练白跑。因此 20 号审计与 03 号训练器都显式断言这一点（本数据集开头是 ASCII 的 `<|PKG_CODE|>`，实测 920/920 满足；但从中文开头的 prompt 会失败）。

## 训练命令（三开关必须齐）

`--package-code-token --package-nested-tokens --target-mask`，标记注册后为 151646/151647/151648，vocab=151649。链路自检 hidden 128/layers 2/CPU/2 epoch 通过（train_loss 10.38→8.54，eval 9.05→8.36）。

## 剩余风险

19.8M / 2 epoch 冒烟模型评估 0/60（输出已收敛到"声明一个…初始值是…"骨架），只证明链路通，不证明能力。真实训练后需重跑 04f 并与 60 条留出集对比。