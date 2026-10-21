# 续训（--init-from）与 Package 标记词表冲突（2026-10-21）

`03_train_pretrain.py` 的 `--init-from` 只做架构一致性校验，**不做 `resize_token_embeddings`**：

- 加了 `--package-code-token --package-nested-tokens` 后 `VOCAB_SIZE = 151649`（三个标记 151646/151647/151648）；
- 而现存 `output_v5c1/final_model` 的 embedding 是 `[151646, 384]`（无 Package 标记）；
- 于是 line 230 的 `cfg.vocab_size != VOCAB_SIZE` 直接报错，**无法从 v5c1 数学基座续训到声明数据**。

要用课程学习（数学基座 → 声明/代码数据）必须先二选一：
1. 给 03 加 `--resize-embeddings`（冻结旧行、只随机初始化新增 3 行，并同步 `tie_word_embeddings`）；
2. 或把 Package 标记改成不扩词表的编码方式（如用已有 token 序列拼装）。

另注：`output_v1/v2/v3/v4a/v5c1` 与 `output_smoke_nave` 目录下**没有 run.json**（只剩权重），它们不能当作有留痕的对照基线；新实验仍按"每次独立 --out"重新留痕。
