# Project Architecture Rules (Non-Obvious Only)

- 模型 = 随机初始化的 Qwen2Config 缩小版（hidden=256 / 3 层 / 4 头 MHA / SwiGLU / RoPE / RMSNorm Pre-LN / tie_word_embeddings），**绝不加载预训练权重**——"从零"是实验核心前提；词表跟随现成 Qwen2 tokenizer（151936），不自己训练 tokenizer。
- Tokenizer 与模型 config 强耦合：`VOCAB_SIZE = max(len(tokenizer), eos_id+1, pad_id+1)`（eos=151643 == vocab_size 会越界）；换 tokenizer 必须同步改 config。
- **数据格式即架构的一部分**：BPE 切分决定"符号通道"是否存在（v7.2 用 `= -X` 空格修复 `=-` 合并问题）；任何数据格式改动必须先用 check_tokens 式脚本验证 token 切分，再训练。
- 训练两种互斥模式：packing（512 块 / batch=2 / 无 mask / labels=input_ids）与 `--no-pack`（独立样本 / batch=64 / padding+mask）；数学实验必须 --no-pack，否则切块切断算式。packing 模式下每条样本前后插 eos 以对齐推理时的 eos 前缀分布。
- 同一事实多种写法的"对齐对"（如 `a+b=c` ↔ `a-(-b)=c`）是教会符号语义的关键数据设计（v7.1 引入），新数据版本应延续。
- 每 epoch checkpoint 约 130MB，`--no-checkpoints` 只存 final_model；final_model 含完整权重+config+tokenizer，可脱离脚本独立加载。
- 已知取舍：删除数轴全序列（no-pack 下长序列 padding 拖慢训练 30 倍），长序列连贯性显式放弃。
