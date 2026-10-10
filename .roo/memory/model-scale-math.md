# 文本基座规模数学（2026-10-10 推导）

词表 V=151,644（Qwen2，eos=151643，见 03_train_pretrain.py:131 注释），tied embedding（:181），SwiGLU intermediate=8H/3。

参数公式：总参数 = V×H（嵌入，tied 共享给 LM-Head）+ L×12H²（每层注意力 4H² + MLP 8H²）。

- v5 实测构成：H=384/L=4 → 58.2M 嵌入 + 7.1M 层 = 65.3M。嵌入占 89%。
- 候选档位：v2a=512/8/8→102.8M；v2b=640/12/10→156.1M；v3=896/16/14→290M。
- 关键结论：151,644 词表下小模型嵌入占比 45-90%，"参数量"对比必须看层参数；tied 时 LM-Head 逐 token 活跃，训练 FLOPs/token ≈ 6×总参数（6ND 中 N=总参数成立）。
- 训练时长：4070 Laptop BF16 小模型有效吞吐假设 2-4 TFLOPS（待用 v5 run 的 tokens/s 实测校准，run.json/train.log 有留痕）；Chinchilla 20:1 → v2a(2B tok)≈4-8 天，v3(5.8B tok)≈1-2 个月连续运行。
- v3 训练显存：290M×14B(AdamW+FP32 master)≈4.1GB + 激活 ≈6.5-7GB，8GB 紧但可行（loss-chunk+batch 调整）。
