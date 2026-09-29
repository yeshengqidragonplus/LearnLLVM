# Project Debug Rules (Non-Obvious Only)

- 模型输出"绝对值对、符号丢"时，先查 **BPE 切分**而非模型能力：写 `check_tokens_tmp.py` 式脚本打印训练文本 vs 测试前缀的 token 切分（v7.1 根因是 `=-` 被合并成单 token，符号通道被吞）。
- 训练日志在 `logs/train.log`（log() 追加写）；后台任务的 stdout/stderr 分别重定向到 `logs/v7x_stdout.log` / `v7x_stderr.log` 与 `logs/eval_v72*.log`。
- 逐层 next-token 分布用 `08_logit_lens.py`（tie_embeddings，lm_head 与 embedding 共享权重，hidden @ lm_head.T 即该层预测）；单点 logits 探针用 `04i/04j/04k_probe_*.py`。
- Windows 控制台中文乱码：脚本已内置 `sys.stdout.reconfigure(encoding="utf-8")`；跑 `05_interact.py` 需加 `-X utf8`。
- eval_loss **先降后升是预期过拟合**（极小数据集），不是 bug；eval_loss 最低点的 checkpoint 泛化最好。
- 范围外算式失败（如 `51+1=47`）是已知边界（|结果|>50 未训练），记录在 HTML 文档 13.5.6，无需排查。
