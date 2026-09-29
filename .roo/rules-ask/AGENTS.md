# Project Documentation Rules (Non-Obvious Only)

- 根目录《大模型预训练学习与实验架构方案.html》是权威实验记录（13.x 章逐版本记录数据设计、结果表、根因分析、已知边界）；`pretrain_lab/README.md` **已过时**——只覆盖最早的 400 条 toy 实验，提到的 `output/` 目录已不存在。
- 脚本编号 = 管线阶段：01 数据生成 / 02 tokenizer 下载 / 03 训练 / 04 推理·评估·探针 / 05 交互 / 06 表示检查 / 07 probe / 08 logit lens；字母后缀 = 版本迭代（01h=v7、01i=v7.1、01j=v7.2）。
- 当前最优模型是 `output_math_v72/final_model/`（v7.2：训练范围 15/15 全对，负号/负负得正已修复）；`output_math_v71/` 是失败对照（符号全丢）。
- `_*` 前缀脚本（_locate_doc/_update_doc_v7/_fix_doc_esc）是 HTML 文档维护工具，非实验代码；`check_*_tmp.py` 是一次性调试残留，可忽略。
- 实验的诚实边界（长序列不续写 10、外推失败、范围内记忆非真代数）记录在 HTML 13.5.6 节，回答效果问题时先查这里。
- 后续路线定稿：Dense → GQA → MoE（只换 FFN）→ SFT → DPO；当前处于 Dense 数学 v7.2 阶段。
