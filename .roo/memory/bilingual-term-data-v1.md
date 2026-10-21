# 中英双语词条数据 v1（2026-10-13，18_prepare_bilingual_v1.py）

用户给定范例词条「猜/guess」（猜 = 只凭经验和直觉，没有事实依据给出的一个结果；例句「我猜的」），要求按范例格式生成多条训练样本作为双语数据格式样板。用户明确选择：只接这一个词条，不擅自发明词汇。

设计要点：

- 双语词条独立成数据集（data/bilingual_v1_train.jsonl），与代码 Package（16/17 号）分开生成、分开审计，避免 v0 的域混杂问题。
- 一个词条展开 12 种样本类型 / 14 条 / 662 token：双语定义、中→英、英→中、中/英释义、例句对照、例句多译法、例句反向、词性、用法辨析（估计/推断/计算 3 条）、场景应用、判断问答。
- 释义逐字采用用户给定内容；英文释义是对应翻译。新增词条往 TERM_ENTRIES 追加同结构条目即可。
- 无随机数天然可复现；审计含 text 唯一、类型齐全、重建逐字节一致、文件 sha256。
- 训练行带 text 字段，可直接喂 03 训练器（无需 --package-code-token）。

**环境坑（重要）**：本机页面文件太小，`import transformers` 会拉起 torch 的 CUDA DLL 导致 WinError 1455（cufft64_11.dll 加载失败）。18 号脚本改用纯 `tokenizers` 库（Tokenizer.from_file 读 tokenizer.json）统计 token，绕开 torch。后续写不需要训练的数据脚本时应沿用此做法；16/17 号脚本用 AutoTokenizer 在页面文件紧张时也可能触发同样错误。

README 3.7 节已记录。
