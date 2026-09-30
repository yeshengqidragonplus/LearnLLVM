# 仓库工作约定

唯一实验室 pretrain_lab/；唯一实验与经验入口为根目录《大模型预训练学习与实验架构方案.html》，当前命令见实验室 README。2026-09-30 已按用户要求清理旧脚本、数据、零散笔记和仅含配置的 output，勿按旧路径重建杂物。

- 所有 Python 命令在 pretrain_lab/ 运行，使用 .venv/Scripts/python。当前保留现代数学 v5、03 训练器、04e 评估器。
- 新实验可新建版本文件作对照；结束后汇总再清理。历史原文已折叠归档，旧推断不代表已证明事实。
- 数学样本 eos 包裹，推理加 eos 前缀；词表取 max(len(tokenizer), eos_id+1, pad_id+1)。
- 数学用 --no-pack --batch 16 --bucket --loss-chunk 64，按显存调整；loss 修改后用 --verify-loss。
- prompt 与训练格式逐字匹配并检查 token 前缀；负结果 = -X、负操作数括号化；比较句格式以对应生成器为准，不跨版本套用。
- NAVE 解码需一致注入前缀值（04e），05 只支持普通模型。
- Python 文件头 UTF-8 声明与目的 docstring，中文注释/参数帮助，控制台 UTF-8；下载 tokenizer 前设置 HF_ENDPOINT。
- 保留 .venv/、tokenizer/、.git/；权重、数据、日志不入 Git；未经用户要求不自动提交。
- 当前生成器 59,078 条与历史 94,942 条不同，不声称精确复现。环境变更更新锁文件和 README。

- 公开项目以跨电脑复现为目标：setup_env.py 区分 CPU/CUDA 安装，requirements-lock.txt 仅为 Windows CUDA 参考快照；不要把某台机器的环境写成通用要求。
- runtime.py 统一设备与精度选择；CUDA BF16 可用时使用 BF16，否则 FP32。macOS 暂用 CPU；未测试的平台须标明未验证。
- 每次训练独立 --out，自动保留 run.json/train.log，记录参数与哈希；不要覆盖既有实验。正式结果与短解码环境自检分开。
