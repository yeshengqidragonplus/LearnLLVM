# Project Coding Rules (Non-Obvious Only)

- 新实验版本 = 新编号文件（01h→01i→01j），**绝不原地修改旧版本脚本**；旧脚本是对照基线，结果已写入根 HTML 文档。
- 数据格式 helper `op_num()`（负数括号）/`eq_res()`（负结果 `= -X`）在数据生成与评估脚本中重复定义；改格式必须同步所有副本，否则评估 prompt 与训练分布不一致，结果全错。
- `03_train_pretrain.py` 是顶层过程式代码（无 main() 包裹），argparse 在模块级执行；`--no-pack` 会把 BATCH_SIZE 从 2 改为 64 并切换 padding+mask 路径（labels=-100）。
- 推理/评估脚本必须设 `model.generation_config.pad_token_id` 与 `eos_token_id`；v4+ 模型输入 ids 必须手动加 `[eos]` 前缀。
- 模型加载统一 `.to(torch.bfloat16).to(device)`，device 取 `"cuda" if torch.cuda.is_available() else "cpu"`；生成统一贪心 `do_sample=False`。
- jsonl 数据写 `json.dumps({"text": t}, ensure_ascii=False)`；shuffle 前固定 `random.seed(42)`。
- 训练脚本 `log()` 同时输出控制台并追加 `logs/train.log`；新训练脚本应沿用此模式。
