# 代码 Package v2 数据审查（2026-10-12）

第一轮 v0（13 号脚本，61 条拼接式样本，9.80M 从零模型）负结果：train loss 11.91→9.68，12 道留出题全失败（代码解释坍缩为重复 CODE）。原因复盘：请求/标记/路由/载荷/回答拼成一段文本，未分阶段；temperature、probability 只在评估集出现，混入未见标识符泛化。

第二轮 v2（16 号脚本 + code_package_frontend_v1.py）审查通过：

- 前端自测通过：3 语句正确切分（含字符串内分号、for 头保护），载荷对路由器隐藏。
- 16 号脚本实测运行通过：`<|PKG_CODE|>` 注册为单 token ID 151646（原先 6 token）；路由载荷泄漏 0；路由 train/eval=14/6，专家 train/eval=16/6（seen_name_new_value 2 + unseen_identifier 4）；专家训练集 1,064 token。
- 数据文件抽查：路由输入只有"意图+包类型"，无代码；专家 prompt 含子包边界与载荷；多语句样本（int total; total = 15;）正确拆成两个子包。
- 03 训练器已加 `--package-code-token` 开关注册专用 token 并随模型保存。

已知问题（不阻塞数据，但训练前要解决）：

1. **03 训练器只认 `text` 字段**（03_train_pretrain.py:126 `json.loads(line)["text"]`），v2 专家集是 `prompt`/`target` 分离字段，直接喂会 KeyError。需要转换脚本或改训练器支持 prompt+target 拼接（且只对 target 计 loss 才符合分阶段设计）。
2. **样本量极小**：专家训练 16 条/1,064 token，v0 用 61 条/2,663 token 都没学成；v2 若原样训练大概率复现 v0 失败。README 已自我声明"只够接口检查"。
3. 路由 eval 6 条中 2 条 CODE_EXPLAIN 与 train 意图措辞接近（"请不要运行，只说明语句的意义" vs train 的"解释这段代码，不要执行"风格），小样本下路由器可能靠关键词而非语义。
4. eval 的 unseen 组（temperature/probability/capacity/sample_count）与 v0 留出集相同，跨轮对比时注意这 4 个标识符在 v0 也是未见的。

结论：数据与流水线本身合格，隔离设计正确；但"数据准备好了"≠"可以训练"，训练入口（text 字段适配）和样本量扩容是两个前置缺口。

## v3 扩容（2026-10-12，17_prepare_code_package_v3.py，用户确认后实施）

针对上述缺口 1、2 的解决：

- **规模**：专家训练 1,000 条/68,460 token（v2 的 64 倍）；专家 eval 每组 24 条（seen_name_new_value + unseen_identifier，对齐 08/v5-C1 判据）；路由 train/eval 每类 100/24 条。
- **多样性来源**：240 个组合标识符池（72 名词 × 16 修饰词/数字后缀）× 10 种 C 类型（新增 unsigned short/long/char）× 训练/评估分离值域随机字面量 × 12 问法模板 × 2 答法变体 × 4 种句式（单声明 640/只声明 120/声明+赋值 120/双声明 120）。
- **隔离审计全过**：载荷泄漏 0、路由 train/eval 零重叠、未见标识符子串级零泄漏（训练池刻意不含 sample，避开 sample_count）、seen eval 名字确在训练中出现、prompt 唯一、同种子复生成逐字节一致（seed=20261012，audit 含文件 sha256）。
- **text 字段适配**：训练行带 `text`（专家=prompt+target，路由=router_input+路由选择），可直接喂 03 训练器 `--package-code-token`；但 03 对整条 text 计 loss，无 target-only 掩码——分阶段设计完全落地仍需改训练器（缺口 1 只解决了一半）。
- eval unseen 组前 4 个标识符（temperature/probability/capacity/sample_count）沿用 v0/v2，便于跨轮对比。
- README 3.6 节已记录；数据在 data/code_package_v3_*.jsonl（gitignore）。
