# 预训练实验室：安装、复现与记录

所有训练、生成和评估命令在本目录运行。两台电脑使用同一份代码，各自创建虚拟环境；不要复制 .venv。项目入口见 [根目录 README](../README.md)，实验经验集中在 [总册](../大模型预训练学习与实验架构方案.html)。

## 前言课与三课 HTML 教材

- [前言课：语言、意义与结构](courses/course-00-preface.html)
- [第 1 课：稠密 Transformer、GPT 与预训练全流程](courses/course-01-dense-decoder.html)
- [第 1 课延伸 Demo：注意力、上下文与 RNN](courses/course-01b-attention-context-rnn.html)
- [Dense 前向逐步演示脚本](12_dense_walkthrough_v1.py)：默认逐步打印前向张量；加 `--train-demo --steps 20` 可观察单样本训练、loss 与梯度，并自动记录到独立 output 目录。
- [第 2 课：MoE 专家架构与数字复制宽度实验](courses/course-02-moe.html)
- [第 3 课：Package + 新式路由与渐进式专家](courses/course-03-package-router.html)

各页区分理论、可运行示例、实际记录和未验证预期；实验与经验仍以根目录总册为唯一汇总入口。

## 1. 安装环境

建议 Python 3.11（本机实测 3.11.9）。安装器将 PyTorch 后端与通用依赖分开：requirements.txt 不绑定 CUDA；requirements-lock.txt 只是 Windows / Python 3.11 / CUDA 12.6 的参考快照，不适合作为所有电脑的安装入口。

**Windows / 两台 NVIDIA 笔记本**：

```powershell
cd pretrain_lab
py -3.11 setup_env.py --backend cuda
.\.venv\Scripts\Activate.ps1
python check_env.py --device cuda
```

也可用 `./setup_env.ps1 -Backend cuda`；若 PowerShell 不允许激活脚本，直接使用 `.venv/Scripts/python.exe` 代替后续命令里的 `python`，不必修改系统执行策略。

**Linux / NVIDIA**：

```bash
cd pretrain_lab
python3.11 setup_env.py --backend cuda
source .venv/bin/activate
python check_env.py --device cuda
```

**无 NVIDIA 的电脑 / macOS**：

```bash
cd pretrain_lab
python3.11 setup_env.py --backend cpu
source .venv/bin/activate
python check_env.py --device cpu
```

Windows CPU 安装同样用 `py -3.11 setup_env.py --backend cpu`。已有 CUDA 环境可直接 `--device cpu` 跑 CPU，无需重装；若要安装单独的 CPU wheel，用 `--venv .venv-cpu` 新建环境并激活它。安装器不会把已有不同后端的环境直接覆盖。

默认 PyTorch 2.14.0，CUDA 通道 cu126，通用依赖 transformers 5.16.1 / sentencepiece 0.2.2。`--backend auto` 用 nvidia-smi 探测，不保证驱动兼容；安装后的 GPU 自检才是真正的验收。需要其他官方组合时用 `--cuda cuXXX --torch-version X.Y.Z`，并按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) 或 [版本表](https://pytorch.org/get-started/previous-versions/) 选择。cu126 指 wheel 的运行库，不是必须在电脑上单独装的 Toolkit。其他版本需重新验证。

先看安装计划（不改环境）：`python setup_env.py --backend cuda --dry-run`。本机环境无需重复下载依赖；其他操作系统和 CPU-only wheel 尚未实际安装验证。macOS 当前走 CPU，MPS、AMD/Intel GPU 加速不在已支持范围。

## 2. 换电脑先跑短实验

```bash
python smoke_test.py --device auto
```

自动完成普通模型/NAVE 前后向、NAVE 因果性自检、12 条样本训练、loss 对拍、保存、重载和每题 4 token 短解码；结束后清理临时产物。也可以 `--device cpu` 检查 CPU 路径。短解码分数不代表数学能力。

## 3. Package 前置打包原型（v1）

先验证 package 边界、规则路由和计算器专家的最小闭环；当前是可替换的规则路由基线，尚未训练 Transformer 路由器：

```powershell
python 06_package_router_v1.py --self-test
python 06_package_router_v1.py --check-tokenizer tokenizer
python 07_train_package_router_v1.py --train --device cuda
python 07_train_package_router_v1.py --text "请计算 1234567+2222"
python 07_train_package_router_v1.py --text "请对 1234567+2222 做竖式计算"
```

`06_package_router_v1.py` 是规则路由基线；`07_train_package_router_v1.py` 训练约 0.876M 参数的 Transformer 分类路由器，并把它接到整数加法专家。训练样本仅让路由器看到请求文本和通用 package 标记，不含算式载荷；专用 `PKG_ARITH` embedding 属于这个小路由器的 259 项 UTF-8 字节词表，不是 Qwen tokenizer 的扩展。专家从运行时会话表取回表达式并计算。未识别/不支持的请求输出 `UNKNOWN`；执行前另有严格路由门拦截步骤请求和不明确意图，避免误调用计算器。v1 只识别一个正整数二元加法表达式。首轮训练句式内路径可用，但未见措辞测试准确率 58.3%，不代表通用自然语言路由已解决。训练权重写到被忽略的 `output_router_v1/`。

针对裸算式、紧贴“计算”的算式和多个空格的请求，先只准备第二轮固定数据并检查：

```powershell
python 07_train_package_router_v1.py --prepare-data --samples-per-class 1200 --out output_router_v1_round2
```

数据写入 `output_router_v1_round2/data/`（训练集与留出集各一个 JSONL），命令不会训练。审核后再以同一数据训练：

```powershell
python 07_train_package_router_v1.py --train --data-dir output_router_v1_round2/data --out output_router_v1_round2 --device cuda --batch 64 --epochs 16
```

连续水平空白会在 package 替换后折叠为一个空格再进入路由器；原始 payload 不变，仍可逐字取回。训练数据包含零空格和规范化后的单空格输入，不需要让模型记住两个/多个空格的差别。每轮会即时输出 train loss、留出集总体及分类准确率、累计耗时和 CUDA 峰值显存；本轮配置为 2,400 条样本、38 个 batch/轮、共 608 次参数更新。裸算式标为 CALCULATOR；规则路由门也允许只有一个算式 package 的请求直接调用计算器。

2026-10-01 第二轮在 RTX 4070 Laptop CUDA 完成：约 6.5 秒，峰值显存约 0.21 GiB。训练 loss 降至近 0，但 13 条留出集准确率从前期最高 84.6% 回落至最后一轮 61.5%（CALCULATOR 80%、UNKNOWN 50%），最终保存的是第 16 轮。对原先失败的多空格、单空格、无空格和裸算式示例均成功调用整数专家，竖式请求返回 UNKNOWN。小留出集与高训练 loss/验证差距表明明显过拟合；本轮只验证接口修正，不证明广泛路由泛化。

## 3.1 独立逐位数学专家（v1）

数学专家与 package 路由基座分开训练；路由基座权重保持不变。训练标签由逐列十进制加法程序生成，专家按个位到高位输出每列操作数、进位、和、写入数字与新进位。当前竖式请求用显式意图规则转给专家，尚未训练路由模型选择第三个 `VERTICAL_MATH_EXPERT` 类别。

```powershell
python 08_train_vertical_expert_v1.py --prepare-data --samples 6000 --valid-samples 512 --ood-per-width 64 --out output_vertical_add_v1
python 08_train_vertical_expert_v1.py --train --out output_vertical_add_v1 --device cuda --batch 64 --epochs 12
python 08_train_vertical_expert_v1.py --text "请对 123+45 做竖式计算" --model output_vertical_add_v1/vertical_expert.pt --device cuda
python 08_train_vertical_expert_v1.py --interactive --model output_vertical_add_v1/vertical_expert.pt --device cuda
```

交互入口只加载一次权重。请求需包含 `竖式`、`步骤`、`过程`、`推导`、`解释`、`怎么计算` 或 `如何计算` 之一，才会交给数学专家。

## 3.2 迭代式单列专家（v1）

对照 08 的整段生成方式，这一版训练 8,260 参数 MLP 作为单列专家，输入仅为 `(a位, b位, 进位)`。训练覆盖完整的 200 种离散输入状态；控制器逐次消费两个十进制字符串的最右位，并复用同一个专家预测本位数字和新进位，数字游标耗尽且进位为零时终止。参数量不随整数位宽变化，循环次数随输入宽度增长。

```powershell
python 09_train_iterative_digit_expert_v1.py --train --device cuda --epochs 200
python 10_train_package_router_v2.py --prepare-data --samples-per-class 2400 --out output_router_v3_three_way
python 10_train_package_router_v2.py --train --out output_router_v3_three_way --init-from output_router_v2_three_way/router.pt --device cuda --batch 64 --epochs 10 --lr 0.0001
python 09_train_iterative_digit_expert_v1.py --interactive --model output_iterative_digit_v1/digit_step_expert.pt --router-model output_router_v3_three_way/router.pt --device cuda
```

实测穷举单列状态 200/200 正确；完整 package→循环专家闭环对 3+9、999+1001、1234567+7654321、99999999+1 及 100 位全 9 加 1 均正确，步骤数分别随位数/最终进位变化，模型仍为 8,260 参数。专家状态域已穷举训练并穷举验收，这说明递归组合可跨位宽运行；它不是从有限训练状态外推单列规则，也不是端到端 Transformer 自主发现算法。

随后训练三分类 Transformer 路由器，输出 `CALCULATOR`、`VERTICAL_MATH_EXPERT`、`UNKNOWN`，从 v2 二分类路由器迁移共享层并微调。第二轮训练 7,200 条，留出 52 条，10 epoch；最佳留出准确率 63.5%（CALCULATOR 81.3%、VERTICAL 70.0%、UNKNOWN 37.5%），说明泛化仍有限。新增触发变体后，手测短句 `3+9 竖式计算` 已自动选择逐位专家；裸算式仍选计算器；明确“不要计算”请求返回 UNKNOWN。竖式请求现在由学习路由器选择而非词面规则；低留出分数意味着不能声称路由已普遍解决。

## 3.3 路由标签修订与架构对照（v4）

类别定义：`CALCULATOR`=要求给出计算结果；`VERTICAL_MATH_EXPERT`=要求逐位/竖式步骤；`UNKNOWN`=不属于上述任一专家的请求。UNKNOWN 包含抄写、翻译、文本格式、符号检查等，不限于“算式问题但不要计算”。v3 留出集中 UNKNOWN 被误判为竖式的 7 条，都是未见过的非执行表达；v3 每类 1,200 行来自少数模板反复抽样（大量重复），并未形成 1,200 种不同说法。

新增 `11_train_router_eval_v4.py`，训练集补入 package-first 控制视图、裸 package 默认求值，以及更广的非数学文本任务；36 条盲测来自不同模板，和训练集没有完全重复句。命令：

```powershell
python 11_train_router_eval_v4.py --prepare
python 11_train_router_eval_v4.py --train --device cuda --epochs 12
```

两种方案从同一 v3 权重初始化，在同一固定留出集比较：原三分类 Transformer 为 21/36（58.3%），其中 CALCULATOR 4/10、VERTICAL 5/10、UNKNOWN 12/16，UNKNOWN 误送专家 4 条；两阶段学习方案（先“是否属于两个数学专家”，再选择计算器/竖式）为 23/36（63.9%），分别 3/10、7/10、13/16，UNKNOWN 误送专家 3 条。v3 原权重在这份新盲测上为 15/36（41.7%）。两阶段方案有小幅改善，但整体准确率仍低，直接计算类尤其需要补强，36 条也太少，不能作为稳定泛化结论。逐条输入、标签、两种预测保存在忽略目录 `output_router_v4_reviewed/evaluation.json`；数据在 `output_router_v4_reviewed/data/`。

实际交互视图的六个回归例（裸算式、两种竖式说法、明确拒绝计算及较大整数）在 v3、v4 三分类和两阶段方案中均选择了预期类别。这是小型人工用例检查，不抵消盲测错误。下一轮需扩充独立人工表达、纳入更多真实控制视图，再判断是否选用两阶段路由；当前两者均不应称为已可靠解决。

v1 专家约 0.958M 参数，UTF-8 字节词表；1–4 位训练/验证，5–8 位未见宽度测试。RTX 4070 Laptop 上 6,000 条、12 轮、每轮 94 更新训练约 29 秒；train loss=0.02891、valid loss=0.01596。未见宽度测试每档 24 条，5/6/7/8 位均为 0/24 正确答案；随机 1–4 位验证小测也不稳定。token loss 低不代表学会算法。本次证明的是 package 可把展开后的 payload 交给一个独立训练专家，未证明专家掌握可泛化的逐位算法。数据和权重保存在被忽略的 `output_vertical_add_v1/`。

## 3.4 通用底座种子 v0：首次试跑与问题复盘

首次试跑使用以下脚本生成数据（已保留为历史对照，不作为当前架构数据入口）：

```powershell
python 13_prepare_code_package_v1.py
```

这批 61 条训练样本把用户请求、打包标记、路由结果、解包载荷和回答拼成了一段因果语言模型文本，未真正分开验证路由阶段与专家阶段。`temperature`、`probability` 只在留出集出现，因此当时把“熟悉概念解释”和“未见标识符泛化”混在一起测。基础 tokenizer 原先把 marker 切成 6 个 token；训练副本扩展后为单 ID 151646。

本轮 RTX 4070 Laptop 训练的从零初始化模型共 9.80M 参数，20 轮后 train loss=9.6761、最低 validation loss=9.7247；12 道留出题均没有可用回答。这个结果表明模型没有学会任务；它不能评判分阶段 Package 架构。

## 3.5 前置代码 Package、会话缓存与分阶段数据（v2）

当前数据入口先在模型 tokenizer 之前调用 `code_package_frontend_v1.py`：从围栏/反引号代码或保守识别的 C 声明中提取代码区域，按顶层语句边界拆分子包，并将载荷存在本次请求的 `SessionCache`。自然语言输入中不要求用户写 `<|PKG_CODE|>`。前端生成父包/子包句柄和路由视图；路由视图只含意图与专用包类型 token，代码字符串、子包数量和缓存句柄都不喂给路由模型。路由命中后，控制器用子包句柄从会话缓存取出载荷，交给代码专家解析。

```powershell
python code_package_frontend_v1.py --self-test
python code_package_frontend_v1.py --request '解释这段代码：int count = 3;'
python 16_prepare_code_package_v2.py
```

`16_prepare_code_package_v2.py` 只准备和审计数据，不启动训练。它输出独立的 `router_train/eval` 与 `expert_train/eval` JSONL：路由数据不含源代码载荷；专家样本包含由缓存解包得到的子 `<|PKG_CODE|>` 和对应代码。专家评估集分成训练中见过的变量名（只换值）与未见标识符两组，避免把两种泛化混成一个分数。当前审计通过：专用 token ID=151646，路由载荷泄漏 0；路由训练/评估 14/6 条，代码专家训练/评估 16/6 条，其中 4 条未见标识符。样本量仍只够做接口和初步行为检查，不足以得出能力扩展结论。

前端目前是 C 代码打包原型：支持代码围栏、反引号与简单声明检测；分号切分会保护字符串、括号和 `for` 头部，但不是完整 C/C++ 词法器或 AST。宏、复杂声明、多种语言和长生命周期缓存尚未实现。当前 v2 数据没有训练路由器或代码专家；先审核数据与流水线，再开始独立训练，避免覆盖 v0 运行目录。

## 3.6 v3 扩容数据：程序化生成 ~1000 条专家样本（2026-10-12）

v2 的 16 条/1,064 token 比 v0 失败时的 61 条/2,663 token 还少，直接训练大概率复现失败。`17_prepare_code_package_v3.py` 在保持 v2 分离式格式与前端打包流程不变的前提下程序化扩容：

```powershell
python 17_prepare_code_package_v3.py            # 默认 seed=20261012
python 17_prepare_code_package_v3.py --seed 7   # 换种子生成对照数据
```

- **专家训练 1,000 条 / 68,460 token**（v2 的 64 倍）：240 个组合生成的训练标识符池 × 10 种 C 类型（含 unsigned short/long/char）× 训练值域随机字面量 × 12 种问法模板（每种 2 个答法变体）。四种句式：单声明初始化 640、只声明 120、声明+赋值（两子包）120、双声明（两子包）120。
- **专家评估每组 24 条**（对齐 08/v5-C1 每档 24 条判据）：seen_name_new_value 取训练见过的名字 + 独立评估值域的新字面量；unseen_identifier 用与训练池完全不相交的 24 个名字（前 4 个沿用 v0/v2 留出集便于跨轮对比）。
- **路由 train/eval 每类 100/24 条**：训练意图由动词×对象×侧面组合生成，评估意图手写；审计断言 train/eval 零重叠、载荷泄漏 0、未见标识符子串级零泄漏、同种子复生成逐字节一致。
- 训练行额外带 `text` 字段（专家=prompt+target，路由=router_input+路由选择），可直接喂 03 训练器（`--package-code-token`）；但注意当前 03 对整条 text 计 loss，未做"只对 target 计 loss"的掩码，分阶段设计要完全落地仍需改训练器。
- 输出 `data/code_package_v3_*.jsonl` 与 `code_package_v3_audit.json`（含各文件 sha256）。

## 3.7 中英双语词条数据 v1：「猜/guess」格式样板（2026-10-13）

`18_prepare_bilingual_v1.py` 把中英词条独立成一个数据集（与代码 Package 分开生成、分开审计），当前只收录用户给定的一个词条「猜/guess」，作为后续词条的**格式样板**：

```powershell
python 18_prepare_bilingual_v1.py
```

- 一个词条展开成 **12 种样本类型 / 14 条样本 / 662 token**：双语术语知识陈述、中→英、英→中、中文释义、英文释义、例句对照（中→英）、例句多译法、例句反向（英→中）、词性说明、用法辨析（估计/estimate、推断/infer、计算/calculate 三条）、场景应用、判断问答。
- 释义与例句逐字采用用户给定内容（猜 = 只凭经验和直觉，没有事实依据给出的一个结果；例句「我猜的」），英文释义是对应翻译，不自行发明词义。
- 新增词条只需往脚本的 `TERM_ENTRIES` 追加同结构条目（zh/en/zh_def/en_def/pos/example_zh/example_en/contrast/scene），样本类型自动复用。
- 无随机数，天然可复现；审计检查 text 非空、含中或英词条、text 唯一、12 种类型齐全、重建逐字节一致，并记录文件 sha256。
- 数据脚本用纯 `tokenizers` 库读 tokenizer.json 统计 token，不导入 torch/transformers（避免小页面文件机器上 CUDA DLL 加载失败 WinError 1455）。
- 输出 `data/bilingual_v1_train.jsonl` 与 `bilingual_v1_audit.json`；训练行带 `text` 字段，可直接喂 03 训练器（不需要 `--package-code-token`）。

## 3.8 步骤顺序语义数据 v1：先/再/然后/最后 与 第一/第二/最后（2026-10-14）

`19_prepare_step_order_v1.py` 生成"步骤顺序"语义训练数据，内容全部围绕编程/软件开发主题（用户明确要求，不用泡茶等日常流程）：

```powershell
python 19_prepare_step_order_v1.py            # 默认 seed=20261013
```

- **两套顺序标记体系**：副词式（先X，再/然后/接着/随后Y，最后Z）与序数式（第一，X；第二，Y；最后，Z），同一组步骤两种体系各渲染一次，并互相转换（convert_marker），让模型学到两套标记表达同一个顺序结构。
- **1,466 条 / 68,629 token / 10 种样本类型**：完整叙述 152、首步/末步问答各 152、之后/之前问答各 239、位置问答 152、步数问答 76、乱序重排 76、标记互转 152、续写下一步 76。
- **内容来源**：16 个手写真实开发流程（编译运行 C 程序、修 bug、提交代码、部署网页、训练小模型、写单元测试、排查崩溃、代码审查、配置环境、处理数据、调试循环、发布版本、合并分支、分析内存泄漏、优化慢代码）+ 60 条编程动作池随机组合序列（3-5 步）。
- **答案全部从步骤列表程序化派生**：问答/重排/续写的答案由同一个 steps 列表推导，审计逐条核对 qa_first/qa_last/qa_count/qa_next 与步骤列表一致；抽象组合序列的问答带叙述上下文前缀（自包含且避免跨序列问句重复）。
- 续写样本的前缀用专门的 render_prefix（只含"先/再/然后"，绝不出现"最后"），避免"前缀宣称结束、续写又继续"的自相矛盾；中间标记不重复采样，避免"然后…然后"。
- 与 18 号一致：纯 `tokenizers` 统计 token（绕开 torch DLL 的 WinError 1455）、固定种子复现（审计含重建逐字节一致与文件 sha256）、训练行带 `text` 字段可直接喂 03 训练器。
- 输出 `data/step_order_v1_train.jsonl` 与 `step_order_v1_audit.json`。

## 3.9 C/C++ 声明双向数据 v1：代码⇄中文注释 + 分号语义（2026-10-14）

`20_prepare_decl_bilingual_v1.py` 生成声明级的双向语义数据，配套扩展了 [`code_package_frontend_v1.py`](code_package_frontend_v1.py)（新增 `PKG_NUM`/`PKG_NAME` 嵌套原子包与 `package_description` 反向打包器）：

```powershell
python 20_prepare_decl_bilingual_v1.py     # 默认 seed=20261014
```

- **两个方向共用一份结构化声明**：`Decl` 从同一结构派生代码与中文描述，保证双向一致。
  - 正向（代码→注释，420 条）：`const unsigned int MAX = 45872U;` 打成 `<|PKG_CODE|>` 语句子包，语句内的数值与变量名再切成 `<|PKG_NUM|>`/`<|PKG_NAME|>` 原子包，答案=中文注释+形态说明。
  - 反向（注释→代码，420 条）：中文描述里的名字与数值同样打成原子包，路由器只看到包结构、看不到载荷，专家解包后写出声明。
  - 分号语义（80 条）：`int a = 1, b = 2;`（一个分号=一句、两个变量）与 `int a = 1; int b = 2;`（两个分号=两句）对比 + 分号知识问答，答案里的句数由 `code.count(";")` 程序化派生。
- **规模**：920 条 / 76,442 token（均 83.1、最长 125）；7 种 C 类型 × 4 种形态（变量带值/变量无值/const/static const）各 **420/420 条，即每个（类型×形态）组合精确 15 条**；评估 60 条（正向 24 / 反向 24 / 分号 12）。
- **隔离**：训练值域与评估值域分离；评估只用 20 个未见名字（10 大写常量名 + 10 小写变量名），审计做子串级零泄漏检查；反向 `marked_view` 断言不含真实载荷。
- **边界自检（target 掩码的前提）**：审计与训练器都会断言 `encode(text) == encode(prompt) + encode(target)`。拼接处一旦被 BPE 合并，监督位置就整体错位、loss 会静默算错；这种错不会报错，只会让训练白跑，所以必须显式检查。
- **训练入口**：必须同时加 `--package-code-token --package-nested-tokens`，否则三个标记各是 6 个 token（实测注册后为 151646/151647/151648，词表 151649）；配合 `--target-mask` 只对 target 计 loss。

### 3.9.1 target-only loss 掩码（`--target-mask`，2026-10-19）

`03_train_pretrain.py` 新增 `--target-mask`：只对 `target` 计 loss，prompt 与 eos 前缀位置一律 `-100`。

```powershell
python 03_train_pretrain.py --data data/decl_bilingual_v1_decl_train.jsonl --out output_decl_v1 `
  --hidden 384 --layers 4 --heads 8 --no-pack --batch 16 --bucket --epochs 30 `
  --loss-chunk 64 --verify-loss --target-mask `
  --package-code-token --package-nested-tokens --device cuda --dtype auto --seed 42
```

- 序列构造：`[eos] + prompt_ids + target_ids + [eos]`，labels 前 `1+len(prompt_ids)` 个位置置 `-100`。
  最后一个 prompt 位置的 logits 仍要预测第一个 target token，所以掩码正好到此为止。
- **必须配 `--no-pack`**：packing 会把多条样本拼成 512 块、切散样本边界，掩码无法稳定对齐；脚本会直接报错拒绝。
- 不开启时行为完全不变（labels = input_ids），数学 v5、双语、步骤顺序等只有 `text` 字段的数据照旧。
- 实测（19.8M 冒烟模型，920 条）：监督 token **16,501 / 54,923 = 30.0%**；`--verify-loss` 对拍 HF 原生 `11.897043` vs 分块 `11.897044`（相对差 0.0000%），说明掩码与分块归约都对。

### 3.9.2 声明数据评估脚本（`04f_eval_decl_v1.py`，2026-10-19）

```powershell
python 04f_eval_decl_v1.py --model output_decl_v1/final_model --device cuda
python 04f_eval_decl_v1.py --model output_decl_v1/final_model --limit 6 --max-new-tokens 32   # 环境自检
```

- 按方向分三组打分，每组同时给**严格**（整句/整行逐字一致）与**部分**（关键字段正确）两档：
  - 代码→注释：部分 = 答案里同时有变量名与初始值（无初始值时要求"没有给出初始值"）。
  - 注释→代码：部分 = 首行以分号结尾且含正确变量名、正确初始值。
  - 分号语义：从输出抽出"是 N 句"，N 必须等于代码分号数。
- 加载时会断言三个 Package 标记仍是单 token（151646/151647/151648），否则直接报错——词表不对齐时评估结果没有意义。
- `--out` 可把逐条 prompt/期望/实际/判分写入 JSON，便于归档到总册。
- CPU 冒烟实测（19.8M / 2 epoch，纯链路验证）：60 条全 0%，但输出已从乱码收敛到"声明一个…初始值是…"句骨架，说明数据格式、掩码、解码链路一致；这个分数**不代表真实模型能力**。
- 输出 `data/decl_bilingual_v1_decl_train.jsonl`、`decl_bilingual_v1_decl_eval.jsonl` 与 `decl_bilingual_v1_audit.json`。

### 3.9.3 target 也必须打包：复制墙的真正解法（2026-10-21）

**这是本仓库最重要的格式修正之一。** 初版 20 号脚本只把 prompt 打包，target 仍是明文：

```
prompt: 原子包：<|PKG_NAME|> → FARADAY；<|PKG_NUM|> → 286020U
target(旧): 声明一个无符号整型常量 FARADAY，初始值是 286020U。   ← 明文
target(新): 声明一个无符号整型常量 <|PKG_NAME|>，初始值是 <|PKG_NUM|>。   ← 打包
```

旧格式逼模型**逐字符复制** `286020U`（7 字符）与标识符，65M 小模型做不到，于是
值被截断（`5005931L`→`50053L`）、名字从训练分布抽（`energy`→`pos`）。这是
v5「定宽复制墙」的同一病症，**不是容量问题，也不是数据正确性问题**。

改法是让 target 只含静态标记：模型学"在句子哪个位置放哪个标记"，程序解包后把
标记替换回载荷得到明文，**零字符复制**。反向（注释→代码）同理，target 为
`const unsigned int <|PKG_NAME|> = <|PKG_NUM|>;`。

同一数据、同一基座、同样超参，只改 target 打包与否：

| 方向 | 打包前 `output_decl_expert_v1` | 打包后 `output_decl_packed_v1` |
|---|---|---|
| 代码→中文注释 | 0/24 | **24/24 严格** |
| 中文注释→代码 | 0/24 | **22/24 严格，24/24 部分** |
| 分号语义 | 12/12 | 12/12 |
| 合计 | **12/60 = 20%** | **58/60 严格 = 96.7%，60/60 部分 = 100%** |

配套的三处工程约束：

1. **数据审计**：每条 target 都断言"不含明文 name/lit（按标识符边界判断，避免
   单字母名 `i` 误命中 `int`）"+"标记解包后与明文期望逐字相同"。
2. **评估必须先解包**：`04f` 拿到模型输出后先做 `replace(PKG_NAME, name)`、
   `replace(PKG_NUM, lit)`，再与 `expected_unpacked` 比对。
3. **`skip_special_tokens` 必须为 False**：`<|PKG_NAME|>`/`<|PKG_NUM|>` 是
   special token，跳过它们会把标记直接从解码结果里删掉，造成"模型学会了但判 0"
   的假失败。只清 eos/pad，标记留到解包那一步。

数据里额外存 `expected_unpacked`（明文期望）与 `answer_plain`（人类可读参考），
训练只用 `target`。剩余 2 条严格失败是类型词判错（`int`→`double`），标记位置
全对，属分类错误，扩类型×形态覆盖即可。

## 3.10 指示代词/包引用探针（v8 定稿，2026-10-23）

四个脚本：`24_prepare_probe_v8.py` + `25_eval_probe_v8.py` 是**当前结论**的来源；
`22_pronoun_probe_v1.py` + `23_eval_pronoun_probe_v1.py` 是 v7，保留用于复现旧结论。

### 3.10.0 为什么推翻了 v1~v6 的设计

v1~v6 三条轴（距离 / 干扰项 / 类型混淆）都用**序数代词**（「第几个和它后面那个」），
要求模型在 N 个同质 key 里**数到第 N 项**再取两个。由此得到「包表超过 12 项就挑不准」
这个结论——**但这是无效需求**：

> 协议里模型从来不需要「数第几个包」。它只需知道 `<|PKG_NUM|,#A>` 是一个数字包；
> 值本身在缓存里，由程序按 (类型, 序号) 解包后交给专家（token filter）。
> 具体值是多少、包排在第几，模型都不需要知道。

所以旧的「包表 12 项上限」测的是自造的题目，不是协议的能力边界。**已作废。**

### 3.10.1 包必须是原子整体：`<|PKG_NUM|,#B>`

用户原话：

> 把包做成一个整体，不要再让大模型理解意思了，就整体是个包就可以了，
> 并且知道包和包 key 不同就是不同的意思就够了，拆包的时候能根据类型和值还原回去，
> 给专家模型处理。

一个包 = 一个不可分的整体 key，类型与序号在**同一个尖括号内**：

```
正确   <|PKG_NUM|,#B>          注册后 1 token
错误   <|PKG_NUM|>#B           3 token，序号游离在外
错误   <|PKG_NUM|>#B 数值(#B)  8 token，编号重复两次
```

三种形态实测（22 项包表总 token）：**258 / 200 / 79**。原子 key 形态让包表 token 数降 3.3 倍。

注册后 ID 唯一、`id_to_token` 可逐字还原；6 类型 × 26 序号 = 156 个 key，
`tokens.json` 的 `_package_keys` 定 `id_base=151660`（14 个协议标记占 151646–151659）。

配套改动：
- `03_train_pretrain.py` 新增 `--package-keys`：注册 156 个 key，断言全部单 token
  且 ID 与 `tokens.json` 一致（跨医生交接的前提）
- 22 号审计新增「包表条目必须恰好是一个 key」，防止把 `数值(#X)` 加回去
- 23 号 `KEY_RE` 整体抽取，`score()` 按 key 整体比较

### 3.10.2 重设计后的三条轴（都不含序数）

| 轴 | 题目 | 测什么 |
|---|---|---|
| `type_id` | 给一个 key，说出它的类型（答案 1 token） | 模型是否真懂 key 的类型前缀 |
| `semantic_pick` | 包表里**只有一个 NUM**，其余是其他类型；按「那个数值」选出它 | 按类型定位，**不数位置** |
| `distance` | 同上 + 无关填充到 800 token | 投影能写多长 |

审计硬断言两条：`num_key_unique`（NUM 包必须唯一，否则题目有歧义）、
`no_ordinal_in_pronoun`（代词里不许出现序数，否则又退化成数位置）。

### 3.10.3 v7 的结果与遗留问题（2026-10-22，`output_pronoun_probe_v7`）

| 轴 | 同分布 | 留出集（102 条） |
|---|---|---|
| `semantic_pick` 干扰 2/5/10 | **1.00** | **1.00** |
| `semantic_pick` 干扰 20（21 项包表） | 0.97 | 0.86 |
| `distance` 10→800 六档 | **1.00** | **1.00** |
| `type_id` 六类型 | 0.88~1.00 | **0.25~1.00** |

语义指代与投影长度两条当时就定了。遗留的是 `type_id`：同分布看着像学会了，
留出集掉到 0.25~1.00，**错误几乎全部把答案吐成 `<|PKG_TASK|>`**。

当时的诊断是「配比坍缩」（type_id 只占 192/512 条、答案是单 token 开集生成）——
但那只是**推断**。v7 的留出集还有两个工程缺陷，使这个推断无法验证：

1. 留出集用 03 的 `--dump-split`（`random.shuffle` 后前 20%），每格只剩 4~10 条；
2. **无法保证已注册的 key 在训练部分都出现过**。未出现的 key 拿着随机 embedding
   去评估，会直接制造无法归因的错。

### 3.10.4 v8：把「类型标记是否被理解」测成可归因的数（2026-10-23）

`24_prepare_probe_v8.py` + `25_eval_probe_v8.py`。v7 的 22/23 号保留不动，用于复现旧结论。

三处改动，都是为了消除混杂变量：

**① `type_id` 拆成四档，配比完全相同，唯一变量是「有没有给候选」**

| 档 | 形态 | 测什么 |
|---|---|---|
| `type_id_choice_k2` | `[候选]` 给 2 个类型标记，选一个 | 候选枚举够不够 |
| `type_id_choice_k6` | `[候选]` 给全部 6 个，选一个 | 候选变多是否退化 |
| `type_id_open` | 不给候选，直接说类型（= v7 形态） | 开集生成是不是瓶颈 |
| `type_id_unseen` | 不给候选，且该 key **从未进训练集** | 负对照：零样本是多少 |

四档每种类型都是同样条数，所以 `k2/k6/open` 之间只差那一行 `[候选]`。

**② 留出集由 24 号分层直出，不靠 03 随机切**

`data/pronoun_probe_v8_train.jsonl`（672 条）/ `data/pronoun_probe_v8_holdout.jsonl`（228 条）。
审计硬断言：**除 6 个刻意留白的 `#Z` key 外，150 个已训练 key 在训练文件里每个至少出现 9 次**；
`#Z` 一次都不许出现。训练集里重合的随机 val 只能用 03 的 `--dump-split` 事后核对。

**③ 负对照必须整档只进留出集**

`type_id_unseen` 的行若混进训练文件，负对照就退化成普通泛化。24 号用
`HOLDOUT_ONLY_AXES` 在造数据层就隔开，审计再断言 `#Z` 零泄漏。

### 3.10.5 v8 结果（`output_pronoun_probe_v8`，672 条，30 epoch 334 s）

训练从零：hidden 384 / 4 层 / 8 头，`--protocol-tokens --package-keys --target-mask --no-pack`，
batch 4 / loss-chunk 32（GPU 被 5 个 Unity.exe 占 3892 MiB，故降档），
train_loss 0.0093 / eval_loss 0.0266。

**留出集 228 条**（每格 6 条，`unseen` 每格 10 条）：

| 轴 | 正确 / 总数 | 严格分 |
|---|---|---|
| `type_id_choice_k2` | 35 / 36 | **0.97** |
| `type_id_choice_k6` | 35 / 36 | **0.97** |
| `type_id_open` | 34 / 36 | **0.94** |
| `type_id_unseen` | **0 / 60** | **0.00** |
| `semantic_pick`（干扰 2/5/10/20） | 23 / 24 | 0.96 |
| `distance`（10→800 六档） | 36 / 36 | **1.00** |

同分布 672 条上 `type_id_open` 为 1.00，留出 0.94——两者差距从 v7 的 0.25~1.00 收敛到 0.94，
说明配比不足确实是 v7 的主因。

### 3.10.6 四条结论

**1. 类型标记确实被理解，v7 的低分是配比不足。**

k2 0.97、k6 0.97、开集 0.94 **都达标**。v7 里 type_id 只占 192/512 条且答案是开集单 token，
样本量不够导致坍缩到 `<|PKG_TASK|>`。加量即可解决，机制本身没问题。

**这也顺带说明：候选枚举不是必需的。** k2/k6 与 open 三档只差 4 个百分点（0.97/0.97/0.94），
在每格 6 条的样本量下没有统计意义。协议里可以要求开集输出类型标记。

**2. 语义指代可用，靠类型前缀而非数位置。**

包表里唯一的 NUM，用「那个数值」就能选出，跨档 0.96。干扰项**全是异类型**，
所以定位靠的是 `<|PKG_NUM|,#X>` 的类型前缀，不是数到第几项。

**3. 投影可写到 800 token 以上。**

10→800 六档严格分全 1.00，未探到跌破点。已测上限约 815 token。

**4. 未训练 key 的类型知识为零，而且是「确定性错误」不是「瞎猜」。**

这条是 v8 最重要的产物。`type_id_unseen` 得 0/60，**显著低于随机基线 1/6 ≈ 0.17**。
看逐条输出，每种期望类型都固定映射到同一个错类，60 条一条不差：

| 期望 | 实际输出 | 条数 |
|---|---|---|
| `<|PKG_NUM|>` | `<|PKG_ARITH|>` | 10/10 |
| `<|PKG_NAME|>` | `<|PKG_ARITH|>` | 10/10 |
| `<|PKG_TEXT|>` | `<|PKG_NAME|>` | 10/10 |
| `<|PKG_CODE|>` | `<|PKG_TEXT|>` | 10/10 |
| `<|PKG_ARITH|>` | `<|PKG_TEXT|>` | 10/10 |
| `<|PKG_TASK|>` | `<|PKG_NAME|>` | 10/10 |

原因是结构性的：`<|PKG_NUM|,#Z>` 注册后是**单个不可分 token**，它的拼写里没有 `NUM`，
与类型标记 `<|PKG_NUM|>` 之间**没有任何词形联系**。所以没进过训练集的 key 无可推广结构。

**这是「包做成原子整体」的固有代价，不是 bug。** 但它对协议有两条硬约束：

- 包表必须**先声明、再训练**；
- 给医生加新 key **必须补训练**，不能指望零样本推广。

比「瞎猜」更危险的是它**不会表示不知道**：未声明的 key 会稳定输出一个错误的类型标记。
所以「key 是否已注册」必须由**程序**校验，不能指望模型自己发现。

### 3.10.7 必须保留的边界

- v8 的 `type_id_choice_k2` / `k6` / `open` 每格留出只有 6 条，1 条错 = 16 个百分点。
  三档差距（0.97 / 0.97 / 0.94）在 9 条错的总量下**不足以断言候选枚举有无必要**；
  要收窄需把每格提到 20 条以上。
- `type_id_unseen` 的 0/60 结论很强（全确定性），但它是**结构性必然**，
  不依赖训练量。加数据不会改善它——这正是要点。
- 反事实探针（旧 B 组）已删除：它的代词措辞属分布外输入，本来就不作判据。
- v7 的 `semantic_pick/20` 只剩 7 条留出（0.86 = 错 1 条）；v8 每格 6 条，仍偏少。
  「21 项包表」的准确上限**尚未定位**，只能说 0.96 上下。

### 3.10.8 距离轴为什么收到 800（保留的显存实测）

| 序列长度 | 峰值显存 | 占卡 | 单步耗时 | 相对 414 |
|---|---|---|---|---|
| 414 token | 3106 MiB | 37.9% | 0.19 s | 1.00x |
| 824 token | 5170 MiB | 63.1% | 0.33 s | 1.77x |
| 1624 token | **9440 MiB** | **115.3%** | **6.22 s** | **33.5x** |

1624 token 超出 8188 MiB 达 1252 MiB。Windows 的 WDDM 此时**不报 OOM，而是静默换页到
系统内存**：耗时涨到 6.22 s（纯 attention O(n²) 只应 15.4x，多出的 ~2x 是换页），
该路径在 BF16 下产生非有限值，`clip_grad_norm_` 又把 nan 扩散到全部 50 个张量，
checkpoint 全废。加上 `MAX_SEQ_LEN` 原本 512、1624 等于 RoPE 外推 3.2 倍，
所以距离轴收到 800，`--max-seq-len` 设 1024。

## 3.11 包引用探针 v10：v9b 失败归因与修复（2026-10-26）

`31_prepare_probe_v10.py`（数据）+ `30_eval_probe_v9.py`（评估，v9/v10 共用）。
v9 的 29 号脚本与 `output_probe_v9`/`output_probe_v9b` 保留作对照。

### 3.11.1 v9b 的失败与根因

v9b（`output_probe_v9b`，1632 条训练，40 epoch）留出集：type_id 四档全 1.00，
但 **semantic_pick 只有 0.46**，distance 从 10 token 档就跌破 0.90。错例全部是
「类型对、编号差 ±1~2」，且输出的错编号**恰好是包表里的干扰项**。

根因在 29 号生成器的干扰编号分配：`cur = (cur + 1) % 400` 连续取，包表永远是
`#297 #298 #299` 这种**相邻块**（holdout 实测确认）。连续块给了记忆捷径：模型
记住「这组指纹 → 答案是块尾」即可拿分，不必学「读 `<|PKG_NUM|>` 后面的编号」
这条 induction 规则。证据链：

1. train_loss 0.0206 / eval_loss 0.3765，**18 倍 gap = 纯记忆**；
2. 留出集上 induction 不精确，错编号全是包表内干扰项（±1~2）；
3. 每个编号作为 target 平均只被监督 ~3 次，401 选 1 的绑定学不牢。

另注：v9b 评估报告的 verdict 把 `type_id_unseen` 的 1.00 写成「编号整体编码的
固有代价」是**自相矛盾的**——分数是 1.00 不是 0.00。v8 的 0/60 负对照结论建立在
per-key 形态（类型与编号在同一个 token 里）上；v9 起类型标记与编号分离，类型
辨识只需读编号前面的类型标记，与编号 embedding 是否训练过无关。v8 的
「未训练 key 零样本为零」结论**不适用于 v9/v10 形态**，引用时必须区分形态。

### 3.11.2 v10 的三处修复（唯一变量是数据）

| 修复 | v9b | v10 |
|---|---|---|
| 干扰编号采样 | `cur+1` 连续块 | 等距网格（随机相位，步长 16），包表内任意两编号差 ≥16 |
| target 编号分配 | LCG 打散，每编号 ~3 次且不保证 | train/holdout 独立轮转计数器，每编号恰好 6~7 次作为答案被监督 |
| 每格训练量 | 150 条 | 300 条（semantic_pick/distance） |

新增两条审计硬断言：`id_gap_ok`（包表内任意两编号差 ≥ MIN_ID_GAP=16，逐条断言）、
`target_id_coverage`（每个已训练编号作为 target 至少 MIN_TARGET_OCC=6 次）。
数据 3132 训练 / 468 留出，seed=20261026，重建逐字节一致。

### 3.11.3 v10 结果（`output_probe_v10`，30 epoch 2964 s）

训练信号先给出归因证据：train_loss 0.0235 / eval_loss 0.0783，**gap 从 18 倍缩到
3.3 倍**——捷径切断后模型学的是可迁移规则而非样本指纹。

留出集 468 条**全部 1.00**：

| 轴 | v9b | v10 |
|---|---|---|
| type_id_choice_k2 / k6 / open | 1.00 | 1.00 |
| type_id_unseen（#400 零训练） | 1.00 | 1.00 |
| semantic_pick（干扰 2/5/10/20） | **0.46** | **1.00** |
| distance（10→800 六档） | 10 档即跌破 0.90 | **1.00** |

### 3.11.4 结论

1. **v9b 的 0.46 是数据捷径造成的假失败，不是机制或容量问题。** 同一模型配置
   （65.48M / hidden 384 / 4 层）、同一评估器，只改数据生成方式即全对。
2. **「类型标记 + 整体编号 token」的引用形态完全成立**：语义指代（按类型前缀
   定位唯一的 NUM 并输出其编号）在 21 项包表、800 token 投影下都满分。
3. **写探针数据必须断言「干扰项与答案在要被区分的维度上不相邻」**。v9b 的
   连续编号块让「读编号」退化成「记指纹」，这类捷径从 loss 上看是 18 倍的
   train/eval gap，从分数上看是留出集崩盘。审计里 `id_gap_ok` 这条断言
   应成为后续所有探针数据的标配。
4. **监督密度要按「答案空间」算**：编号是 401 选 1，每个编号作为答案至少要
   被监督 6 次才稳（v9b 平均 3 次不够）。「出现在包表里」≠「作为答案被监督」，
   两者要分开审计。

### 3.11.5 v10 现场测试：5 组分布外用例（`32_live_test_v10.py`，2026-10-27）

留出集全对只证明同分布泛化；32 号脚本探训练分布之外：

| 测试 | 内容 | 结果 |
|---|---|---|
| T1 未见问法 | 换代词（数值包/数字包）、换句式（"数值包是哪一个？请交给输出"） | **10/10 = 1.00** |
| T2 跨类型指代 | 问「那个名字/那段代码/那段文本/那个算式/那个任务」（训练只在 NUM 上监督过选包） | **0/30 = 0.00** |
| T3 包表容量 | 26/31 项包表（训练最多 21 项） | **12/12 = 1.00** |
| T4 投影长度 | 1000 token（训练最长 813，RoPE 外推 1.23x） | **8/8 = 1.00** |
| T5 同表多类型切换 | 一张表 NUM/NAME/CODE 各一，同一张表问三种类型词 | 4/12 = 0.33（NUM 全对，NAME/CODE 全错，同 T2 根因） |

**T1/T3/T4 证明机制真成立**：语义绑定不是背模板（换问法全对）、容量上限远未到
（31 项仍满分）、投影可写到 1000 token。**T2/T5 暴露的是训练覆盖缺口，不是机制
失败**：v10 的 semantic_pick 只在 NUM 上监督过「按类型词选包」，「那个名字」等
代词从未与选包任务组合；type_id 轴学的是「读类型标记」（给引用说类型），不能
替代「按类型词选引用」（两个任务的监督信号不同）。错例形态：输出
`<|PKG_NUM|>#<干扰编号>`——模型知道要选包，但类型词→类型标记的映射只学了 NUM。

### 3.11.6 v11 系列：跨类型修复的三轮迭代（`33_prepare_probe_v11.py`）

修复方向：semantic_pick / distance 的目标类型从只有 NUM 扩到全部 6 类型。
三轮结果（同配置 65.48M / 30 epoch / seed 42）：

| 版本 | 每格 | 每 (类型,编号) 组合监督 | 留出 semantic_pick | 留出 distance |
|---|---|---|---|---|
| v11 初版 | 300 | ~1 次 + **同步绑定 bug** | 类型 1.00 / 编号 0.00~0.20 | 同左 |
| v11b 解耦 | 300 | ~1 次 | 类型 1.00 / 编号 0.03~0.23 | 同左 |
| v11c 加量 | 600 | ~2.4 次 | **0.56** | 0.33~0.60 |
| v11d 数据已备 | 1200 | ~5 次 | （未训练，数据在 data/probe_v11_train.jsonl） | |

三轮迭代揭示的两个新教训：

1. **多维度答案空间下，监督密度按「组合数」算**。v10 每编号 6~7 次监督全部
   集中在 NUM，编号绑定学得牢；v11 同样 6~7 次摊到 6 个类型，每个 (类型,编号)
   组合只剩 ~1 次，401 选 1 的绑定学不会。答案空间从 401 变成 6×401=2406，
   监督总量要按比例扩。
2. **两个轮转计数器同步递增 = 隐性绑定**。v11 初版 `goal=PKG_TYPES[i%6]` 与
   `tid=train_ref%400` 同步走，模型学到「类型→编号 mod 6」的确定性映射，
   类型全对、编号全错。多维度采样必须让各维度来自**独立随机源**。
   （v11b 证明解耦本身不够——监督密度才是主因，但绑定 bug 会掩盖这一点。）

v11c 的 0.56（每组合 2.4 次）相对 v11b 的 0.14（1 次）已证明监督密度是主因；
v11d 数据（每组合 ~5 次）已构建并通过全部审计，训练命令同 v11c，输出目录
`output_probe_v11d`，待跑。

## 4. 两台笔记本的正式实验入口

先生成同一份数据（本地已有 tokenizer，无需重复下载）：

```bash
python 01e_make_math_v5.py --total 100000 --seed 42
```

当前实际生成 **59,078 条**，--total 是截取上限，不会自动扩量。与历史 v5-C1 94,942 条不同；原始数据可从 Git 历史查找，当前命令不承诺精确复现旧实验。

4060 Laptop 和 4070 Laptop 可以使用相同命令，下面先以 batch=8 保守起步；用不同输出目录记录每次运行：

```bash
python 03_train_pretrain.py --data data/math_v5.jsonl --out output_math_v5_run01 --hidden 384 --layers 4 --heads 8 --no-pack --batch 8 --bucket --epochs 1 --loss-chunk 64 --verify-loss --no-checkpoints --nave --device cuda --dtype auto --seed 42
python 04e_eval_v5.py --model output_math_v5_run01/final_model --nave --device cuda --n 10
```

另一台电脑可命名 output_math_v5_run02。默认自动精度：CUDA 支持 BF16 时用 BF16，其余用 FP32；可显式 `--dtype float32`。显存不足先把 batch 降至 4/2，再尝试 loss-chunk=32/16。batch=16 是历史使用过的配置，不保证适合所有样本长度与后台显存占用。FP32 内存开销更高。

不要静默换参数后直接比较 GPU 速度：对比实验保持数据哈希、代码、层数、种子、batch、精度和训练步数一致，记录电源模式与其他 GPU 占用。相同种子不能保证跨设备/版本逐位一致。

普通对照组同时去掉训练和评估的 --nave，并更换输出目录；05_interact.py 只支持普通模型。CPU 正式训练很慢，优先用 smoke_test.py 演示完整过程。

## 5. 每次运行自动留痕

每个 --out 目录包含：

- run.json：设备/版本/精度、参数、数据及 tokenizer 哈希、脚本哈希、Git 提交与工作区状态、每轮 loss、开始/完成时间。
- train.log：本次训练日志，不再与其他运行混写。
- final_model/：模型、tokenizer，以及 NAVE 模型的 nave.pt。

已有 run.json 或 final_model 的目录会拒绝覆盖。status=started 只表示已开始；没有 completed 时应检查中断或失败。Git 工作区有改动时，提交号不能完整代表代码，脚本哈希用于辅助核对；要严格复现还应保存对应源码版本。

评估另存文本（在已激活环境中）：

```bash
python 04e_eval_v5.py --model output_math_v5_run01/final_model --nave --device cuda --n 10 > output_math_v5_run01/eval.txt
python -m pip freeze > output_math_v5_run01/requirements.txt
```

两个 output 目录默认不入 Git。将需要公开的指标、环境、失败样例和结论整理进总册；权重需自行备份/传输，不要以为换电脑 git pull 后就能续训。

## 6. 视频记录模板

每一期在总册增加一条实验记录即可，避免再维护多套分散笔记：

1. 这次要验证什么，事先约定成功判据。
2. 展示电脑/GPU、环境自检与 Git 版本；介绍数据样例、条数、哈希和挖洞协议。
3. 展示完整训练命令和参数理由，记录 loss、时间、显存；训练是否收敛与能力是否泛化分别判断。
4. 展示训练内、挖洞、外推结果和失败样例，保留分母与评估命令。
5. 总结实测事实、待验证解释、下一轮只改什么；标注是否可按当前代码复现。

## 验证范围与文件

本机 RTX 4070 Laptop / Windows 上验证 CUDA BF16 与 CPU FP32。CPU 测试使用同一 CUDA wheel 的 CPU 执行路径，不能代替 CPU-only wheel 安装测试。4060 Laptop、Linux、macOS 需在对应机器执行自检。

01e/02/03/04e/05 分别负责数据、tokenizer、训练、评估、交互；nave.py/readnum.py 为模型与数据模块，runtime.py 统一设备选择，setup_env.py/setup_env.ps1 安装环境，check_env.py/smoke_test.py 检查运行链路。环境与运行产物不入库，tokenizer 保留一份。

新研究设想（结构化打包、按需展开与直接执行）见 [实验总册的研究方向记录](../大模型预训练学习与实验架构方案.html#structured-computation)。该方案尚未实现或验证。
