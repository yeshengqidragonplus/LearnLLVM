# 预训练实验室：安装、复现与记录

所有训练、生成和评估命令在本目录运行。两台电脑使用同一份代码，各自创建虚拟环境；不要复制 .venv。项目入口见 [根目录 README](../README.md)，实验经验集中在 [总册](../大模型预训练学习与实验架构方案.html)。

## 前言课与三课 HTML 教材

- [前言课：语言、意义与结构](courses/course-00-preface.html)
- [第 1 课：稠密 Transformer、GPT 与预训练全流程](courses/course-01-dense-decoder.html)
- [第 1 课延伸 Demo：注意力、上下文与 RNN](courses/course-01b-attention-context-rnn.html)
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
