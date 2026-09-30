# 预训练实验室：安装、复现与记录

所有训练、生成和评估命令在本目录运行。两台电脑使用同一份代码，各自创建虚拟环境；不要复制 .venv。项目入口见 [根目录 README](../README.md)，实验经验集中在 [总册](../大模型预训练学习与实验架构方案.html)。

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

## 3. 两台笔记本的正式实验入口

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

## 4. 每次运行自动留痕

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

## 5. 视频记录模板

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
