# -*- coding: utf-8 -*-
"""生成步骤顺序语义预训练数据 v1：先/再/然后/最后 与 第一/第二/最后 两套标记。

目标：让模型学会"步骤顺序"这一语义结构，而不只是背固定句子。做法是把
同一个步骤列表渲染成多种样本类型，所有问答答案都从步骤列表程序化派生，
保证问题与答案在语义上必然一致：

- sequence_narrative：完整步骤叙述（两套标记体系各渲染一次）
- qa_first / qa_last：最先/最后一步做什么
- qa_next / qa_prev：某步之后/之前是什么
- qa_position："X"是第几步（序数体系）
- qa_count：一共有几步
- reorder：打乱顺序的步骤重新排好
- convert_marker：两套标记体系互相转换
- continue_seq：给出前几步，续写下一步（含正确的顺序标记）

内容来源（全部围绕编程/软件开发主题）：
1. PROCEDURES：手写的真实开发流程（编译、修 bug、提交、部署等，步骤有真实先后逻辑）；
2. GENERIC_ACTIONS：编程动作池随机组合（专门练"顺序结构"本身）。

与 18 号一致：无 transformers/torch 依赖（纯 tokenizers 统计 token），
固定种子可复现，审计含 text 唯一性、答案一致性抽查、重建逐字节一致与
文件 sha256。训练行带 text 字段，可直接喂 03 训练器。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

# 用纯 tokenizers 库直接读 tokenizer.json：数据脚本不需要 torch，
# 避免 transformers 导入链拉起 CUDA DLL（小页面文件机器上会 WinError 1455）
from tokenizers import Tokenizer

# Windows 控制台输出 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
SEED = 20261013

N_GENERIC = 60          # 抽象动作组合序列条数（每条再 × 2 套标记体系）
CN_NUM = ["一", "二", "三", "四", "五", "六", "七", "八"]

# ================= 内容池（全部围绕编程/软件开发主题） =================

# 手写真实开发流程：步骤之间有真实的先后依赖，问答答案有实际意义
PROCEDURES = [
    ("编译并运行一个 C 程序", ["写好源代码文件", "用编译器编译", "检查有没有报错",
                          "运行生成的可执行文件", "核对输出结果"]),
    ("修复一个 bug", ["复现问题", "阅读报错信息", "定位出错的代码行",
                  "修改代码", "重新运行测试确认修复"]),
    ("提交一次代码", ["查看改动了哪些文件", "把改动加入暂存区", "写提交说明",
                  "提交到本地仓库", "推送到远端"]),
    ("部署一个网页", ["打包前端资源", "上传到服务器", "更新配置文件",
                  "重启服务", "打开页面验证"]),
    ("训练一个小模型", ["准备训练数据", "加载分词器", "初始化模型",
                   "跑若干轮训练", "保存权重并评估"]),
    ("写一个函数", ["想清楚输入和输出", "写函数签名", "实现函数体",
                "写几个测试用例", "运行测试确认通过"]),
    ("排查一次崩溃", ["收集崩溃日志", "找到调用栈", "定位崩溃的那一行",
                  "检查相关变量的取值", "补上边界判断"]),
    ("做一次代码审查", ["拉取对方的分支", "通读改动", "检查边界条件",
                   "留下评论", "确认修改后合并"]),
    ("配置开发环境", ["安装依赖包", "创建虚拟环境", "设置环境变量",
                  "运行自检脚本", "确认版本一致"]),
    ("处理一份数据", ["读入原始文件", "清洗缺失值", "转换成统一格式",
                  "切分训练集和验证集", "写出结果文件"]),
    ("调试一段循环", ["打印每轮的变量值", "检查循环边界", "确认退出条件",
                  "修正越界访问", "重新运行验证"]),
    ("发布一个版本", ["更新版本号", "生成变更日志", "打标签",
                  "构建发布包", "上传并公告"]),
    ("写单元测试", ["列出要覆盖的分支", "为每个分支写断言", "运行测试",
                "查看失败原因", "修到全部通过"]),
    ("优化一段慢代码", ["测量当前耗时", "找出热点函数", "替换低效算法",
                   "重新测量对比", "确认结果不变"]),
    ("合并两个分支", ["切到目标分支", "拉取最新代码", "执行合并",
                  "解决冲突", "跑一遍测试再提交"]),
    ("分析一次内存泄漏", ["记录内存占用曲线", "定位持续增长的对象", "检查有没有释放",
                    "补上释放逻辑", "复测确认稳定"]),
]

# 编程动作池：专门练顺序结构本身；组合序列没有真实依赖，
# 但"叙述顺序=答案顺序"仍然严格成立
GENERIC_ACTIONS = [
    "读取配置文件", "校验参数合法性", "打开源文件", "关闭文件句柄",
    "把结果写入日志", "遍历整个列表", "统计元素个数", "排序后取前几项",
    "把数据序列化成 JSON", "反序列化还原对象", "捕获异常并记录", "重试一次请求",
    "刷新缓存", "清空临时变量", "断言结果符合预期", "打印调试信息",
    "设置断点", "单步执行到下一行", "查看变量当前值", "继续运行到结束",
    "编译当前模块", "链接生成可执行文件", "运行冒烟测试", "收集覆盖率报告",
    "提交改动到暂存区", "写提交说明", "推送到远端分支", "拉取最新代码",
    "解决合并冲突", "打版本标签", "构建发布包", "上传到制品库",
    "回滚到上一个版本", "重启服务进程", "检查端口是否监听", "查看服务日志",
    "扩容一个实例", "缩容多余实例", "备份数据库快照", "恢复指定时间点的数据",
]

# 副词式（A 体系）中间标记：先 X，{mid} Y，…，最后 Z
A_MIDS = ["再", "然后", "接着", "随后"]
# 续写/转换样本的引导语
LEAD_INS = ["完成这个开发任务的步骤是：", "整个流程按顺序来：",
            "按先后顺序：", "操作步骤如下："]


# ================= 渲染器 =================

def _pick_mids(count: int, rng: random.Random) -> list[str]:
    """挑 count 个中间标记；够用就不重复（避免"然后…然后"），不够才允许重复。"""
    if count <= len(A_MIDS):
        return rng.sample(A_MIDS, count)
    return [rng.choice(A_MIDS) for _ in range(count)]


def render_a(steps: list[str], rng: random.Random) -> str:
    """副词式：先X，再Y，然后Z，最后W。"""
    if len(steps) == 1:
        return f"先{steps[0]}。"
    if len(steps) == 2:
        return f"先{steps[0]}，最后{steps[1]}。"
    mids = _pick_mids(len(steps) - 2, rng)
    parts = [f"先{steps[0]}"]
    parts += [f"{m}{s}" for m, s in zip(mids, steps[1:-1])]
    parts.append(f"最后{steps[-1]}")
    return "，".join(parts) + "。"


def render_b(steps: list[str]) -> str:
    """序数式：第一X，第二Y，…，最后W。"""
    if len(steps) == 2:
        return f"第一，{steps[0]}；最后，{steps[1]}。"
    parts = [f"第{CN_NUM[i]}，{s}" for i, s in enumerate(steps[:-1])]
    parts.append(f"最后，{steps[-1]}")
    return "；".join(parts) + "。"


def ordinal(i: int) -> str:
    return f"第{CN_NUM[i]}"


def render_prefix(steps: list[str], rng: random.Random) -> str:
    """未完成前缀（供续写题）：只用"先/再/然后"，绝不出现"最后"，
    否则前缀等于宣称过程已结束，与"下一步做什么"自相矛盾。"""
    if len(steps) == 1:
        return f"先{steps[0]}"
    mids = _pick_mids(len(steps) - 1, rng)
    parts = [f"先{steps[0]}"] + [f"{m}{s}" for m, s in zip(mids, steps[1:])]
    return "，".join(parts)


# ================= 样本生成 =================

def samples_for_sequence(steps: list[str], topic: str | None,
                         rng: random.Random) -> list[dict]:
    """对一个步骤列表生成全部样本类型；答案一律从 steps 派生。"""
    rows: list[dict] = []
    n = len(steps)

    # 1) 两套标记体系的完整叙述（先算出来，抽象序列要用它当上下文）
    narr_a, narr_b = render_a(steps, rng), render_b(steps)

    # 抽象组合序列没有主题名：用叙述本身作上下文前缀，
    # 既让每条问答自包含（答案可从上下文验证），又避免不同序列的问句互相重复
    ctx = "" if topic else f"过程：{narr_a}\n"
    topic_txt = f"“{topic}”的过程" if topic else "这个过程"
    QA_KINDS = {"qa_first", "qa_last", "qa_next", "qa_prev",
                "qa_position", "qa_count"}

    def add(kind: str, text: str) -> None:
        if kind in QA_KINDS:
            text = ctx + text
        rows.append({"text": text.strip(), "kind": kind,
                     "topic": topic or "generic", "n_steps": n})

    add("sequence_narrative", f"{rng.choice(LEAD_INS)}{narr_a}")
    add("sequence_narrative", f"{rng.choice(LEAD_INS)}{narr_b}")

    # 2) 首/末步问答（两套体系各问一次，问法不同）
    add("qa_first", f"问题：{topic_txt}最先做什么？\n回答：最先{steps[0]}。")
    add("qa_first", f"问题：{topic_txt}的第一步是什么？\n回答：第一步是{steps[0]}。")
    add("qa_last", f"问题：{topic_txt}最后做什么？\n回答：最后{steps[-1]}。")
    add("qa_last", f"问题：{topic_txt}的收尾步骤是什么？\n回答：收尾是{steps[-1]}，在这之前已经完成了前面所有步骤。")

    # 3) 之后/之前问答：每个内部边界一条
    for i in range(n - 1):
        add("qa_next", f"问题：{topic_txt}里，{steps[i]}之后是什么？\n回答：{steps[i]}之后是{steps[i + 1]}。")
    for i in range(1, n):
        add("qa_prev", f"问题：{topic_txt}里，{steps[i]}之前是什么？\n回答：{steps[i]}之前是{steps[i - 1]}。")

    # 4) 位置问答（序数体系）：随机抽 2 个位置
    for i in sorted(rng.sample(range(n), min(2, n))):
        if i == n - 1:
            add("qa_position", f"问题：{topic_txt}里，“{steps[i]}”是第几步？\n"
                               f"回答：它是最后一步；在“第一/第二…”的说法里，"
                               f"最后一步不再编号，直接叫“最后”。")
        else:
            add("qa_position", f"问题：{topic_txt}里，“{steps[i]}”是第几步？\n回答：是{ordinal(i)}步。")

    # 5) 步数问答
    add("qa_count", f"问题：{topic_txt}一共有几步？\n回答：一共{n}步，从{steps[0]}开始，到{steps[-1]}结束。")

    # 6) 乱序重排：打乱后要求按原顺序输出（序数式答案）
    shuffled = steps[:]
    guard = 0
    while shuffled == steps and guard < 20:
        rng.shuffle(shuffled)
        guard += 1
    if shuffled != steps:
        mixed = "；".join(shuffled)
        add("reorder", f"下面的步骤被打乱了：{mixed}。\n"
                       f"问题：按正确顺序重新排列。\n回答：{narr_b}")

    # 7) 标记体系互转（A→B 与 B→A 各一条）
    add("convert_marker", f"把下面的说法改成“第一/第二/最后”的形式：{narr_a}\n回答：{narr_b}")
    add("convert_marker", f"把下面的说法改成“先/再/然后/最后”的形式：{narr_b}\n回答：{narr_a}")

    # 8) 续写下一步：给前 k 步（1<=k<=n-1），要求带正确标记续写。
    #    k == n-1 时下一步就是末步，答案必须用"最后"；否则用中间标记。
    if n >= 3:
        k = rng.randrange(1, n)
        prefix = render_prefix(steps[:k], rng)
        nxt = steps[k]
        marker = "最后" if k == n - 1 else rng.choice(A_MIDS)
        add("continue_seq", f"{prefix}\n问题：下一步做什么？\n回答：{marker}{nxt}。")

    return rows


def build(seed: int) -> list[dict]:
    """固定种子生成全部样本；顺序确定，可复现。"""
    rng = random.Random(seed)
    rows: list[dict] = []

    # 真实流程：每条固定生成（无随机取舍），保证覆盖
    for topic, steps in PROCEDURES:
        rows.extend(samples_for_sequence(steps, topic, rng))

    # 抽象组合序列：随机抽 3-5 个不重复动作
    seen_combos: set[tuple[str, ...]] = set()
    guard = 0
    made = 0
    while made < N_GENERIC and guard < N_GENERIC * 50:
        guard += 1
        n = rng.randint(3, 5)
        combo = tuple(rng.sample(GENERIC_ACTIONS, n))
        if combo in seen_combos:
            continue
        seen_combos.add(combo)
        rows.extend(samples_for_sequence(list(combo), None, rng))
        made += 1
    if made < N_GENERIC:
        raise RuntimeError(f"抽象序列生成不收敛：{made} < {N_GENERIC}")

    rng.shuffle(rows)
    return rows


# ================= 审计 =================

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(rows: list[dict], tokenizer_dir: Path, data_dir: Path,
          seed: int) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    # 1) text 非空且唯一
    texts = [r["text"] for r in rows]
    if any(not t.strip() for t in texts):
        raise RuntimeError("存在空 text")
    if len(texts) != len(set(texts)):
        dupes = sorted({t for t in texts if texts.count(t) > 1})
        raise RuntimeError(f"存在重复样本：{dupes[:2]}")

    # 2) 类型覆盖齐全
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    expected = {"sequence_narrative", "qa_first", "qa_last", "qa_next",
                "qa_prev", "qa_position", "qa_count", "reorder",
                "convert_marker", "continue_seq"}
    missing = expected - set(kinds)
    if missing:
        raise RuntimeError(f"缺少样本类型：{sorted(missing)}")

    # 3) 答案一致性抽查：对真实流程重新派生 qa_first/qa_last/qa_count，
    #    逐条核对生成结果里的答案确实来自步骤列表
    for topic, steps in PROCEDURES:
        topic_rows = [r for r in rows if r["topic"] == topic]
        firsts = [r["text"] for r in topic_rows if r["kind"] == "qa_first"]
        if not all(steps[0] in t for t in firsts):
            raise RuntimeError(f"{topic}：qa_first 答案与步骤列表不一致")
        lasts = [r["text"] for r in topic_rows if r["kind"] == "qa_last"]
        if not all(steps[-1] in t for t in lasts):
            raise RuntimeError(f"{topic}：qa_last 答案与步骤列表不一致")
        counts = [r["text"] for r in topic_rows if r["kind"] == "qa_count"]
        if not all(f"一共{len(steps)}步" in t for t in counts):
            raise RuntimeError(f"{topic}：qa_count 答案与步骤数不一致")
        # qa_next 的每条答案都必须包含真实的后继步骤
        for r in topic_rows:
            if r["kind"] != "qa_next":
                continue
            for i in range(len(steps) - 1):
                if f"里，{steps[i]}之后是什么" in r["text"]:
                    if f"之后是{steps[i + 1]}" not in r["text"]:
                        raise RuntimeError(f"{topic}：qa_next 答案错误：{r['text']}")

    # 4) 标记体系纯净性：A 体系叙述不得混入"第一/第二"，B 体系不得混入"先…最后"式开头
    for r in rows:
        if r["kind"] != "sequence_narrative":
            continue
        body = r["text"].split("：", 1)[-1]
        if body.startswith("先") and ("第一" in body or "第二" in body):
            raise RuntimeError(f"A 体系混入序数标记：{r['text']}")
        if body.startswith("第一") and body.count("先") > 0 and "，先" in body:
            raise RuntimeError(f"B 体系混入副词标记：{r['text']}")

    # 5) 复现性：同种子重建逐字节一致
    if json.dumps(build(seed), ensure_ascii=False) != json.dumps(rows, ensure_ascii=False):
        raise RuntimeError("复现校验失败：重建结果与写盘结果不一致")

    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False).ids)
               for r in rows]
    topics = sorted({r["topic"] for r in rows})
    return {
        "status": "datasets_prepared",
        "seed": seed,
        "sample_count": len(rows),
        "sample_kinds": kinds,
        "procedure_count": len(PROCEDURES),
        "generic_sequence_count": N_GENERIC,
        "topics": topics,
        "total_tokens": sum(lengths),
        "average_tokens": round(sum(lengths) / len(lengths), 1),
        "max_tokens": max(lengths),
        "duplicate_texts": 0,
        "answer_consistency_checked": True,
        "reproducibility_ok": True,
        "file_sha256": {"step_order_v1_train.jsonl":
                        sha256_file(data_dir / "step_order_v1_train.jsonl")},
        "note": ("两套顺序标记（先/再/然后/最后 与 第一/第二/最后）；"
                 "所有问答答案从步骤列表程序化派生并经审计核对。"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成步骤顺序语义预训练数据 v1；不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR,
                        help="本地基础 tokenizer 目录")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 20261013）")
    args = parser.parse_args()

    rows = build(args.seed)
    write_jsonl(args.data_dir / "step_order_v1_train.jsonl", rows)
    report = audit(rows, args.tokenizer, args.data_dir, args.seed)
    (args.data_dir / "step_order_v1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
