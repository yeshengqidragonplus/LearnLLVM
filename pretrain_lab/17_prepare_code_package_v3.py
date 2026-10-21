# -*- coding: utf-8 -*-
"""生成 v3 扩容版代码 Package 路由/专家数据：标识符池、字面量与问法程序化扩容。

与 v2（16_prepare_code_package_v2.py）的区别：
- 专家训练集从 16 条扩到 ~1000 条：240 个训练标识符池 × 10 种 C 类型 × 随机字面量 × 多种问法/答法；
- 专家评估集每组 24 条（对齐 08/v5-C1 的每档 24 条判据）：
  seen_name_new_value 取训练见过的标识符 + 评估值域的新字面量；
  unseen_identifier 使用与训练池完全不相交的独立标识符池（含 v0/v2 沿用来的 4 个，便于跨轮对比）；
- 路由 train/eval 每类 ~100/24 条：训练意图由组合模板生成，评估意图手写并断言与训练零重叠；
- 训练行额外带 text 字段（prompt+target / router_input+路由选择），可直接喂 03 训练器；
- 固定随机种子，审计含输出文件 sha256 与复生成一致性检查，可跨机复现。

仍然只生成与审计数据，不启动任何训练。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import string
import sys
from pathlib import Path

from transformers import AddedToken, AutoTokenizer

from code_package_frontend_v1 import PACKAGE, SessionCache, expert_input, package_request

# Windows 控制台输出 UTF-8，避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
SEED = 20261012

# 目标规模（用户确认的扩容档位）
EXPERT_SINGLE_INIT = 640   # 单条声明+初始化
EXPERT_UNINIT = 120        # 只声明不初始化
EXPERT_DECL_ASSIGN = 120   # 声明+后续赋值（两个子包）
EXPERT_TWO_DECLS = 120     # 两条独立声明（两个子包）
EVAL_PER_GROUP = 24        # seen_name_new_value / unseen_identifier 各 24
ROUTER_PER_CLASS = 100     # CODE_EXPLAIN / UNKNOWN 各 ~100
ROUTER_EVAL_PER_CLASS = 24

# ================= C 类型体系 =================

TYPE_KEYS = ["int", "char", "short", "long", "unsigned int", "unsigned short",
             "unsigned long", "unsigned char", "float", "double"]

TYPE_CN = {
    "int": "int 是整数类型",
    "char": "char 是字符类型",
    "short": "short 是短整数类型",
    "long": "long 是长整数类型，具体位宽由 C 实现决定",
    "unsigned int": "unsigned int 是无符号整数类型",
    "unsigned short": "unsigned short 是无符号短整数类型",
    "unsigned long": "unsigned long 是无符号长整数类型",
    "unsigned char": "unsigned char 是无符号字符类型",
    "float": "float 是单精度浮点类型",
    "double": "double 是双精度浮点类型",
}

# 训练值域与评估值域分离：seen_name_new_value 组的“新值”由值域保证
TRAIN_SPACE = {
    "int": (0, 99999), "short": (-999, 999), "long": (0, 99999),
    "unsigned int": (0, 65535), "unsigned short": (0, 9999),
    "unsigned long": (0, 99999), "unsigned char": (0, 255),
    "float": (0.1, 99.9), "double": (0.1, 99.9),
}
EVAL_SPACE = {
    "int": (100000, 999999), "short": (1000, 9999), "long": (100000, 999999),
    "unsigned int": (100000, 999999), "unsigned short": (10000, 65535),
    "unsigned long": (100000, 999999), "unsigned char": (0, 255),
    "float": (100.0, 999.9), "double": (100.0, 999.9),
}
SUFFIX = {"unsigned int": "U", "unsigned short": "U", "unsigned long": "UL",
          "long": "L", "float": "f"}
SIGNED_NEG_P = {"int": 0.15, "short": 0.3, "long": 0.1}


def lit_desc(type_key: str, lit: str) -> str:
    """按类型生成字面量的中文说明。"""
    if type_key == "char":
        return f"{lit} 是字符常量"
    if type_key in ("float",):
        return f"{lit} 是单精度浮点字面量"
    if type_key == "double":
        return f"{lit} 是双精度浮点字面量"
    if type_key == "long":
        return f"{lit} 是 long 型整数字面量"
    if type_key == "unsigned long":
        return f"{lit} 是无符号 long 型字面量"
    if type_key.startswith("unsigned"):
        return f"{lit} 是无符号整数字面量"
    return f"{lit} 是整数字面量"


def gen_literal(type_key: str, rng: random.Random, space: str) -> str:
    """按类型与值域生成随机字面量。"""
    table = TRAIN_SPACE if space == "train" else EVAL_SPACE
    if type_key == "char":
        return "'" + rng.choice(string.ascii_letters + string.digits) + "'"
    lo, hi = table[type_key]
    if type_key in ("float", "double"):
        prec = rng.choice([1, 2, 3]) if space == "train" else rng.choice([2, 3])
        value = f"{round(rng.uniform(lo, hi), prec):.{prec}f}"
        return value + SUFFIX.get(type_key, "")
    value = rng.randint(lo, hi)
    if rng.random() < SIGNED_NEG_P.get(type_key, 0.0):
        value = -rng.randint(max(1, lo if lo > 0 else 1), max(2, hi))
    return str(value) + SUFFIX.get(type_key, "")


# ================= 标识符池 =================

# 训练名词池：不得与 EVAL_UNSEEN 相交；刻意不含 sample（评估有 sample_count）
TRAIN_NOUNS = [
    "buffer", "counter", "sensor", "packet", "offset", "length", "width",
    "height", "depth", "score", "level", "limit", "threshold", "frame",
    "block", "chunk", "slot", "channel", "device", "socket", "thread",
    "handle", "cursor", "marker", "record", "entry", "item", "element",
    "node", "edge", "matrix", "vector", "array", "queue", "stack", "cache",
    "index", "total", "count", "size", "position", "angle", "distance",
    "duration", "interval", "retry", "timeout", "delay", "stride", "span",
    "pitch", "gain", "bias", "scale", "shift", "mask", "flags", "state",
    "mode", "phase", "step", "tick", "epoch", "batch", "row", "col",
    "pixel", "tile", "grid", "axis", "origin", "target", "source", "result",
]
MODS = ["max", "min", "cur", "prev", "next", "new", "old", "raw", "tmp",
        "init", "peak", "base", "src", "dst", "ref", "aux"]

# 未见标识符评估池（24 个）：前 4 个沿用 v0/v2 留出集，便于跨轮对比
EVAL_UNSEEN = [
    "temperature", "probability", "capacity", "sample_count",
    "humidity", "voltage", "current", "frequency", "amplitude", "diameter",
    "circumference", "velocity", "acceleration", "momentum", "pressure",
    "density", "volume", "luminance", "contrast", "saturation",
    "brightness", "opacity", "viscosity", "resistance",
]


def build_train_pool(rng: random.Random, target: int = 240) -> list[str]:
    """组合生成训练标识符池：裸名词 + 前缀/后缀修饰 + 数字后缀。"""
    pool = set(TRAIN_NOUNS)
    guard = 0
    while len(pool) < target:
        guard += 1
        if guard > target * 50:
            raise RuntimeError("标识符池生成不收敛")
        noun = rng.choice(TRAIN_NOUNS)
        style = rng.randrange(3)
        if style == 0:
            pool.add(f"{rng.choice(MODS)}_{noun}")
        elif style == 1:
            pool.add(f"{noun}_{rng.choice(MODS)}")
        else:
            pool.add(f"{noun}{rng.randint(2, 99)}")
    overlap = pool & set(EVAL_UNSEEN)
    if overlap:
        raise RuntimeError(f"训练池与未见评估池相交：{sorted(overlap)}")
    return sorted(pool)


# ================= 问法/答法模板 =================
# 每条：(问题模板, [答法变体...])；占位符 name/lit/type_kw/type_desc/lit_desc

SINGLE_QA = [
    ("解释每个部分", [
        "{type_desc}；{name} 是变量名/标识符；{lit_desc}，用作初始值。",
        "这行代码声明了 {type_kw} 变量 {name}；{lit_desc}，作为初始值。"]),
    ("{name} 是什么？", [
        "{name} 是变量名，也是标识符；{type_desc}；{lit_desc}，作为初始值。",
        "{name} 是标识符，命名一个 {type_kw} 类型变量；初始值由 {lit} 提供。"]),
    ("{name} 的类型是什么？", [
        "{name} 的类型是 {type_kw}；{lit_desc}。",
        "{type_desc}；变量名是 {name}，初始值是 {lit}。"]),
    ("{lit} 是什么？", [
        "{lit_desc}，用作 {name} 的初始值；{name} 是变量名。",
        "{lit_desc}；它初始化了 {type_kw} 类型变量 {name}。"]),
    ("解释类型和字面量", [
        "{type_desc}；{lit_desc}，作为 {name} 的初始值。",
        "类型是 {type_kw}；字面量 {lit} 提供初始值；{name} 是标识符。"]),
    ("{name} 保存什么？", [
        "{name} 保存初始值 {lit}；{type_desc}。",
        "{name} 中保存的值是 {lit}；{type_desc}。"]),
    ("{name} 是变量名还是值？", [
        "{name} 是变量名/标识符，不是值；{lit_desc}。",
        "变量名是 {name}；值由 {lit} 提供；{type_desc}。"]),
    ("{name} 是 C 关键字吗？", [
        "不是。{name} 是用户命名的标识符；{type_kw} 才是类型关键字；{lit_desc}。",
        "不是。{type_kw} 是关键字；{name} 只是变量名；{lit_desc}。"]),
    ("{name} 是 identifier 吗？", [
        "是。{name} 是标识符，命名一个 {type_kw} 类型变量；{lit_desc}。",
        "是。{name} 作为标识符使用；{type_desc}；{lit_desc}。"]),
    ("这行代码声明了什么？", [
        "声明了 {type_kw} 类型变量 {name}，初始值是 {lit}。",
        "这行代码声明 {name} 为 {type_kw} 变量，并用 {lit} 初始化。"]),
    ("说明等号两边的内容", [
        "等号左边声明 {type_kw} 变量 {name}；等号右边 {lit} 是初始值。",
        "左边是类型 {type_kw} 和变量名 {name}；右边是初始值 {lit}。"]),
    ("逐个说明关键字、变量名和值", [
        "关键字是 {type_kw}；变量名是 {name}；值是 {lit}。",
        "{type_kw} 是类型关键字，{name} 是标识符，{lit} 是字面量初始值。"]),
]

UNINIT_QA = [
    ("这行代码声明了什么？", [
        "声明了 {type_kw} 类型变量 {name}；没有初始化，值尚未定义。",
        "{type_desc}；{name} 是标识符；这里没有提供初始值。"]),
    ("{name} 有初始值吗？", [
        "没有。这行只声明了 {type_kw} 变量 {name}，未初始化。",
        "没有初始值；{name} 是标识符，{type_desc}，值要等后续赋值。"]),
    ("解释每个部分", [
        "{type_desc}；{name} 是变量名/标识符；分号结束声明，本行没有初始值。",
        "{type_kw} 是类型关键字，{name} 是标识符；该声明未初始化。"]),
]

DECL_ASSIGN_QA = [
    ("区分声明和赋值", [
        "{type_kw} {name}; 是声明；{name} = {lit}; 是赋值，不是声明时初始化。",
        "第一句声明 {type_kw} 变量 {name}；第二句把 {lit} 赋给它，属于赋值。"]),
    ("两句各做了什么？", [
        "第一句声明 {type_kw} 变量 {name}；第二句将 {lit} 赋值给 {name}。",
        "先声明 {name}（类型 {type_kw}），再赋值 {lit}；两句分别是声明和赋值。"]),
]

TWO_DECLS_QA = [
    ("有几个变量？", [
        "有两个变量：{name} 和 {name2}；类型分别是 {type_kw} 和 {type_kw2}，初始值分别为 {lit} 和 {lit2}。",
        "共两个：{type_kw} 类型的 {name}（初始值 {lit}）和 {type_kw2} 类型的 {name2}（初始值 {lit2}）。"]),
    ("分别声明了什么？", [
        "声明了 {type_kw} 变量 {name}，初始值 {lit}；以及 {type_kw2} 变量 {name2}，初始值 {lit2}。",
        "第一句声明 {name}（{type_kw}，初值 {lit}）；第二句声明 {name2}（{type_kw2}，初值 {lit2}）。"]),
]


def make_ctx(rng: random.Random, pool: list[str], space: str,
             type_key: str | None = None) -> dict:
    """抽取类型/标识符/字面量，组装模板占位符上下文。"""
    type_key = type_key or TYPE_KEYS[rng.randrange(len(TYPE_KEYS))]
    name = rng.choice(pool)
    lit = gen_literal(type_key, rng, space)
    return {"type_kw": type_key, "type_desc": TYPE_CN[type_key],
            "name": name, "lit": lit, "lit_desc": lit_desc(type_key, lit)}


def pick_qa(rng: random.Random, qa_table: list, ctx: dict) -> tuple[str, str]:
    """从模板表抽一组问法/答法并实例化。"""
    q_fmt, a_variants = qa_table[rng.randrange(len(qa_table))]
    question = q_fmt.format(**ctx)
    answer = rng.choice(a_variants).format(**ctx)
    return question, answer


# ================= 前端打包（复用 v2 流程） =================

def make_children(code: str, question: str, cache: SessionCache) -> tuple[list[str], str]:
    """走真实前端：原始请求 -> 子包切分 -> 缓存载荷 -> 专家 prompt。"""
    packaged = package_request(f"解释代码：{code}", cache)
    handles = packaged["child_handles"]
    if not handles:
        raise ValueError(f"样本没有生成子包：{code!r}")
    prompt = "\n".join(expert_input(cache, handle, question) for handle in handles)
    return handles, prompt


# ================= 路由意图 =================

EXPLAIN_VERBS = ["解释", "说明", "讲解", "分析一下", "解读", "看一下", "讲讲",
                 "帮我理解", "请解释"]
EXPLAIN_OBJS = ["这段代码", "这行程序", "这个声明", "代码片段", "这条语句",
                "这行代码", "该程序片段", "这段 C 代码"]
EXPLAIN_ASPECTS = ["每一部分的意思", "变量和数值", "英文术语", "关键字、变量名和值",
                   "类型和初始值", "声明了什么", "含义", "各部分的作用",
                   "等号两边的关系", "标识符代表什么"]

# 评估意图手写（每类 24 条）；训练组合生成后断言与评估零重叠
EVAL_EXPLAIN = [
    "我看不懂这行代码，讲讲各部分含义", "请不要运行，只说明语句的意义",
    "这段 C 声明在表达什么？", "帮我逐词拆解这行程序",
    "这里的 identifier 指的是什么？", "分析声明里的变量名、类型和值",
    "用中文讲讲这段代码的意思", "这个语句里哪个是名字哪个是数值？",
    "说明该声明中每个单词的角色", "解释 variable 和 identifier 在这行代码中的区别",
    "这条语句完成了什么操作？", "请讲解代码中的类型和初始化",
    "我想理解这条语句", "代码里的 name 和 value 分别指什么？",
    "这段程序片段该如何读？", "逐符号解释这行声明",
    "这个等号表达了什么关系？", "告诉我这行代码定义了什么",
    "该变量的类型和初值分别是什么？", "解释这个声明语句的结构",
    "这段代码如何表示一个变量？", "说明每个代码符号的作用",
    "解释这段代码，不要执行", "请用中文说明程序片段",
]
EVAL_UNKNOWN = [
    "把这段代码实际执行一下", "帮我把程序改成使用浮点数",
    "代码中的文字请逐字复制出来", "给程序加上注释并重构",
    "运行这段程序并告诉我输出", "检查代码能不能编译通过",
    "为这段代码画流程图", "解释 variable 这个英文单词的普通词义",
    "把代码移植到 Python", "计算这段程序运行所需的时间复杂度",
    "为代码生成完整测试套件", "请格式化这段程序",
    "修复这段代码里的编译错误", "把这段代码翻译成另一种语言",
    "仅原样复述代码，不作解释", "总结这段设计文档",
    "优化这段程序的性能", "给这段代码写单元测试",
    "在代码字符串中查找单词 value", "调试这段程序直到能运行",
    "把变量名全部重命名为单字母", "静态分析这段代码的潜在缺陷",
    "将这行声明改写成宏定义", "逐字朗读这段代码",
]


def explain_candidates() -> list[str]:
    cands = [f"{v}{o}" for v in EXPLAIN_VERBS for o in EXPLAIN_OBJS]
    cands += [f"{v}{o}，{a}" for v in EXPLAIN_VERBS for o in EXPLAIN_OBJS
              for a in EXPLAIN_ASPECTS]
    return cands


def unknown_candidates() -> list[str]:
    run_v = ["运行", "执行", "编译并运行", "实际跑一下", "先编译再运行"]
    code_o = ["这段程序", "这段代码", "这个程序片段", "该代码", "这行程序"]
    cands = [f"{v}{o}" for v in run_v for o in code_o]
    cands += [f"{v}{o}并给出输出" for v in run_v for o in code_o]
    cands += [f"{v}{o}，告诉我结果" for v in run_v for o in code_o]
    mod_v = ["修改", "重构", "优化", "改写", "调整"]
    mod_o = ["以修复编译错误", "以提升性能", "为使用浮点数", "成递归写法",
             "以处理边界情况", "以支持负数输入"]
    cands += [f"{mv}这段代码{mo}" for mv in mod_v for mo in mod_o]
    cands += [f"{mv}程序{mo}" for mv in mod_v for mo in mod_o]
    langs = ["Python", "Java", "Rust", "汇编", "C++"]
    cands += [f"把{o}翻译成{l}" for o in ("这段代码", "这个程序", "该代码片段")
              for l in langs]
    cands += ["把代码原样输出，不要解释", "原样抄写这段程序",
              "把代码内容一字不差地抄一遍"]
    cands += ["为这段代码写集成测试", "给程序补充边界测试",
              "生成测试用例覆盖所有分支", "为代码设计测试计划"]
    cands += ["验证程序有没有未定义行为", "找出代码里的潜在 bug",
              "静态分析这段程序", "检查这段代码的内存安全"]
    cands += ["概括这份设计文档的要点", "为这段程序绘制调用关系图",
              "计算这段程序的时间复杂度", "把代码写入头文件并生成文档",
              "给这段程序写使用说明书", "请对这段程序做代码美化"]
    return cands


# ================= 数据构建 =================

def build(seed: int) -> dict[str, list[dict]]:
    """按固定种子生成全部四个数据集（可重复调用做复现校验）。"""
    rng = random.Random(seed)
    pool = build_train_pool(rng)

    # ---- 路由 ----
    reserved = set(EVAL_EXPLAIN) | set(EVAL_UNKNOWN)

    def pick(cands: list[str], n: int) -> list[str]:
        uniq = sorted({c for c in cands if c not in reserved})
        rng.shuffle(uniq)
        if len(uniq) < n:
            raise RuntimeError(f"路由意图候选不足：{len(uniq)} < {n}")
        return uniq[:n]

    explain_train = pick(explain_candidates(), ROUTER_PER_CLASS)
    unknown_train = pick(unknown_candidates(), ROUTER_PER_CLASS)
    router_train = ([{"router_input": f"意图：{i}\n包类型：{PACKAGE}", "label": "CODE_EXPLAIN",
                      "text": f"意图：{i}\n包类型：{PACKAGE}\n路由选择：CODE_EXPLAIN"}
                     for i in explain_train]
                    + [{"router_input": f"意图：{i}\n包类型：{PACKAGE}", "label": "UNKNOWN",
                        "text": f"意图：{i}\n包类型：{PACKAGE}\n路由选择：UNKNOWN"}
                       for i in unknown_train])
    rng.shuffle(router_train)
    router_eval = ([{"router_input": f"意图：{i}\n包类型：{PACKAGE}", "label": "CODE_EXPLAIN"}
                    for i in EVAL_EXPLAIN]
                   + [{"router_input": f"意图：{i}\n包类型：{PACKAGE}", "label": "UNKNOWN"}
                      for i in EVAL_UNKNOWN])
    rng.shuffle(router_eval)

    # ---- 专家训练 ----
    expert_train: list[dict] = []
    seen_prompts: set[str] = set()
    train_pairs: set[tuple[str, str]] = set()   # (标识符, 字面量)，供 seen eval 排重
    names_used: set[str] = set()

    def add_sample(code: str, ctx: dict, qa_table: list, kind: str) -> None:
        question, answer = pick_qa(rng, qa_table, ctx)
        cache = SessionCache()
        handles, prompt = make_children(code, question, cache)
        if prompt in seen_prompts:
            return
        seen_prompts.add(prompt)
        payload = "\n".join(cache.children[h].payload for h in handles)
        expert_train.append({
            "prompt": prompt, "target": answer, "text": prompt + answer,
            "payload": payload, "identifier_group": "train_seen",
            "package_type": "code_statement", "qkind": kind,
        })
        names_used.add(ctx["name"])
        train_pairs.add((ctx["name"], ctx["lit"]))

    # 单条声明+初始化：按类型轮转保证 10 种类型覆盖均衡
    attempt = 0
    while sum(r["qkind"] == "single_init" for r in expert_train) < EXPERT_SINGLE_INIT:
        attempt += 1
        if attempt > EXPERT_SINGLE_INIT * 60:
            raise RuntimeError("single_init 生成不收敛")
        idx = len([r for r in expert_train if r["qkind"] == "single_init"])
        type_key = TYPE_KEYS[idx % len(TYPE_KEYS)]
        ctx = make_ctx(rng, pool, "train", type_key)
        code = f"{ctx['type_kw']} {ctx['name']} = {ctx['lit']};"
        add_sample(code, ctx, SINGLE_QA, "single_init")

    # 只声明不初始化
    attempt = 0
    while sum(r["qkind"] == "uninit" for r in expert_train) < EXPERT_UNINIT:
        attempt += 1
        if attempt > EXPERT_UNINIT * 60:
            raise RuntimeError("uninit 生成不收敛")
        ctx = make_ctx(rng, pool, "train")
        if ctx["type_kw"] in ("float", "double"):
            continue  # 无初始值场景对浮点无差别意义，跳过保持类型多样
        code = f"{ctx['type_kw']} {ctx['name']};"
        add_sample(code, ctx, UNINIT_QA, "uninit")

    # 声明 + 后续赋值（两个子包）
    attempt = 0
    while sum(r["qkind"] == "decl_assign" for r in expert_train) < EXPERT_DECL_ASSIGN:
        attempt += 1
        if attempt > EXPERT_DECL_ASSIGN * 60:
            raise RuntimeError("decl_assign 生成不收敛")
        ctx = make_ctx(rng, pool, "train")
        if ctx["type_kw"] in ("float", "double"):
            continue
        code = f"{ctx['type_kw']} {ctx['name']}; {ctx['name']} = {ctx['lit']};"
        add_sample(code, ctx, DECL_ASSIGN_QA, "decl_assign")

    # 两条独立声明（两个子包）
    attempt = 0
    while sum(r["qkind"] == "two_decls" for r in expert_train) < EXPERT_TWO_DECLS:
        attempt += 1
        if attempt > EXPERT_TWO_DECLS * 60:
            raise RuntimeError("two_decls 生成不收敛")
        ctx = make_ctx(rng, pool, "train")
        ctx2 = make_ctx(rng, pool, "train")
        if ctx2["name"] == ctx["name"]:
            continue
        ctx["name2"], ctx["type_kw2"], ctx["lit2"] = ctx2["name"], ctx2["type_kw"], ctx2["lit"]
        code = (f"{ctx['type_kw']} {ctx['name']} = {ctx['lit']}; "
                f"{ctx['type_kw2']} {ctx['name2']} = {ctx['lit2']};")
        add_sample(code, ctx, TWO_DECLS_QA, "two_decls")

    rng.shuffle(expert_train)

    # ---- 专家评估 ----
    def eval_row(type_key: str, name: str, group: str) -> dict:
        lit = gen_literal(type_key, rng, "eval")
        tries = 0
        while (name, lit) in train_pairs and tries < 50:
            lit = gen_literal(type_key, rng, "eval")
            tries += 1
        ctx = {"type_kw": type_key, "type_desc": TYPE_CN[type_key], "name": name,
               "lit": lit, "lit_desc": lit_desc(type_key, lit)}
        question, _ = pick_qa(rng, SINGLE_QA, ctx)
        code = f"{type_key} {name} = {lit};"
        cache = SessionCache()
        handles, prompt = make_children(code, question, cache)
        payload = "\n".join(cache.children[h].payload for h in handles)
        return {"prompt": prompt, "payload": payload, "question": question,
                "expected_points": [f"{name} 是标识符/变量名", f"类型是 {type_key}",
                                    f"{lit} 是初始值"],
                "identifier_group": group, "package_type": "code_statement"}

    expert_eval: list[dict] = []
    seen_names = sorted(n for n in names_used if n not in set(EVAL_UNSEEN))
    rng.shuffle(seen_names)
    for name in seen_names[:EVAL_PER_GROUP]:
        expert_eval.append(eval_row(TYPE_KEYS[rng.randrange(len(TYPE_KEYS))],
                                    name, "seen_name_new_value"))
    for name in EVAL_UNSEEN[:EVAL_PER_GROUP]:
        expert_eval.append(eval_row(TYPE_KEYS[rng.randrange(len(TYPE_KEYS))],
                                    name, "unseen_identifier"))

    return {"router_train": router_train, "router_eval": router_eval,
            "expert_train": expert_train, "expert_eval": expert_eval,
            "_meta": {"pool": pool, "names_used": sorted(names_used)}}


# ================= 审计 =================

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(data: dict, meta: dict, tokenizer_dir: Path, seed: int,
          data_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    before = tokenizer.encode(PACKAGE, add_special_tokens=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [
        AddedToken(PACKAGE, special=True, normalized=False)]})
    package_ids = tokenizer.encode(PACKAGE, add_special_tokens=False)
    if len(package_ids) != 1:
        raise RuntimeError(f"Package 未编码为单 token：{package_ids}")

    # 1) 路由隔离：任何专家载荷都不得出现在路由输入里
    route_text = [r["router_input"] for s in ("router_train", "router_eval")
                  for r in data[s]]
    payloads = [r["payload"] for s in ("expert_train", "expert_eval")
                for r in data[s]]
    leaked = [(p, t) for p in payloads for t in route_text if p and p in t]
    if leaked:
        raise RuntimeError(f"代码载荷泄漏进路由输入：{leaked[:2]}")
    if any(PACKAGE not in t for t in route_text):
        raise RuntimeError("路由样本缺少 package 类型标记")

    # 2) 路由 train/eval 意图零重叠
    overlap = ({r["router_input"] for r in data["router_train"]}
               & {r["router_input"] for r in data["router_eval"]})
    if overlap:
        raise RuntimeError(f"路由 train/eval 重叠：{sorted(overlap)[:3]}")

    # 3) 标识符隔离：未见评估池不得出现在任何专家训练文本中（子串级）
    train_text_blob = "\n".join(r["prompt"] + r["target"]
                                for r in data["expert_train"])
    unseen_hits = [w for w in EVAL_UNSEEN if w in train_text_blob]
    if unseen_hits:
        raise RuntimeError(f"未见标识符泄漏进专家训练集：{unseen_hits}")
    pool_overlap = set(meta["pool"]) & set(EVAL_UNSEEN)
    if pool_overlap:
        raise RuntimeError(f"训练池与未见池相交：{sorted(pool_overlap)}")

    # 4) seen eval 的标识符必须真的在训练里出现过
    # 名称取 "=" 前最后一个词，兼容 unsigned int 等多词类型
    seen_eval_names = [r["payload"].split("=")[0].strip().split()[-1]
                       for r in data["expert_eval"]
                       if r["identifier_group"] == "seen_name_new_value"]
    missing = [n for n in seen_eval_names if n not in set(meta["names_used"])]
    if missing:
        raise RuntimeError(f"seen eval 标识符未在训练中出现：{missing}")

    # 5) 专家训练 prompt 唯一且格式一致
    prompts = [r["prompt"] for r in data["expert_train"]]
    if len(prompts) != len(set(prompts)):
        raise RuntimeError("专家训练集存在重复 prompt")
    if any(not (p.startswith(PACKAGE) and p.endswith("回答：")) for p in prompts):
        raise RuntimeError("专家 prompt 格式不一致")

    # 6) 复现性：同种子重建必须逐字节一致
    rebuilt = build(seed)
    for split in ("router_train", "router_eval", "expert_train", "expert_eval"):
        if json.dumps(rebuilt[split], ensure_ascii=False) != json.dumps(data[split], ensure_ascii=False):
            raise RuntimeError(f"复现校验失败：{split}")

    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False))
               for r in data["expert_train"]]
    qkinds: dict[str, int] = {}
    for r in data["expert_train"]:
        qkinds[r["qkind"]] = qkinds.get(r["qkind"], 0) + 1
    router_labels = {split: {lab: sum(r["label"] == lab for r in data[split])
                             for lab in ("CODE_EXPLAIN", "UNKNOWN")}
                     for split in ("router_train", "router_eval")}
    files = {f"code_package_v3_{s}.jsonl": sha256_file(data_dir / f"code_package_v3_{s}.jsonl")
             for s in ("router_train", "router_eval", "expert_train", "expert_eval")}
    return {
        "seed": seed,
        "package_marker_ids_before_extension": before,
        "package_marker_id_after_extension": package_ids[0],
        "package_marker_is_one_token": True,
        "router_payload_leaks": len(leaked),
        "router_train_eval_overlap": len(overlap),
        "router_label_counts": router_labels,
        "identifier_pool_size": len(meta["pool"]),
        "names_used_in_training": len(meta["names_used"]),
        "unseen_identifier_leaks": len(unseen_hits),
        "expert_train_count": len(data["expert_train"]),
        "expert_train_qkinds": qkinds,
        "expert_train_tokens": sum(lengths),
        "expert_train_avg_tokens": round(sum(lengths) / len(lengths), 1),
        "expert_train_max_tokens": max(lengths),
        "expert_eval_count": len(data["expert_eval"]),
        "expert_eval_groups": {g: sum(r["identifier_group"] == g
                                      for r in data["expert_eval"])
                               for g in ("seen_name_new_value", "unseen_identifier")},
        "reproducibility_ok": True,
        "file_sha256": files,
        "semantics": "raw request -> pre-tokenizer Package/cache -> router(intent + type only) -> selected expert unpacks child payload",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 v3 扩容版代码 Package 路由/专家数据并审计；不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR,
                        help="Tokenizer 目录")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 20261012）")
    args = parser.parse_args()

    data = build(args.seed)
    meta = data.pop("_meta")
    for name, rows in data.items():
        write_jsonl(args.data_dir / f"code_package_v3_{name}.jsonl", rows)
    report = audit(data, meta, args.tokenizer, args.seed, args.data_dir)
    (args.data_dir / "code_package_v3_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
