# -*- coding: utf-8 -*-
"""生成 C/C++ 声明基础双向数据 v1：代码⇄中文注释 + 分号语义 + 嵌套原子包。

用户指定的设计（方案 A）：
- 正向（代码→注释）：`int i = 3;` 能得到"声明一个整型变量 i，初始值是 3"；
  语句打成 <|PKG_CODE|> 语句子包，内部的数值与变量名再打成嵌套原子包
  <|PKG_NUM|>（载荷=3）与 <|PKG_NAME|>（载荷=i）。
- 反向（注释→代码）："声明一个整型变量 i，初始值是 20" 能写出 `int i = 20;`；
  描述里的名字与数值同样打包，路由器只看到包结构、看不到载荷，
  专家解包后才拿到真实的 i 和 20。
- 声明形态：变量 / const 常量 / static const 静态常量；初始值可选。
- 分号语义：专门训练"; 是一句结束"，含 `int a = 1, b = 2;`（一个分号=一句、
  两个变量）与 `int a = 1; int b = 2;`（两个分号=两句）的对比。

所有样本走真实前端（code_package_frontend_v1 的 nested 打包与 package_description），
答案从声明结构程序化派生。固定种子可复现；审计含原子包一致性、反向载荷隐藏、
eval 未见名字零泄漏、分号计数一致、text 唯一、重建逐字节一致与文件 sha256。
训练行带 text 字段，喂 03 训练器需 --package-code-token --package-nested-tokens。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
from pathlib import Path

# 纯 tokenizers 统计 token：数据脚本不导入 torch（本机页面文件小，
# transformers 导入链会拉 CUDA DLL 触发 WinError 1455）
from tokenizers import Tokenizer

from code_package_frontend_v1 import (
    PACKAGE, PKG_NAME, PKG_NUM, SessionCache,
    expert_input_description, expert_input_nested,
    package_description, package_request,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")
SEED = 20261014

PER_COMBO = 15          # 每（类型×形态）组合的正向/反向样本数
N_SEMICOLON = 80        # 分号语义样本数
EVAL_FWD = 24           # 正向评估（未见名字）
EVAL_REV = 24           # 反向评估（未见名字）
EVAL_SEMI = 12          # 分号语义评估

# ================= 类型与形态 =================

# (C 类型, 中文名, 值类别)
TYPES = [
    ("int", "整型", "int"),
    ("char", "字符型", "char"),
    ("short", "短整型", "int"),
    ("long", "长整型", "long"),
    ("float", "单精度浮点型", "float"),
    ("double", "双精度浮点型", "double"),
    ("unsigned int", "无符号整型", "unsigned"),
]

# (形态键, C 前缀, 中文形态词, 是否带初始值)
FORMS = [
    ("var_init", "", "变量", True),
    ("var_uninit", "", "变量", False),
    ("const", "const ", "常量", True),
    ("static_const", "static const ", "静态常量", True),
]

# 训练名字池（小写普通变量 + 大写常量风格）；与 EVAL_NAMES 零交集且互不为子串
TRAIN_NAMES_LOWER = [
    "i", "j", "k", "n", "m", "x", "y", "z", "a", "b", "count", "total",
    "sum", "num", "idx", "size", "len", "val", "tmp", "buf", "pos", "ch",
    "grade", "score", "age", "year", "month", "day", "hour", "minute",
    "second", "base", "step", "flag", "mode", "key", "port", "offset",
    "width", "height", "depth", "ratio", "rate", "scale", "level",
    "start", "finish", "cursor", "handle", "slot",
]
TRAIN_NAMES_UPPER = [
    "MAX", "MIN", "LIMIT", "CAP", "DEFAULT", "VERSION", "TIMEOUT",
    "RETRY", "BOUND", "SEED", "BASE", "SPAN", "UNIT", "QUOTA", "THRESHOLD",
]
# 评估未见名字池：不得出现在任何训练文本中（审计做子串级检查）
EVAL_NAMES_LOWER = [
    "alpha", "beta", "gamma", "delta", "theta", "sigma", "radius",
    "volume", "mass", "speed", "force", "energy",
]
EVAL_NAMES_UPPER = [
    "PI", "GOLDEN", "LIGHT_SPEED", "GRAVITY", "PLANCK", "AVOGADRO",
    "BOHR", "FARADAY", "BOLTZMANN", "STEFAN", "WIEN", "RYDBERG",
]

TRAIN_SPACE = {
    "int": (0, 9999), "short": (-999, 999), "long": (0, 99999),
    "unsigned": (0, 65535), "float": (0.1, 99.9), "double": (0.1, 999.9),
}
EVAL_SPACE = {
    "int": (100000, 999999), "short": (10000, 30000), "long": (1000000, 9999999),
    "unsigned": (100000, 999999), "float": (1000.0, 9999.0), "double": (1000.0, 99999.0),
}
SUFFIX = {"long": "L", "unsigned": "U", "float": "f"}


def gen_value(kind: str, rng: random.Random, space: str) -> str:
    """按值类别与值域生成字面量（训练/评估值域分离）。"""
    table = TRAIN_SPACE if space == "train" else EVAL_SPACE
    if kind == "char":
        return "'" + rng.choice(string.ascii_letters) + "'"
    lo, hi = table[kind]
    if kind in ("float", "double"):
        prec = rng.choice([1, 2])
        return f"{round(rng.uniform(lo, hi), prec):.{prec}f}" + SUFFIX.get(kind, "")
    value = rng.randint(lo, hi)
    if kind == "short" and rng.random() < 0.25:
        value = -rng.randint(1, 999)
    return str(value) + SUFFIX.get(kind, "")


def init_phrase_cn(kind: str, lit: str) -> str:
    """初始值的完整中文说法（用于注释答案）。

    lit 可以是明文，也可以是 `<|PKG_NUM|>` 标记 —— 本函数只做"把值放进句子"
    的排版，不关心里面是明文还是标记。打包版答案就靠这个复用。
    """
    if kind == "char":
        return f"初始值是字符 {lit}"
    if kind == "float":
        return f"初始值是 {lit}（单精度浮点字面量）"
    if kind == "double":
        return f"初始值是 {lit}（双精度浮点字面量）"
    if kind == "long":
        return f"初始值是 {lit}（long 字面量）"
    if kind == "unsigned":
        return f"初始值是 {lit}（无符号字面量）"
    return f"初始值是 {lit}"


# ================= 声明结构 =================

class Decl:
    """一条声明的结构化描述；代码与中文注释都从这里派生，保证双向一致。"""

    def __init__(self, type_c: str, type_cn: str, kind: str, form: str,
                 prefix: str, form_cn: str, name: str, lit: str | None):
        self.type_c, self.type_cn, self.kind = type_c, type_cn, kind
        self.form, self.prefix, self.form_cn = form, prefix, form_cn
        self.name, self.lit = name, lit

    @property
    def code(self) -> str:
        """明文 C 代码（参考译法，不用于训练）。"""
        if self.lit is None:
            return f"{self.prefix}{self.type_c} {self.name};"
        return f"{self.prefix}{self.type_c} {self.name} = {self.lit};"

    @property
    def code_packed(self) -> str:
        """打包版 C 代码：名字/值换成原子包标记 —— 反向训练的 target。

        解包后与 `code` 逐字相同；模型只需输出固定的几个标记，不复制载荷。
        """
        if self.lit is None:
            return f"{self.prefix}{self.type_c} {PKG_NAME};"
        return f"{self.prefix}{self.type_c} {PKG_NAME} = {PKG_NUM};"

    # ---------- 明文版（供人类阅读、供审计核对） ----------
    @property
    def comment_plain(self) -> str:
        """中文注释前半句，名字与值都是明文。"""
        if self.lit is None:
            return f"声明一个{self.type_cn}{self.form_cn} {self.name}，没有给出初始值"
        return (f"声明一个{self.type_cn}{self.form_cn} {self.name}，"
                f"{init_phrase_cn(self.kind, self.lit)}")

    @property
    def full_answer_plain(self) -> str:
        """明文完整答案（参考译文，不用于训练）。"""
        return self._assemble_answer(self.comment_plain)

    # ---------- 打包版（真正用于训练 / 期望答案） ----------
    # 关键：模型只输出 `<|PKG_NAME|>` 与 `<|PKG_NUM|>` 这两个**静态标记**，
    # 不逐字符复制 `286020U`。程序解包后把标记替换回载荷，得到明文答案。
    # 这是 65M 小模型能学会的形式；明文版逼它做字符级复制，必然失败。
    @property
    def comment_packed(self) -> str:
        """中文注释前半句，名字/值替换成原子包标记。"""
        if self.lit is None:
            return f"声明一个{self.type_cn}{self.form_cn} {PKG_NAME}，没有给出初始值"
        return (f"声明一个{self.type_cn}{self.form_cn} {PKG_NAME}，"
                f"{init_phrase_cn(self.kind, PKG_NUM)}")

    @property
    def full_answer_packed(self) -> str:
        """打包版完整答案 —— 训练用的 target，零字符复制。"""
        return self._assemble_answer(self.comment_packed)

    def _assemble_answer(self, comment: str) -> str:
        """把注释前半句与形态说明拼成完整答案（明文/打包共用同一套拼接逻辑）。"""
        base = comment + "。"
        if self.form == "const":
            return base + "const 表示这个值初始化后不能再修改。"
        if self.form == "static_const":
            return base + "static const 表示程序运行期间保持不变的静态常量。"
        if self.lit is None:
            return base + "未初始化的变量在使用前应先赋值。"
        return base

    @property
    def rev_question_desc(self) -> str:
        """反向任务的中文描述（名字/数值会由前端打成原子包）。"""
        if self.lit is None:
            return f"声明一个{self.type_cn}{self.form_cn} {self.name}，不给初始值"
        return f"声明一个{self.type_cn}{self.form_cn} {self.name}，初始值是 {self.lit}"


def make_decl(rng: random.Random, names_lower: list[str], names_upper: list[str],
              space: str, type_idx: int | None = None,
              form_idx: int | None = None) -> Decl:
    type_c, type_cn, kind = TYPES[type_idx if type_idx is not None
                                  else rng.randrange(len(TYPES))]
    form, prefix, form_cn, has_init = FORMS[form_idx if form_idx is not None
                                            else rng.randrange(len(FORMS))]
    # 常量形态习惯用大写名，普通变量用小写名
    name = rng.choice(names_upper if form in ("const", "static_const")
                      else names_lower)
    lit = gen_value(kind, rng, space) if has_init else None
    return Decl(type_c, type_cn, kind, form, prefix, form_cn, name, lit)


# ================= 样本生成（全部走真实前端） =================

FWD_QUESTIONS = [
    "用一句中文描述这行声明语句。",
    "这行代码声明了什么？",
    "说明这个声明的类型、名字和初始值。",
]
REV_QUESTIONS = [
    "写出对应的 C 声明语句。",
    "把这个描述写成一行 C 代码。",
]


def fwd_sample(decl: Decl, rng: random.Random) -> dict:
    """正向：代码（嵌套打包）→ 中文注释（**答案是打包版**）。

    target 里用 `<|PKG_NAME|>`/`<|PKG_NUM|>` 指向已有的原子包，不含明文载荷。
    模型只需学会"在句子的哪个位置放出哪个标记"，零字符复制。
    """
    code = decl.code
    cache = SessionCache()
    packaged = package_request(f"解释代码：{code}", cache, nested=True)
    handles = packaged["child_handles"]
    if len(handles) != 1:
        raise ValueError(f"声明应切成一个语句子包：{code!r} -> {handles}")
    prompt = expert_input_nested(cache, handles[0], rng.choice(FWD_QUESTIONS))
    atoms = [cache.atoms[a] for a in packaged["atom_handles"]]
    target = decl.full_answer_packed
    return {
        "prompt": prompt, "target": target, "text": prompt + target,
        # answer_plain 只作参考与审计用（解包后应当与它一致），不参与训练
        "answer_plain": decl.full_answer_plain,
        "expected_unpacked": decl.full_answer_plain,
        "direction": "code_to_comment", "form": decl.form, "type_c": decl.type_c,
        "code": code, "name": decl.name, "lit": decl.lit,
        "atoms": [{"marker": a.marker, "payload": a.payload} for a in atoms],
    }


def rev_sample(decl: Decl, rng: random.Random) -> dict:
    """反向：中文描述（名字/数值打包）→ 代码（**目标也是打包版**）。

    目标是 `const unsigned int <|PKG_NAME|> = <|PKG_NUM|>;`，
    模型只输出类型/形态这些固定词 + 两个标记；解包后得到完整明文声明。
    """
    desc = decl.rev_question_desc
    cache = SessionCache()
    packaged = package_description(desc, cache)
    prompt = expert_input_description(cache, packaged["child_handle"],
                                      rng.choice(REV_QUESTIONS))
    target = decl.code_packed
    atoms = [cache.atoms[a] for a in packaged["atom_handles"]]
    return {
        "prompt": prompt, "target": target, "text": prompt + target,
        "answer_plain": decl.code,
        "expected_unpacked": decl.code,
        "direction": "comment_to_code", "form": decl.form, "type_c": decl.type_c,
        "code": decl.code, "name": decl.name, "lit": decl.lit,
        "marked_view": packaged["marked_view"],
        "atoms": [{"marker": a.marker, "payload": a.payload} for a in atoms],
    }


def semicolon_samples(rng: random.Random) -> list[dict]:
    """分号语义：一句多变量 / 两句 / 知识问答，答案由分号计数程序化派生。"""
    rows: list[dict] = []

    # 1) 一句声明两个变量（一个分号）：两个变量必须同类型，这才是逗号声明的语义
    ti = rng.randrange(len(TYPES))
    type_c, type_cn, kind = TYPES[ti]
    n1, n2 = rng.sample(TRAIN_NAMES_LOWER, 2)
    v1 = gen_value(kind, rng, "train")
    v2 = gen_value(kind, rng, "train")
    code = f"{type_c} {n1} = {v1}, {n2} = {v2};"
    n_semi = code.count(";")
    rows.append({
        "prompt": f"问题：下面的代码是几句话？\n代码：{code}\n回答：",
        "target": (f"是 {n_semi} 句话。C 语言里分号 ; 标志一句结束；"
                   f"这句用一个分号收尾，在同一个 {type_cn} 类型下"
                   f"一次声明了 {n1} 和 {n2} 两个变量。"),
        "text": "", "direction": "semicolon", "form": "multi_var_one_stmt",
        "type_c": type_c, "code": code, "name": None, "lit": None,
        "n_semicolons": n_semi,
    })

    # 2) 两个独立声明（两个分号=两句）：类型可以不同
    d3 = make_decl(rng, TRAIN_NAMES_LOWER, TRAIN_NAMES_UPPER, "train", form_idx=0)
    d4 = make_decl(rng, TRAIN_NAMES_LOWER, TRAIN_NAMES_UPPER, "train", form_idx=0)
    if d3.name != d4.name:
        code = f"{d3.code} {d4.code}"
        n_semi = code.count(";")
        rows.append({
            "prompt": f"问题：下面的代码是几句话？\n代码：{code}\n回答：",
            "target": (f"是 {n_semi} 句话。每个分号 ; 各结束一句："
                       f"第一句声明 {d3.name}，第二句声明 {d4.name}。"),
            "text": "", "direction": "semicolon", "form": "two_stmts",
            "type_c": d3.type_c, "code": code, "name": None, "lit": None,
            "n_semicolons": n_semi,
        })

    # 3) 分号知识问答（固定知识 + 措辞变体）
    knowledge = [
        ("在 C 语言里，分号 ; 表示什么？",
         "分号标志一条语句的结束。编译器靠它知道一句话到哪里为止；"
         "漏写分号是最常见的编译错误之一。"),
        ("int a = 1, b = 2; 结尾为什么只有一个分号？",
         "因为这是一句声明语句，逗号只是在一个类型下声明多个变量；"
         "整句用一个分号结束。"),
        ("int a = 1; int b = 2; 为什么有两个分号？",
         "因为这是两句独立的声明语句；每句各自用一个分号结束。"),
        ("写声明语句时什么时候加分号？",
         "每写完一条完整声明就加一个分号，表示这句结束；"
         "一条语句里声明多个变量也只需要末尾一个分号。"),
    ]
    for q, a in knowledge:
        rows.append({
            "prompt": f"问题：{q}\n回答：", "target": a, "text": "",
            "direction": "semicolon", "form": "knowledge", "type_c": None,
            "code": None, "name": None, "lit": None, "n_semicolons": None,
        })

    for r in rows:
        r["text"] = r["prompt"] + r["target"]
    return rows


def build(seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    train_rows: list[dict] = []
    seen_texts: set[str] = set()

    def add(row: dict) -> bool:
        """加入训练集；重复 text 返回 False（调用方据此计数）。"""
        if row["text"] in seen_texts:
            return False
        seen_texts.add(row["text"])
        train_rows.append(row)
        return True

    # 正向+反向：7 类型 × 4 形态 × PER_COMBO 组（名字/值随机）。
    # 两个方向各自计数：重复 text 被 add() 丢弃时不虚增计数，
    # 否则每个组合实际会略少于 PER_COMBO（旧版缺 32 条就是这里）。
    pool_lower = TRAIN_NAMES_LOWER
    pool_upper = TRAIN_NAMES_UPPER
    for ti, (type_c, type_cn, kind) in enumerate(TYPES):
        for fi, (form, prefix, form_cn, has_init) in enumerate(FORMS):
            made_fwd = made_rev = 0
            attempt = 0
            limit = PER_COMBO * 200
            while (made_fwd < PER_COMBO or made_rev < PER_COMBO) and attempt < limit:
                attempt += 1
                # var_uninit 形态没有初始值，码空间只有"类型×名字"，
                # 名字池大小决定上限；不足时换名字池顺序重排（不新增名字，保持隔离）
                decl = make_decl(rng, pool_lower, pool_upper, "train",
                                 type_idx=ti, form_idx=fi)
                if made_fwd < PER_COMBO and add(fwd_sample(decl, rng)):
                    made_fwd += 1
                if made_rev < PER_COMBO and add(rev_sample(decl, rng)):
                    made_rev += 1
            if made_fwd < PER_COMBO or made_rev < PER_COMBO:
                raise RuntimeError(
                    f"组合 ({type_c}, {form}) 生成不收敛：正向 {made_fwd}/{PER_COMBO}、"
                    f"反向 {made_rev}/{PER_COMBO}（尝试 {attempt} 次）。"
                    f"无初值形态的码空间受名字池大小限制，需扩充名字池或降低 PER_COMBO。")

    # 分号语义：固定知识 4 条 + 随机对比样本补足
    semi_fixed = semicolon_samples(rng)
    for row in semi_fixed:
        add(row)
    made = len([r for r in train_rows if r["direction"] == "semicolon"])
    attempt = 0
    while made < N_SEMICOLON and attempt < N_SEMICOLON * 60:
        attempt += 1
        before = made
        for row in semicolon_samples(rng):
            add(row)
        made = len([r for r in train_rows if r["direction"] == "semicolon"])
        if made == before and attempt > N_SEMICOLON * 30:
            # 随机组合已饱和（知识问答等固定行会反复命中重复），按实际数量收尾
            break
    if made < N_SEMICOLON:
        raise RuntimeError(
            f"分号语义样本不足：{made}/{N_SEMICOLON}。"
            f"随机组合池已饱和，需扩充 TRAIN_NAMES_LOWER 或类型表。")
    rng.shuffle(train_rows)

    # ---- 评估集：未见名字（大写常量名 + 小写变量名各半），值域也分离 ----
    eval_rows: list[dict] = []
    eval_decls: list[Decl] = []
    attempt = 0
    while len(eval_decls) < EVAL_FWD + EVAL_REV and attempt < (EVAL_FWD + EVAL_REV) * 60:
        attempt += 1
        ti = rng.randrange(len(TYPES))
        fi = rng.randrange(len(FORMS))
        names_lo, names_up = EVAL_NAMES_LOWER, EVAL_NAMES_UPPER
        decl = make_decl(rng, names_lo, names_up, "eval",
                         type_idx=ti, form_idx=fi)
        if any(decl.code == d.code for d in eval_decls):
            continue
        eval_decls.append(decl)
    for decl in eval_decls[:EVAL_FWD]:
        row = fwd_sample(decl, rng)
        row["identifier_group"] = "unseen_name"
        eval_rows.append(row)
    for decl in eval_decls[EVAL_FWD:EVAL_FWD + EVAL_REV]:
        row = rev_sample(decl, rng)
        row["identifier_group"] = "unseen_name"
        eval_rows.append(row)
    # 分号评估：未见名字的两句/一句多变量对比。
    # one_stmt 是逗号声明，两个变量必须同类型，所以第二个字面量
    # 按 da 的类型重新生成（直接用 db.lit 会产生 char x='A', y=1.5; 这类类型错误代码）
    semi_eval: list[dict] = []
    attempt = 0
    while len(semi_eval) < EVAL_SEMI and attempt < EVAL_SEMI * 60:
        attempt += 1
        da = make_decl(rng, EVAL_NAMES_LOWER, EVAL_NAMES_UPPER, "eval",
                       type_idx=rng.randrange(len(TYPES)), form_idx=0)
        db = make_decl(rng, EVAL_NAMES_LOWER, EVAL_NAMES_UPPER, "eval",
                       type_idx=rng.randrange(len(TYPES)), form_idx=0)
        if da.name == db.name:
            continue
        lit_b_same_type = gen_value(da.kind, rng, "eval")
        one_stmt = (f"{da.type_c} {da.name} = {da.lit}, "
                    f"{db.name} = {lit_b_same_type};")
        two_stmt = f"{da.code} {db.code}"
        for code, answer in (
            (one_stmt, f"是 1 句话。分号 ; 标志一句结束；这句一次声明了 {da.name} 和 {db.name} 两个变量。"),
            (two_stmt, f"是 2 句话。每个分号各结束一句：先声明 {da.name}，再声明 {db.name}。"),
        ):
            if len(semi_eval) < EVAL_SEMI:
                prompt = f"问题：下面的代码是几句话？\n代码：{code}\n回答："
                semi_eval.append({
                    "prompt": prompt, "target": answer, "text": prompt + answer,
                    "direction": "semicolon", "form": "eval", "type_c": da.type_c,
                    "code": code, "name": None, "lit": None,
                    "n_semicolons": code.count(";"),
                    "identifier_group": "unseen_name",
                })
    eval_rows.extend(semi_eval)

    return {"decl_train": train_rows, "decl_eval": eval_rows}


# ================= 审计 =================

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has_plain_payload(text: str, payload: str) -> bool:
    """target 里是否以「完整载荷」的形式出现了明文（按标识符/字面量边界判断）。

    不能用 `payload in text` 做子串检查：单字母变量名 `i` 是 `int` 的子串，
    会产生大量误报。这里要求载荷前后都不是 [A-Za-z0-9_]，即它是独立出现的一个词。
    中文不是 ASCII 字母数字，因此中文与载荷相邻时仍能正确判定为独立出现。
    """
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(payload) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def audit(data: dict[str, list[dict]], tokenizer_dir: Path,
          data_dir: Path, seed: int) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
    train_rows, eval_rows = data["decl_train"], data["decl_eval"]

    # 1) text 唯一且非空
    texts = [r["text"] for r in train_rows + eval_rows]
    if any(not t.strip() for t in texts):
        raise RuntimeError("存在空 text")
    if len(texts) != len(set(texts)):
        dupes = sorted({t for t in texts if texts.count(t) > 1})
        raise RuntimeError(f"存在重复样本：{dupes[:1]}")

    # 1b) text == prompt + target，且 prompt/target 分开编码等于整条编码。
    #     后者是 03 训练器 --target-mask 的前提：拼接处一旦被 BPE 合并，
    #     监督位置就会整体错位（静默算错 loss），必须在数据侧保证。
    for r in train_rows + eval_rows:
        if r["text"] != r["prompt"] + r["target"]:
            raise RuntimeError(f"text 不等于 prompt+target：{r['text'][:60]!r}")
        prompt_ids = tokenizer.encode(r["prompt"], add_special_tokens=False).ids
        target_ids = tokenizer.encode(r["target"], add_special_tokens=False).ids
        whole_ids = tokenizer.encode(r["text"], add_special_tokens=False).ids
        if whole_ids != prompt_ids + target_ids:
            raise RuntimeError(
                f"prompt/target token 边界不一致（{len(prompt_ids)}+{len(target_ids)} "
                f"vs {len(whole_ids)}）：{r['text'][:80]!r}")

    # 2) 正向样本：prompt 含语句子包与嵌套原子包，载荷与结构一致
    for r in train_rows + eval_rows:
        if r["direction"] != "code_to_comment":
            continue
        if PACKAGE not in r["prompt"] or r["code"] not in r["prompt"]:
            raise RuntimeError(f"正向 prompt 缺语句子包或载荷：{r['code']}")
        atom_payloads = {a["payload"] for a in r["atoms"]}
        if r["name"] not in atom_payloads:
            raise RuntimeError(f"正向原子包缺变量名：{r['name']}")
        if r["lit"] is not None and r["lit"] not in atom_payloads:
            raise RuntimeError(f"正向原子包缺数值：{r['lit']}")
        if PKG_NAME not in r["prompt"] or (r["lit"] and PKG_NUM not in r["prompt"]):
            raise RuntimeError(f"正向 prompt 缺原子包标记：{r['code']}")
        # 4) 关键审计：target 必须是打包版 —— 不得出现明文变量名或数值字面量。
        #    这是 copy wall 的根因防线：target 一旦写明文，模型就被迫逐字符复制，
        #    65M 小模型必然做不到（实测 5005931L -> 50053L、FARADAY -> SPAN）。
        #    注意：检查时要排除 prompt 里无关的普通词，只查本条的 name/lit。
        if has_plain_payload(r["target"], r["name"]):
            raise RuntimeError(
                f"正向 target 含明文变量名 {r['name']!r}（应为 {PKG_NAME}）：{r['target']}")
        if r["lit"] is not None and has_plain_payload(r["target"], r["lit"]):
            raise RuntimeError(
                f"正向 target 含明文数值 {r['lit']!r}（应为 {PKG_NUM}）：{r['target']}")
        if PKG_NAME not in r["target"]:
            raise RuntimeError(f"正向 target 缺 {PKG_NAME} 标记：{r['target']}")
        if r["lit"] is not None and PKG_NUM not in r["target"]:
            raise RuntimeError(f"正向 target 缺 {PKG_NUM} 标记：{r['target']}")

        # 5) 解包正确性：把 target 里的标记替换回本条载荷，必须与明文期望逐字相同。
        #    这保证"打包训练 + 程序解包"这条链路能真正还原答案。
        unpacked = r["target"].replace(PKG_NAME, r["name"])
        if r["lit"] is not None:
            unpacked = unpacked.replace(PKG_NUM, r["lit"])
        if unpacked != r["expected_unpacked"]:
            raise RuntimeError(
                f"解包结果与明文期望不一致：\n  解包 {unpacked!r}\n  期望 {r['expected_unpacked']!r}")

    # 3) 反向样本：marked_view 不得泄漏真实载荷；原子包解包清单必须含载荷
    for r in train_rows + eval_rows:
        if r["direction"] != "comment_to_code":
            continue
        if r["name"] in r["marked_view"] or (r["lit"] and r["lit"] in r["marked_view"]):
            raise RuntimeError(f"反向 marked_view 泄漏载荷：{r['marked_view']}")
        atom_payloads = {a["payload"] for a in r["atoms"]}
        if r["name"] not in atom_payloads:
            raise RuntimeError(f"反向原子包缺变量名：{r['name']}")
        if r["lit"] is not None and r["lit"] not in atom_payloads:
            raise RuntimeError(f"反向原子包缺数值：{r['lit']}")
        # target 必须是合法形态的打包声明：以分号结束、含 <|PKG_NAME|>、
        # 且**不含明文名字/数值**（与正向同一条防线：不许字符级复制）。
        if not r["target"].endswith(";"):
            raise RuntimeError(f"反向 target 不是完整声明：{r['target']}")
        if PKG_NAME not in r["target"]:
            raise RuntimeError(f"反向 target 缺 {PKG_NAME}：{r['target']}")
        if has_plain_payload(r["target"], r["name"]):
            raise RuntimeError(
                f"反向 target 含明文变量名 {r['name']!r}（应为 {PKG_NAME}）：{r['target']}")
        if r["lit"] is not None:
            if PKG_NUM not in r["target"]:
                raise RuntimeError(f"反向 target 缺 {PKG_NUM}：{r['target']}")
            if has_plain_payload(r["target"], r["lit"]):
                raise RuntimeError(
                    f"反向 target 含明文数值 {r['lit']!r}（应为 {PKG_NUM}）：{r['target']}")
        # 解包正确性：标记替换回载荷后必须与明文代码逐字相同
        unpacked_rev = r["target"].replace(PKG_NAME, r["name"])
        if r["lit"] is not None:
            unpacked_rev = unpacked_rev.replace(PKG_NUM, r["lit"])
        if unpacked_rev != r["expected_unpacked"]:
            raise RuntimeError(
                f"反向解包结果与明文期望不一致：\n  解包 {unpacked_rev!r}\n"
                f"  期望 {r['expected_unpacked']!r}")

    # 4) 分号样本：答案里的句数必须等于代码分号数
    for r in train_rows + eval_rows:
        if r["direction"] != "semicolon" or r["code"] is None:
            continue
        n = r["code"].count(";")
        if f"是 {n} 句" not in r["target"]:
            raise RuntimeError(f"分号答案与代码不一致（{n} 个分号）：{r['target']}")

    # 5) eval 未见名字对训练文本零泄漏（子串级，最严格形式）
    #    注：不做"训练名 ⊂ eval 名"的交叉检查——C 单字母变量名（a/i/n）天然是
    #    alpha/sigma/energy 的子串，该检查与单字母名不兼容；真正要保证的是
    #    eval 标识符本身没在训练文本里出现过，即下面的子串检查 + 名字池零交集。
    train_blob = "\n".join(r["text"] for r in train_rows)
    eval_names = {r["name"] for r in eval_rows if r.get("name")}
    leaks = sorted(w for w in eval_names if w in train_blob)
    if leaks:
        raise RuntimeError(f"eval 名字泄漏进训练集：{leaks}")
    pool_overlap = (set(EVAL_NAMES_LOWER) | set(EVAL_NAMES_UPPER)) & (
        set(TRAIN_NAMES_LOWER) | set(TRAIN_NAMES_UPPER))
    if pool_overlap:
        raise RuntimeError(f"eval 名字池与训练池相交：{sorted(pool_overlap)}")

    # 6) 复现性：同种子重建逐字节一致
    rebuilt = build(seed)
    for split in ("decl_train", "decl_eval"):
        if json.dumps(rebuilt[split], ensure_ascii=False) != json.dumps(data[split], ensure_ascii=False):
            raise RuntimeError(f"复现校验失败：{split}")

    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False).ids)
               for r in train_rows]
    dirs: dict[str, int] = {}
    forms: dict[str, int] = {}
    for r in train_rows:
        dirs[r["direction"]] = dirs.get(r["direction"], 0) + 1
        forms[r["form"]] = forms.get(r["form"], 0) + 1
    eval_dirs: dict[str, int] = {}
    for r in eval_rows:
        eval_dirs[r["direction"]] = eval_dirs.get(r["direction"], 0) + 1
    files = {f"decl_bilingual_v1_{s}.jsonl":
             sha256_file(data_dir / f"decl_bilingual_v1_{s}.jsonl")
             for s in ("decl_train", "decl_eval")}
    return {
        "seed": seed,
        "markers": {"statement": PACKAGE, "num": PKG_NUM, "name": PKG_NAME},
        "train_count": len(train_rows),
        "train_directions": dirs,
        "train_forms": forms,
        "train_tokens": sum(lengths),
        "train_avg_tokens": round(sum(lengths) / len(lengths), 1),
        "train_max_tokens": max(lengths),
        "eval_count": len(eval_rows),
        "eval_directions": eval_dirs,
        "eval_names_unseen": sorted(eval_names),
        "eval_name_leaks": len(leaks),
        "reverse_payload_hidden": True,
        "semicolon_answers_verified": True,
        "duplicate_texts": 0,
        "reproducibility_ok": True,
        "file_sha256": files,
        "semantics": ("code -> nested atoms (PKG_NUM/PKG_NAME) -> Chinese comment; "
                      "Chinese description -> packed atoms -> C declaration; "
                      "semicolon = end of one statement"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 C/C++ 声明基础双向数据（代码⇄注释+分号语义）；不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR,
                        help="Tokenizer 目录")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子（默认 20261014）")
    args = parser.parse_args()

    data = build(args.seed)
    for name, rows in data.items():
        write_jsonl(args.data_dir / f"decl_bilingual_v1_{name}.jsonl", rows)
    report = audit(data, args.tokenizer, args.data_dir, args.seed)
    (args.data_dir / "decl_bilingual_v1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
