# -*- coding: utf-8 -*-
"""准备通用底座种子 v0、代码 Package 路由数据并审计专用 token；不启动模型训练。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AddedToken, AutoTokenizer


PACKAGE = "<|PKG_CODE|>"
DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")

# 路由训练只提供意图与 package 类型；原始代码仅作评审元数据，不进入 text。
ROUTER_TRAIN = [
    ("解释这段代码", "CODE_EXPLAIN"), ("说明代码每一部分的意思", "CODE_EXPLAIN"),
    ("分析代码中的变量和数值", "CODE_EXPLAIN"), ("告诉我标识符代表什么", "CODE_EXPLAIN"),
    ("解释代码里的英文术语", "CODE_EXPLAIN"), ("这行程序声明了什么", "CODE_EXPLAIN"),
    ("逐个说明关键字、变量名和值", "CODE_EXPLAIN"), ("说明等号两边的关系", "CODE_EXPLAIN"),
    ("把代码含义用中文讲清楚", "CODE_EXPLAIN"), ("解释 variable 和 identifier 的区别", "CODE_EXPLAIN"),
    ("看一下这段程序在表达什么", "CODE_EXPLAIN"), ("我想理解这条语句", "CODE_EXPLAIN"),
    ("请讲解代码中的类型和初始化", "CODE_EXPLAIN"), ("帮我理解这段 C 程序", "CODE_EXPLAIN"),
    ("代码里的 name 和 value 分别指什么", "CODE_EXPLAIN"), ("解释这个声明语句的结构", "CODE_EXPLAIN"),
    ("这段代码如何表示一个变量", "CODE_EXPLAIN"), ("说明每个代码符号的作用", "CODE_EXPLAIN"),
    ("解释这段代码，不要执行", "CODE_EXPLAIN"), ("请用中文说明程序片段", "CODE_EXPLAIN"),
    ("运行这段程序并给出输出", "UNKNOWN"), ("修改代码以修复编译错误", "UNKNOWN"),
    ("把这段代码翻译成另一种语言", "UNKNOWN"), ("仅原样复述代码，不作解释", "UNKNOWN"),
    ("计算这段程序运行所需的时间复杂度", "UNKNOWN"), ("为代码生成完整测试套件", "UNKNOWN"),
    ("解释中文词语变量，不涉及代码", "UNKNOWN"), ("总结这段设计文档", "UNKNOWN"),
    ("在代码字符串中查找单词 value", "UNKNOWN"), ("请格式化这段程序", "UNKNOWN"),
]

ROUTER_EVAL = [
    ("我看不懂这行代码，讲讲各部分含义", "CODE_EXPLAIN"),
    ("identifier 在这里指的是什么？", "CODE_EXPLAIN"),
    ("分析声明里的变量名、类型和值", "CODE_EXPLAIN"),
    ("请不要运行，只说明语句的意义", "CODE_EXPLAIN"),
    ("把这段代码实际执行一下", "UNKNOWN"),
    ("帮我把程序改成使用浮点数", "UNKNOWN"),
    ("代码中的文字请逐字复制出来", "UNKNOWN"),
    ("解释 variable 这个英文单词的普通词义", "UNKNOWN"),
    ("给程序加上注释并重构", "UNKNOWN"),
    ("检查代码能不能编译通过", "UNKNOWN"),
    ("解释这一句：`int index = 17;`", "CODE_EXPLAIN"),
    ("为这段代码画流程图", "UNKNOWN"),
]

TERMS = [
    ("变量", "variable", "可通过名称在程序中引用、并在运行中保存状态的对象。"),
    ("标识符", "identifier", "源代码中给变量等程序实体使用的合法名称。"),
    ("名称", "name", "人或程序员为实体选取的称呼；在代码里通常由标识符表示。"),
    ("类型", "type", "规定数据如何解释，以及允许哪些操作的数据分类。"),
    ("值", "value", "某个变量在特定时刻保存的数据内容。"),
    ("关键字", "keyword", "由语言规定、具有固定语法作用的保留词。"),
    ("字面量", "literal", "直接写在源代码中的常量表示，例如 17 或 'A'。"),
    ("声明", "declaration", "向编译器说明一个名字及其类型等信息。"),
    ("初始化", "initialization", "创建对象时为它提供初始值。"),
    ("赋值", "assignment", "把一个值存入已存在的可修改对象。"),
    ("整数", "integer", "没有小数部分的整数数值类别。"),
    ("字符", "character", "文本或字符数据中的单个字符；C 的 char 是基本字符类型。"),
    ("浮点数", "floating-point number", "用浮点格式近似表示的实数数值。"),
    ("有符号", "signed", "整数类型可表示负值、零和正值（具体范围依类型与实现）。"),
    ("无符号", "unsigned", "整数类型不表示负数，范围用于零及正值。"),
]

# 小型“通用底座种子”包含双向术语映射、概念关系与 Package 之后的代码解释。
CODE_TRAIN = [
    ("int i = 0;", "解释每个部分", "int 是 C 整数类型关键字；i 是 identifier（标识符），此处命名一个 variable（变量）；= 后的 0 是整数 literal（字面量），用作初始化 value（初始值）。"),
    ("int count = 3;", "count 是什么？", "count 是变量名，也是标识符；int 表示整数类型，3 是初始值。"),
    ("char initial = 'A';", "initial 和 'A' 分别是什么？", "initial 是 char 类型变量的标识符；char 是字符类型关键字；'A' 是字符常量，用作初始值。"),
    ("float ratio = 0.5f;", "解释类型和字面量", "ratio 是变量标识符；float 表示单精度浮点类型；0.5f 是浮点字面量并作为初始值。"),
    ("double average = 3.75;", "average 保存什么？", "average 是 double 类型变量名；double 是双精度浮点类型，3.75 是浮点字面量，作为初始值。"),
    ("unsigned int attempts = 4U;", "unsigned 和 4U 分别表示什么？", "unsigned int 是无符号整数类型；attempts 是变量标识符；4U 是无符号整数字面量，用作初始值。"),
    ("long distance = 12L;", "long 是否一定是 64 位？", "不一定。long 是至少不窄于 int 的整数类型，其具体位宽由 C 实现决定；12L 是 long 类型整数字面量。"),
    ("short level = 7;", "level 的类型是什么？", "level 是变量名，类型为 short int（短整数类型）；7 是整数初始值。"),
    ("int total; total = 15;", "区分 declaration 和 assignment", "int total; 是声明；后面的 total = 15; 是对已声明变量赋值。assignment 是赋值，不是声明时初始化。"),
    ("int width = 640; int height = 480;", "有几个变量？", "有两个变量，名称分别是 width 和 height；两者都是 int 类型，初始值分别为 640 和 480。"),
    ("char grade = 'B';", "B 是变量名吗？", "不是。grade 是变量标识符；'B' 是字符字面量。char 是字符类型关键字。"),
    ("float rate = 1.25f;", "rate 的 type、name、value 是什么？", "type 是 float；name/identifier 是 rate；value 是由浮点字面量 1.25f 提供的初始值。"),
    ("double radius = 2.5;", "radius 是 identifier 吗？", "是。radius 是标识符，在这里命名一个 double 类型变量；2.5 是浮点初始值。"),
    ("unsigned int mask = 8U;", "mask 是 C 关键字吗？", "不是。mask 是用户选择的变量名/标识符；unsigned 和 int 是类型说明符；8U 是无符号整数初始值。"),
    ("long timestamp = 42L;", "42L 是什么？", "42L 是 long 类型的整数字面量，用作 timestamp 的初始值；timestamp 是变量名。"),
    ("int item_count = 6;", "下划线把名称拆成两个变量了吗？", "没有。item_count 是一个完整标识符，用作一个变量名；它的类型是 int，初始值是 6。"),
]

BASE_EVAL = [
    ("int index = 17;", "index 是什么，17 是什么？", ["index 是标识符/变量名", "17 是整数字面量/初始值", "int 是整数类型"]),
    ("char symbol = 'Z';", "symbol 和 'Z' 分别是什么？", ["symbol 是标识符/变量名", "char 是字符类型", "'Z' 是字符常量/初始值"]),
    ("float temperature = 2.75f;", "temperature 的类型和初始值是什么？", ["float 类型", "2.75f 是浮点字面量/初始值"]),
    ("double probability = 0.125;", "probability 是值还是变量名？", ["probability 是标识符/变量名", "double 类型", "0.125 是浮点初始值"]),
    ("unsigned int capacity = 32U;", "capacity 是关键字吗？", ["capacity 是用户命名的标识符", "unsigned int 是无符号整数类型", "32U 是初始值"]),
    ("int point; point = 23;", "哪一部分是声明，哪一部分是后续赋值？", ["int point 是声明", "point = 23 是赋值", "不是声明时初始化"]),
    ("short sample_count = 12;", "sample_count 是几个 identifier？", ["一个完整标识符", "short 类型", "12 是初始值"]),
    ("long file_size = 900L;", "能否断定 long 占 64 位？", ["不能断定", "位宽依赖 C 实现", "long 是整数类型"]),
    ("变量的英文术语是什么？", "双向术语", ["variable"]),
    ("What does identifier mean in Chinese?", "双向术语", ["标识符", "代码中的名称"]),
    ("Explain the difference between a variable name and its value.", "英文概念理解", ["名称用于引用变量", "值是变量保存的数据"]),
    ("C 代码里的 keyword 和 identifier 有什么不同？", "中英术语辨析", ["keyword 是保留关键字", "identifier 是程序实体的名称"]),
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_data() -> dict[str, list[dict]]:
    train_payloads = [row[0] for row in CODE_TRAIN]
    eval_payloads = [row[0] for row in BASE_EVAL if "=" in row[0] and ";" in row[0]]

    def router_rows(items: list[tuple[str, str]], payloads: list[str]) -> list[dict]:
        rows = []
        for index, (intent, label) in enumerate(items):
            payload = payloads[index % len(payloads)]
            source_request = f"用户：{intent}：`{payload}`"
            controller_view = f"请求：{intent}\n路由输入：{PACKAGE}"
            if payload in controller_view:
                raise RuntimeError("路由视图泄漏了代码载荷")
            rows.append({
                "source_request": source_request,
                "payload": payload,
                "controller_view": controller_view,
                "text": controller_view,
                "label": label,
            })
        return rows

    router_train = router_rows(ROUTER_TRAIN, train_payloads)
    router_eval = router_rows(ROUTER_EVAL, eval_payloads)
    base_train = []
    for chinese, english, definition in TERMS:
        base_train.append({"text": f"知识：中文术语“{chinese}”对应英文术语“{english}”。{definition}", "kind": "bilingual_definition"})
        base_train.append({"text": f"问题：请用中文解释 {english}。\n回答：{chinese}（{english}）是指：{definition}", "kind": "english_to_chinese"})
        base_train.append({"text": f"问题：{chinese} 在编程中的英文术语是什么？\n回答：{english}。", "kind": "chinese_to_english"})
    for code, question, answer in CODE_TRAIN:
        sample = (
            f"用户任务：解释一段 C 代码。\n"
            f"打包器识别代码区域：{PACKAGE}\n"
            f"路由器可见输入：解释代码 {PACKAGE}\n"
            f"路由选择：CODE_EXPLAIN\n"
            f"解包交给通用底座的载荷：语言=C；原始代码={code}\n"
            f"问题：{question}\n底座回答：{answer}"
        )
        base_train.append({"text": sample, "kind": "packaged_code_semantics"})
    base_eval = [
        ({"input": (
              f"用户任务：解释一段 C 代码。\n"
              f"打包器识别代码区域：{PACKAGE}\n"
              f"路由器可见输入：解释代码 {PACKAGE}\n"
              f"路由选择：CODE_EXPLAIN\n"
              f"解包交给通用底座的载荷：语言=C；原始代码={code}\n"
              f"问题：{question}\n底座回答："),
          "code": code, "question": question, "must_include": expected}
         if ";" in code else
         {"input": f"问题：{code}", "code": None, "question": code, "must_include": expected})
        for code, question, expected in BASE_EVAL
    ]
    return {"router_train": router_train, "router_eval": router_eval,
            "base_train": base_train, "base_eval": base_eval}


def audit(tokenizer_dir: Path, datasets: dict[str, list[dict]]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    before = tokenizer.encode(PACKAGE, add_special_tokens=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [
        AddedToken(PACKAGE, special=True, normalized=False)
    ]})
    after = tokenizer.encode(PACKAGE, add_special_tokens=False)
    if len(after) != 1:
        raise RuntimeError(f"专用 Package token 未成为单个 token：{after}")
    package_id = after[0]
    package_checks = []
    for split in ("base_train", "base_eval", "router_train", "router_eval"):
        rows = datasets[split]
        for row in rows:
            text = row.get("text", row.get("input", ""))
            ids = tokenizer.encode(text, add_special_tokens=False)
            package_checks.append(ids.count(package_id) == text.count(PACKAGE))
    if not all(package_checks):
        raise RuntimeError("数据中的 Package 标记没有全部映射到同一个单 token ID")
    base_lengths = [len(tokenizer.encode(row["text"], add_special_tokens=False))
                    for row in datasets["base_train"]]
    return {"base_vocab_size": len(tokenizer) - 1, "before_ids": before,
            "package_token_id": package_id, "after_ids": after,
            "all_package_occurrences_use_one_id": True,
            "base_train_tokens": sum(base_lengths),
            "base_train_average_tokens": round(sum(base_lengths) / len(base_lengths), 1),
            "base_train_max_tokens": max(base_lengths)}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成并检查通用底座种子 v0 与代码 Package 路由数据；不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR, help="本地基础 tokenizer 目录")
    args = parser.parse_args()

    datasets = build_data()
    for name, rows in datasets.items():
        write_jsonl(args.data_dir / f"code_package_v1_{name}.jsonl", rows)
    token_audit = audit(args.tokenizer, datasets)
    report = {
        "status": "datasets_prepared",
        "counts": {name: len(rows) for name, rows in datasets.items()},
        "router_train_label_counts": {
            label: sum(row["label"] == label for row in datasets["router_train"])
            for label in ("CODE_EXPLAIN", "UNKNOWN")
        },
        "package_token_audit": token_audit,
        "base_train_duplicates": len(datasets["base_train"]) - len({r["text"] for r in datasets["base_train"]}),
        "router_eval_text_overlap": len(
            {r["text"] for r in datasets["router_train"]}
            & {r["text"] for r in datasets["router_eval"]}
        ),
        "router_payload_leaks": sum(
            row["payload"] in row["text"] for row in datasets["router_train"] + datasets["router_eval"]
        ),
        "base_eval_code_overlap": len(
            {row["code"] for row in datasets["base_eval"] if row["code"]}
            & {row[0] for row in CODE_TRAIN}
        ),
    }
    if any(report[key] for key in (
        "base_train_duplicates", "router_eval_text_overlap",
        "router_payload_leaks", "base_eval_code_overlap",
    )):
        raise RuntimeError(f"发现重复或训练/评估泄漏：{report}")
    (args.data_dir / "code_package_v1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
