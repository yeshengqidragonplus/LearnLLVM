# -*- coding: utf-8 -*-
"""生成前置打包架构 v2 的路由/代码专家数据，并审计缓存载荷隔离。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AddedToken, AutoTokenizer

from code_package_frontend_v1 import PACKAGE, SessionCache, expert_input, package_request


DATA_DIR = Path("data")
TOKENIZER_DIR = Path("tokenizer")

# 训练集与评估集分开标注：既包含见过的标识符，也包含未见标识符。
CODE_TRAIN = [
    ("int i = 0;", "解释每个部分", "int 是 C 整数类型关键字；i 是变量名/标识符；0 是整数初始值。"),
    ("int count = 3;", "count 是什么？", "count 是变量名，也是标识符；int 表示整数类型，3 是初始值。"),
    ("char initial = 'A';", "initial 和 'A' 分别是什么？", "initial 是 char 类型变量的标识符；char 是字符类型；'A' 是字符常量，用作初始值。"),
    ("float ratio = 0.5f;", "解释类型和字面量", "ratio 是变量标识符；float 表示单精度浮点类型；0.5f 是浮点字面量并作为初始值。"),
    ("double average = 3.75;", "average 保存什么？", "average 是 double 类型变量名；3.75 是浮点字面量，作为初始值。"),
    ("unsigned int attempts = 4U;", "unsigned 和 4U 分别表示什么？", "unsigned int 是无符号整数类型；attempts 是变量标识符；4U 是无符号整数字面量。"),
    ("long distance = 12L;", "long 是否一定是 64 位？", "不一定。long 的具体位宽由 C 实现决定；12L 是 long 类型整数字面量。"),
    ("short level = 7;", "level 的类型是什么？", "level 是变量名，类型为 short int；7 是整数初始值。"),
    ("int total; total = 15;", "区分声明和赋值", "int total; 是声明；total = 15; 是对已声明变量赋值，不是声明时初始化。"),
    ("int width = 640; int height = 480;", "有几个变量？", "有两个变量，名称是 width 和 height；两者都是 int 类型，初始值分别为 640 和 480。"),
    ("char grade = 'B';", "B 是变量名吗？", "不是。grade 是变量标识符；'B' 是字符字面量。char 是字符类型。"),
    ("float rate = 1.25f;", "rate 的类型、名称和值是什么？", "类型是 float；名称/标识符是 rate；初始值由浮点字面量 1.25f 提供。"),
    ("double radius = 2.5;", "radius 是 identifier 吗？", "是。radius 是标识符，在这里命名 double 类型变量；2.5 是浮点初始值。"),
    ("unsigned int mask = 8U;", "mask 是 C 关键字吗？", "不是。mask 是用户选择的变量名/标识符；unsigned 和 int 是类型说明符；8U 是无符号整数初始值。"),
    ("long timestamp = 42L;", "42L 是什么？", "42L 是 long 类型整数字面量，用作 timestamp 的初始值；timestamp 是变量名。"),
    ("int item_count = 6;", "下划线把名称拆成两个变量了吗？", "没有。item_count 是一个完整标识符，用作一个变量名；类型是 int，初始值是 6。"),
]

CODE_EVAL_SEEN_NAME = [
    ("int count = 19;", "count 是变量名还是值？", ["count 是变量名/标识符", "19 是初始值"]),
    ("float ratio = 6.5f;", "ratio 的类型和值是什么？", ["float 类型", "6.5f 是初始值"]),
]
CODE_EVAL_NEW_NAME = [
    ("float temperature = 2.75f;", "temperature 的类型和初始值是什么？", ["temperature 是标识符/变量名", "float 类型", "2.75f 是初始值"]),
    ("double probability = 0.125;", "probability 是值还是变量名？", ["probability 是标识符/变量名", "double 类型", "0.125 是初始值"]),
    ("unsigned int capacity = 32U;", "capacity 是关键字吗？", ["capacity 是用户命名的标识符", "32U 是初始值"]),
    ("short sample_count = 12;", "sample_count 是几个 identifier？", ["一个完整标识符", "12 是初始值"]),
]

ROUTER_TRAIN = [
    ("解释这段代码", "CODE_EXPLAIN"), ("说明代码每一部分的意思", "CODE_EXPLAIN"),
    ("分析代码中的变量和数值", "CODE_EXPLAIN"), ("解释代码里的英文术语", "CODE_EXPLAIN"),
    ("这行程序声明了什么", "CODE_EXPLAIN"), ("逐个说明关键字、变量名和值", "CODE_EXPLAIN"),
    ("把代码含义用中文讲清楚", "CODE_EXPLAIN"), ("看一下这段程序在表达什么", "CODE_EXPLAIN"),
    ("运行这段程序并给出输出", "UNKNOWN"), ("修改代码以修复编译错误", "UNKNOWN"),
    ("把这段代码翻译成另一种语言", "UNKNOWN"), ("仅原样复述代码，不作解释", "UNKNOWN"),
    ("为代码生成完整测试套件", "UNKNOWN"), ("总结这段设计文档", "UNKNOWN"),
]
ROUTER_EVAL = [
    ("我看不懂这行代码，讲讲各部分含义", "CODE_EXPLAIN"),
    ("请不要运行，只说明语句的意义", "CODE_EXPLAIN"),
    ("把这段代码实际执行一下", "UNKNOWN"), ("帮我把程序改成使用浮点数", "UNKNOWN"),
    ("代码中的文字请逐字复制出来", "UNKNOWN"), ("给程序加上注释并重构", "UNKNOWN"),
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_children(code: str, question: str, cache: SessionCache) -> tuple[list[str], str]:
    packaged = package_request(f"解释代码：{code}", cache)
    handles = packaged["child_handles"]
    if not handles:
        raise ValueError(f"样本没有生成子包：{code!r}")
    # 多条 C 语句拆为多个缓存子包；专家提示保留各自 package 边界。
    prompt = "\n".join(expert_input(cache, handle, question) for handle in handles)
    return handles, prompt


def build() -> dict[str, list[dict]]:
    router_train = [{"router_input": f"意图：{intent}\n包类型：{PACKAGE}", "label": label}
                    for intent, label in ROUTER_TRAIN]
    router_eval = [{"router_input": f"意图：{intent}\n包类型：{PACKAGE}", "label": label}
                   for intent, label in ROUTER_EVAL]

    expert_train = []
    for code, question, answer in CODE_TRAIN:
        cache = SessionCache()
        handles, prompt = make_children(code, question, cache)
        payload = "\n".join(cache.children[handle].payload for handle in handles)
        expert_train.append({"prompt": prompt, "target": answer, "payload": payload,
                             "identifier_group": "train_seen", "package_type": "code_statement"})
    expert_eval = []
    for group, cases in (("seen_name_new_value", CODE_EVAL_SEEN_NAME),
                         ("unseen_identifier", CODE_EVAL_NEW_NAME)):
        for code, question, expected in cases:
            cache = SessionCache()
            handles, prompt = make_children(code, question, cache)
            payload = "\n".join(cache.children[handle].payload for handle in handles)
            expert_eval.append({"prompt": prompt, "payload": payload,
                                "question": question, "expected_points": expected,
                                "identifier_group": group, "package_type": "code_statement"})
    return {"router_train": router_train, "router_eval": router_eval,
            "expert_train": expert_train, "expert_eval": expert_eval}


def audit(data: dict[str, list[dict]], tokenizer_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    before = tokenizer.encode(PACKAGE, add_special_tokens=False)
    tokenizer.add_special_tokens({"additional_special_tokens": [AddedToken(PACKAGE, special=True, normalized=False)]})
    package_ids = tokenizer.encode(PACKAGE, add_special_tokens=False)
    if len(package_ids) != 1:
        raise RuntimeError(f"Package 未编码为单 token：{package_ids}")
    route_text = [row["router_input"] for split in ("router_train", "router_eval") for row in data[split]]
    payloads = [row["payload"] for split in ("expert_train", "expert_eval") for row in data[split]]
    leaked = [(payload, text) for payload in payloads for text in route_text if payload and payload in text]
    if leaked:
        raise RuntimeError(f"代码载荷泄漏进路由输入：{leaked[:2]}")
    if any(PACKAGE not in text for text in route_text):
        raise RuntimeError("路由样本缺少由前置打包器生成的 package 类型标记")
    return {
        "package_marker_ids_before_extension": before,
        "package_marker_id_after_extension": package_ids[0],
        "package_marker_is_one_token": True,
        "router_payload_leaks": len(leaked),
        "router_samples_include_source_code": False,
        "router_train_count": len(data["router_train"]),
        "router_eval_count": len(data["router_eval"]),
        "expert_train_count": len(data["expert_train"]),
        "expert_eval_count": len(data["expert_eval"]),
        "expert_eval_groups": {group: sum(r["identifier_group"] == group for r in data["expert_eval"])
                               for group in sorted({r["identifier_group"] for r in data["expert_eval"]})},
        "semantics": "raw request -> pre-tokenizer Package/cache -> router(intent + type only) -> selected expert unpacks child payload",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="准备分离的代码 Package 路由集与子包专家集，不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR, help="Tokenizer 目录")
    args = parser.parse_args()
    data = build()
    report = audit(data, args.tokenizer)
    for name, rows in data.items():
        write_jsonl(args.data_dir / f"code_package_v2_{name}.jsonl", rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    report["base_train_tokens"] = sum(
        len(tokenizer.encode(r["prompt"] + r["target"], add_special_tokens=False))
        for r in data["expert_train"]
    )
    (args.data_dir / "code_package_v2_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
