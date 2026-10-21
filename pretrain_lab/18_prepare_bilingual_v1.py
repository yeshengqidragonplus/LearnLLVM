# -*- coding: utf-8 -*-
"""生成中英双语词条预训练数据 v1：以「猜/guess」为格式样板，不启动训练。

设计意图：
- v0（13 号）曾把 15 组术语的双向映射混进代码 Package 数据，域混杂且样本极少；
  本脚本把「中英词条」独立成一个数据集，与代码 Package（16/17 号）分开生成、
  分开审计，训练时可各自单独喂 03 训练器，也可拼接后一起喂。
- 当前只收录用户给定的一个词条「猜/guess」，作为**格式样板**：
  一个词条展开成多种样本类型（中→英、英→中、中文释义、英文释义、例句对照、
  反向例句、词性、用法辨析、场景判断），让模型从多个角度见到同一个概念，
  而不是只背一条固定问答。
- 后续新增词条只需往 TERM_ENTRIES 追加同结构条目，样本类型自动复用。

释义与例句严格采用用户给定内容，不自行发明词义：
  猜 = 只凭经验和直觉，没有事实依据给出的一个结果。
  例句：「我猜的」。
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

# ================= 词条库 =================
# 每个词条一个 dict；新增词条照此结构追加即可，样本生成器自动复用。

TERM_ENTRIES = [
    {
        "zh": "猜",
        "en": "guess",
        # 中文释义：用户给定原文，逐字保留
        "zh_def": "只凭经验和直觉，没有事实依据给出的一个结果。",
        # 英文释义：中文释义的对应翻译，语义严格一致
        "en_def": ("To give a result based only on experience and intuition, "
                   "without any factual basis."),
        "pos": "动词（verb）；名词形式同为 guess，意为“猜测、猜想”",
        # 例句：用户给定中文例句 + 对应英文译法
        "example_zh": "我猜的",
        "example_en": ["I'm just guessing.", "That was a guess.", "I guessed it."],
        "example_note": ("说话人用这句话表明：结论不是算出来的，也不是从证据推出来的，"
                         "只是凭直觉给出的。"),
        # 用法辨析：与近义动作的区别，全部围绕“有无事实依据”这一核心
        "contrast": [
            ("估计 / estimate",
             "估计通常有已知数据、范围或先例作参考；猜没有任何事实依据。"),
            ("推断 / infer",
             "推断是从已有证据出发推出结论；猜只凭经验和直觉，不依赖证据。"),
            ("计算 / calculate",
             "计算按确定规则和输入数据得出结果；猜既不用规则也不用数据。"),
        ],
        "scene": ("当你手上没有任何数据、只凭直觉给出一个答案时，应该用“猜”（guess），"
                  "而不是“计算”或“推断”。"),
    },
]


# ================= 样本生成器 =================
# 每个函数接收一个词条，返回若干 {"text","kind"} 行。
# 所有样本都带 text 字段，可直接喂 03_train_pretrain.py。

def build_samples(entry: dict) -> list[dict]:
    """把一个词条展开成多种样本类型。"""
    zh, en = entry["zh"], entry["en"]
    zh_def, en_def = entry["zh_def"], entry["en_def"]
    rows: list[dict] = []

    def add(kind: str, text: str) -> None:
        rows.append({"text": text.strip(), "kind": kind,
                     "term_zh": zh, "term_en": en})

    # 1) 双语术语知识陈述：把中英对应关系与释义写在一段里
    add("bilingual_definition",
        f"知识：中文术语“{zh}”对应英文术语“{en}”。{zh}的意思是：{zh_def}"
        f"In English, to {en} means: {en_def}")

    # 2) 中→英：问中文词的英文说法
    add("chinese_to_english",
        f"问题：“{zh}”的英文术语是什么？\n"
        f"回答：{en}。“{zh}”指的是{zh_def}对应的英文动词是 to {en}。")

    # 3) 英→中：问英文词的中文说法（en_def 首字母转小写，避免 "means To ..."）
    add("english_to_chinese",
        f"Question: What does \"{en}\" mean in Chinese?\n"
        f"回答：中文是“{zh}”。{zh}的意思是{zh_def}"
        f"To {en} means {en_def[0].lower() + en_def[1:]}")

    # 4) 中文释义问答：只问词义，不给英文
    add("definition_zh",
        f"问题：什么叫“{zh}”？\n回答：{zh_def}")

    # 5) 英文释义问答：只问词义，不给中文
    add("definition_en",
        f"Question: What does it mean to {en}?\nAnswer: {en_def}")

    # 6) 例句对照（中→英）：用户给定例句 + 主译法 + 释义回扣
    add("example_translation",
        f"例句对照：\n中文：{entry['example_zh']}\n英文：{entry['example_en'][0]}\n"
        f"说明：这里的“{zh}”表示{zh_def}{entry['example_note']}")

    # 7) 例句多译法：同一中文例句的几种英文说法（英文之间用 " / " 分隔）
    variants = " / ".join(entry["example_en"])
    add("example_variants",
        f"中文例句“{entry['example_zh']}”可以翻译成：{variants}"
        f"这几种说法都表示说话人没有事实依据，只是凭经验和直觉给出结果，"
        f"也就是“{zh}”（to {en}）。")

    # 8) 例句对照（英→中）：反向，从英文例句回到中文
    add("example_reverse",
        f"English: {entry['example_en'][0]}\n中文：{entry['example_zh']}\n"
        f"说明：这句里的 {en} 对应中文的“{zh}”。{zh}指的是{zh_def}")

    # 9) 词性说明
    add("part_of_speech",
        f"词性说明：“{zh}”（{en}）是{entry['pos']}。{zh_def}")

    # 10) 用法辨析：每个近义动作一条，突出“有无事实依据”
    for other, explanation in entry["contrast"]:
        add("usage_contrast",
            f"辨析：“{zh}”（{en}）与“{other}”的区别：{explanation}"
            f"“{zh}”的关键在于{zh_def}")

    # 11) 场景应用：什么时候该用这个词
    add("scene_application",
        f"用法场景：{entry['scene']}"
        f"因为“{zh}”（to {en}）指的是{zh_def}")

    # 12) 判断问答：给情境选词，强化“无事实依据”这一判据
    add("judgement_qa",
        f"问题：一个人手上没有任何数据，只凭直觉说出一个答案，"
        f"这属于“{zh}”还是“计算”？\n"
        f"回答：属于“{zh}”（{en}）。因为{zh_def}"
        f"而计算需要确定的规则和输入数据。")

    return rows


def build() -> list[dict]:
    """展开全部词条；顺序固定，不依赖随机数，天然可复现。"""
    rows: list[dict] = []
    for entry in TERM_ENTRIES:
        rows.extend(build_samples(entry))
    return rows


# ================= 审计 =================

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(rows: list[dict], tokenizer_dir: Path, data_dir: Path) -> dict:
    tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))

    # 1) 每条样本必须含中文词或英文词（definition_zh/definition_en 是刻意的
    #    单语释义样本），且 text 非空；整体必须两个方向都覆盖到
    zh_hit = en_hit = 0
    for row in rows:
        if not row["text"].strip():
            raise RuntimeError(f"存在空 text：{row}")
        has_zh = row["term_zh"] in row["text"]
        has_en = row["term_en"] in row["text"]
        if not (has_zh or has_en):
            raise RuntimeError(f"样本既不含中文词也不含英文词：{row['kind']}")
        zh_hit += has_zh
        en_hit += has_en
    if zh_hit == 0 or en_hit == 0:
        raise RuntimeError("数据集缺少某一语言方向的样本")

    # 2) text 唯一，避免重复样本把 loss 带偏
    texts = [r["text"] for r in rows]
    if len(texts) != len(set(texts)):
        dupes = {t for t in texts if texts.count(t) > 1}
        raise RuntimeError(f"存在重复样本：{sorted(dupes)[:2]}")

    # 3) 样本类型覆盖：确认样板的各角度都生成到了
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
    expected_kinds = {"bilingual_definition", "chinese_to_english",
                      "english_to_chinese", "definition_zh", "definition_en",
                      "example_translation", "example_variants",
                      "example_reverse", "part_of_speech", "usage_contrast",
                      "scene_application", "judgement_qa"}
    missing = expected_kinds - set(kinds)
    if missing:
        raise RuntimeError(f"缺少样本类型：{sorted(missing)}")

    # 4) 复现性：重建必须逐字节一致（无随机数，理应恒等）
    if json.dumps(build(), ensure_ascii=False) != json.dumps(rows, ensure_ascii=False):
        raise RuntimeError("复现校验失败：重建结果与写盘结果不一致")

    lengths = [len(tokenizer.encode(r["text"], add_special_tokens=False).ids)
               for r in rows]
    return {
        "status": "datasets_prepared",
        "term_count": len(TERM_ENTRIES),
        "terms": [{"zh": e["zh"], "en": e["en"]} for e in TERM_ENTRIES],
        "sample_count": len(rows),
        "sample_kinds": kinds,
        "total_tokens": sum(lengths),
        "average_tokens": round(sum(lengths) / len(lengths), 1),
        "max_tokens": max(lengths),
        "duplicate_texts": 0,
        "samples_containing_zh_term": zh_hit,
        "samples_containing_en_term": en_hit,
        "reproducibility_ok": True,
        "file_sha256": {"bilingual_v1_train.jsonl":
                        sha256_file(data_dir / "bilingual_v1_train.jsonl")},
        "note": ("当前仅收录「猜/guess」一个词条，作为双语数据格式样板；"
                 "新增词条往 TERM_ENTRIES 追加同结构条目即可。"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成中英双语词条预训练数据 v1（当前仅「猜/guess」样板）；不执行训练。")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="数据输出目录")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER_DIR,
                        help="本地基础 tokenizer 目录")
    args = parser.parse_args()

    rows = build()
    write_jsonl(args.data_dir / "bilingual_v1_train.jsonl", rows)
    report = audit(rows, args.tokenizer, args.data_dir)
    (args.data_dir / "bilingual_v1_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
