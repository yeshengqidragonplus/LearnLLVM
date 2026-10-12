# -*- coding: utf-8 -*-
"""对通用底座种子 v0 的固定留出题生成回答，保存供人工复核。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from runtime import add_runtime_args, resolve_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="运行通用底座种子 v0 留出题；结果需人工评分。")
    parser.add_argument("--model", required=True, help="训练输出中的 final_model 目录")
    parser.add_argument("--data", default="data/code_package_v1_base_eval.jsonl", help="留出题 JSONL")
    parser.add_argument("--out", default=None, help="评估 JSON 输出路径（默认写入模型目录）")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="每题最大生成 token 数")
    add_runtime_args(parser)
    args = parser.parse_args()
    device, dtype = resolve_runtime(args.device, args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Qwen2ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()
    rows = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line]
    results = []
    for index, row in enumerate(rows, start=1):
        encoded = tokenizer(row["input"], return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        answer = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        result = {"index": index, **row, "answer": answer}
        results.append(result)
        print(f"[{index:02d}/{len(rows):02d}] {row['question']}\n  生成：{answer}\n  参考点：{'；'.join(row['must_include'])}")

    out_path = Path(args.out) if args.out else Path(args.model).parent / "base_eval_generations.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": args.model, "device": str(device),
                                    "dtype": str(dtype), "results": results},
                                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"人工复核结果已保存：{out_path}")


if __name__ == "__main__":
    main()
