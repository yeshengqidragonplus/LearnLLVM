# -*- coding: utf-8 -*-
"""
05_interact.py
交互式续写：加载训练好的模型，自己输入任意文本，看它怎么接。

用法：
    .venv\Scripts\python -X utf8 05_interact.py              # 贪心解码（默认）
    .venv\Scripts\python -X utf8 05_interact.py --temp 0.8    # 温度采样，输出更多样
    .venv\Scripts\python -X utf8 05_interact.py --max-tokens 40

输入 exit / quit / q 退出；空行直接回车可再来一次（同样输入看不同结果）。
"""
import argparse
import sys

import torch
from runtime import add_runtime_args, resolve_runtime
from transformers import Qwen2ForCausalLM, AutoTokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    parser = argparse.ArgumentParser(description="交互式续写")
    parser.add_argument("--model", default="output/final_model", help="模型目录")
    parser.add_argument("--max-tokens", type=int, default=30, help="每次续写最多生成几个 token")
    parser.add_argument("--temp", type=float, default=0.0,
                        help="采样温度：0=贪心（最稳），0.7~1.0=更多样（更容易看到重复循环）")
    parser.add_argument("--no-eos-prefix", action="store_true",
                        help="不加 eos 前缀（旧模型用；数学实验模型 v4+ 都是 eos 前缀训练的，默认加）")
    add_runtime_args(parser)
    args = parser.parse_args()

    device, dtype = resolve_runtime(args.device, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型 {args.model} ...")
    model = Qwen2ForCausalLM.from_pretrained(args.model)
    model = model.to(dtype).to(device)
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    print(f"就绪：设备={device}，精度={dtype}，temp={args.temp}，max_new_tokens={args.max_tokens}"
          f"{'' if args.no_eos_prefix else '，eos前缀=开'}")
    print("输入想测试的文本（中文/英文都行），回车续写；exit 退出\n")

    with torch.no_grad():
        while True:
            try:
                prompt = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit", "q"):
                print("再见！")
                break

            if args.no_eos_prefix:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
            else:
                ids = [tokenizer.eos_token_id] + tokenizer.encode(prompt)
                inputs = {"input_ids": torch.tensor([ids], dtype=torch.long).to(device)}
            if args.temp > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=args.temp,
                    top_p=0.9,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    do_sample=False,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  续写：{text}\n")


if __name__ == "__main__":
    main()
