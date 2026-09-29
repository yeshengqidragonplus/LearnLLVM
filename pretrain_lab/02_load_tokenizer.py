# -*- coding: utf-8 -*-
"""
02_load_tokenizer.py
下载并复用现成 Qwen2 tokenizer（讨论定稿：个人实验复用现成 tokenizer，不从头训练）。

关键点（讨论纪要）：
  - 完整工业流程里 tokenizer 是独立训练的一步；个人实验复用现成 tokenizer，但要记住这一步存在；
  - Tokenizer 和模型强绑定：词表变了，Embedding / LM-Head 维度必须同步 ——
    因此模型 config 的 vocab_size 将跟随 Qwen2 词表（151936），不再是我们定稿表的 16384。
  - 本步骤需联网下载（约 10MB，已配置国内镜像加速）。

输出: tokenizer/ 目录下的 Qwen2 官方 tokenizer 文件（tokenizer_config.json / vocab.json / merges.txt / tokenizer.json）
"""
import os

# 使用国内镜像下载 HuggingFace 资源（默认源在国内可能超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2-0.5B"   # 只取 tokenizer，不下载模型权重
SAVE_DIR = "tokenizer"

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"正在下载并加载 {MODEL_NAME} 的 tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 保存到本地，后续步骤离线使用
tokenizer.save_pretrained(SAVE_DIR)
print(f"已保存到 {SAVE_DIR}/")

print("\nTokenizer 信息：")
print(f"  vocab_size           = {tokenizer.vocab_size}")
print(f"  pad_token            = {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
print(f"  bos_token            = {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"  eos_token            = {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"  unk_token            = {tokenizer.unk_token} (id={tokenizer.unk_token_id})")

# 抽样展示分词效果
samples = ["一加一等于二。苹果 apple。", "The cat likes milk.", "猫在椅子上睡觉。"]
for s in samples:
    ids = tokenizer.encode(s)
    print(f"\n  输入：{s}")
    print(f"  token 数：{len(ids)}")
    print(f"  ids：{ids[:20]}")

# 统计平均每条约多少 token（用于估算总 token 规模）
import json
texts = []
with open("data/toy_pretrain.jsonl", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
lens = [len(tokenizer.encode(t)) for t in texts]
print(f"\n数据规模统计：共 {len(texts)} 条，平均 {sum(lens)/len(lens):.1f} token/条，"
      f"总 token ≈ {sum(lens)}")
