# -*- coding: utf-8 -*-
"""check_env.py：验证 CUDA、BF16、tokenizer 和模型前后向。"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import argparse
import os
from pathlib import Path
import torch
import transformers
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from nave import NaveWrapper

from runtime import add_runtime_args, resolve_runtime, environment_info
parser = argparse.ArgumentParser(description="验证实验环境")
add_runtime_args(parser)
args = parser.parse_args()
device, dtype = resolve_runtime(args.device, args.dtype)
os.chdir(Path(__file__).resolve().parent)
print(environment_info(device, dtype))
tok = AutoTokenizer.from_pretrained("tokenizer", local_files_only=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
vocab = max(len(tok), tok.eos_token_id + 1, tok.pad_token_id + 1)
model = Qwen2ForCausalLM(Qwen2Config(vocab_size=vocab, hidden_size=64,
    intermediate_size=128, num_hidden_layers=1, num_attention_heads=4,
    num_key_value_heads=4, tie_word_embeddings=True)).to(dtype).to(device)
ids = torch.tensor([[tok.eos_token_id] + tok.encode("12+3=15") + [tok.eos_token_id]], device=device)
loss = model(ids, labels=ids).loss
assert torch.isfinite(loss)
loss.backward()
if device == "cuda":
    torch.cuda.synchronize()
print(f"普通模型前后向通过，loss={loss.item():.4f}")
digits = [tok.encode(d, add_special_tokens=False) for d in "0123456789"]
assert all(len(x) == 1 for x in digits)
wrapper = NaveWrapper(model, model.config.hidden_size, [x[0] for x in digits], n_theta=8)
wrapper.enc = wrapper.enc.to(dtype).to(device)
model.nave = wrapper.enc
model.zero_grad(set_to_none=True)
nave_loss = model(inputs_embeds=wrapper.embed(model, ids), labels=ids).loss
assert torch.isfinite(nave_loss)
nave_loss.backward()
assert wrapper.enc.proj.weight.grad is not None
assert torch.isfinite(wrapper.enc.proj.weight.grad).all()
if device == "cuda":
    torch.cuda.synchronize()
print(f"NAVE 前后向通过，loss={nave_loss.item():.4f}")
