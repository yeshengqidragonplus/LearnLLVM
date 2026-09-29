# -*- coding: utf-8 -*-
"""check_vram_tmp.py — v5 (NAVE) 显存评估（一次性脚本）
目的：用户 8G 显卡，确认 NAVE + batch=16 + 最长样本不 OOM。

方法：真实加载模型 + NAVE，构造"最长样本 batch"，跑一次 forward+backward，
      读 torch.cuda.max_memory_allocated 峰值。
对比：无 NAVE vs 有 NAVE；batch=16 vs 12/8。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

device = "cuda"
tok = AutoTokenizer.from_pretrained("tokenizer")
VOCAB = max(len(tok), tok.eos_token_id + 1, tok.pad_token_id + 1)

# 找最长样本
lens = []
for p in ("data/math_v4.jsonl",):
    for line in open(p, encoding="utf-8"):
        t = json.loads(line)["text"]
        lens.append((len(tok.encode(t)) + 2, t))
lens.sort(reverse=True)
L_max = lens[0][0]
print(f"最长样本 token 数 = {L_max}")
print(f"次长 = {[x[0] for x in lens[:5]]}")


def build_model():
    cfg = Qwen2Config(
        vocab_size=VOCAB, hidden_size=384, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=8,
        intermediate_size=int(384 * 8 / 3), max_position_embeddings=512,
        hidden_act="silu", rms_norm_eps=1e-6, rope_theta=1000000.0,
        tie_word_embeddings=True, use_cache=False,
    )
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16).to(device)
    return m


import torch.nn.functional as F


def trial(batch, seq, use_nave, chunk=64):
    """跑一次 forward+backward，返回峰值显存(GB)"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build_model()
    nave = None
    if use_nave:
        from nave import NaveWrapper
        dig = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
        nave = NaveWrapper(m, 384, dig, n_theta=8)
        nave.enc = nave.enc.to(torch.bfloat16).to(device)
        m.nave = nave.enc
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    # 构造 batch：用最长样本填充（最坏情况）
    sample = tok.encode(lens[0][1])
    sample = [tok.eos_token_id] + sample + [tok.eos_token_id]
    ids = torch.full((batch, seq), tok.pad_token_id, dtype=torch.long)
    mask = torch.zeros((batch, seq), dtype=torch.long)
    for i in range(batch):
        n = min(len(sample), seq)
        ids[i, :n] = torch.tensor(sample[:n])
        mask[i, :n] = 1
    ids, mask = ids.to(device), mask.to(device)
    labels = ids.clone()
    labels[mask == 0] = -100

    try:
        if use_nave:
            embeds = nave.embed(m, ids)
            hidden = m.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        else:
            hidden = m.model(input_ids=ids, attention_mask=mask).last_hidden_state
        sh, sl = hidden[:, :-1, :], labels[:, 1:]
        total_sum, total_cnt = 0.0, 0
        for i in range(0, sh.shape[1], chunk):
            lg = m.lm_head(sh[:, i:i + chunk, :]).float()
            s = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                sl[:, i:i + chunk].reshape(-1),
                                reduction="sum", ignore_index=-100)
            total_sum = total_sum + s
            total_cnt += (sl[:, i:i + chunk] != -100).sum()
        loss = total_sum / total_cnt.clamp(min=1)
        loss.backward()
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 1e9
        ok = True
    except RuntimeError as e:
        peak = torch.cuda.max_memory_allocated() / 1e9
        ok = False
        print(f"      OOM: {str(e)[:60]}")
    del m, opt, nave
    torch.cuda.empty_cache()
    return peak, ok


print("\n" + "=" * 66)
print(f"{'配置':<34} {'峰值显存':>10} {'状态':>8}")
print("=" * 66)
for use_nave in (False, True):
    for b in (16, 12, 8):
        peak, ok = trial(b, L_max, use_nave)
        tag = "NAVE" if use_nave else "无 NAVE"
        print(f"{tag} batch={b} seq={L_max:<6}          {peak:>7.2f}GB {'OK' if ok else 'OOM':>8}")

print("\n参考：8G 卡可用约 7.2~7.5GB（系统占用 ~0.5-0.8GB）")
print("v4-A 实测（无 NAVE, batch=16, loss-chunk 64）= 7.6GB（贴上限但未爆）")