import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

flash_attn_v1 = load(name="flash_attn_v1", sources=["main.cpp", "flash_attn_v1.cu"], extra_cuda_cflags=["-O2"])

batch_size = 16
n_heads = 12
seq_len = 64
head_embd = 64

q = torch.randn(batch_size, n_heads, seq_len, head_embd).cuda()
k = torch.randn(batch_size, n_heads, seq_len, head_embd).cuda()
v = torch.randn(batch_size, n_heads, seq_len, head_embd).cuda()

print("profiling manual attn")


def manual_attn(q, k, v):
    att = q @ k.transpose(-2, -1) * (1 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


with torch.autograd.profiler.profile(use_device="cuda") as prof:
    manual_result = manual_attn(q, k, v)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

print("profiling flash attn")

with torch.autograd.profiler.profile(use_device="cuda") as prof:
    minimal_result = flash_attn_v1.forward(q, k, v)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


print(f"attn vals sanity check: {torch.allclose(minimal_result, manual_result, rtol=1e-2, atol=1e-2)}")
