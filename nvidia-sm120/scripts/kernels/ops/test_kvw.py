# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# KV-write gate: horizon_ops.kv_write vs vLLM's reshape_and_cache_flash (fp8 e4m3, HND cache with K|V interleaved,
# as FlashInferImpl.do_kv_cache_update passes it): bit-identity of the whole cache, padding slots, speed.
import importlib, json, sys
import torch
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C_cache_ops)
OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
kw = importlib.import_module(f"tilelang.contrib.{PKG}.kv_write")
dev = torch.device("cuda")
KVH, D, PS, NB = 4, 256, 1600, 12


def log(r):
    print(json.dumps(r), flush=True); open(OUT, "a").write(json.dumps(r) + "\n")


gen = torch.Generator(device=dev); gen.manual_seed(0)
tot = 0
for NT in (1, 4, 32, 1000):
    kv = torch.zeros(NB, KVH, PS, 2 * D, dtype=torch.uint8, device=dev)
    kv2 = kv.clone()
    qkv = (torch.randn(NT, 24 * 512 + 2 * KVH * D, device=dev, generator=gen) * 3).to(torch.bfloat16)
    key = qkv[:, 24 * 512:24 * 512 + KVH * D].view(NT, KVH, D)
    val = qkv[:, 24 * 512 + KVH * D:].view(NT, KVH, D)
    slots = torch.randperm(NB * PS, generator=torch.Generator().manual_seed(NT))[:NT].to(dev)
    slots[0] = -1
    ks = torch.tensor([0.07], device=dev); vs = torch.tensor([0.05], device=dev)
    for cache, fn in ((kv, "vllm"), (kv2, "ours")):
        kcv, vcv = cache.view(torch.float8_e4m3fn).transpose(1, 2).split(D, dim=-1)
        if fn == "vllm":
            torch.ops._C_cache_ops.reshape_and_cache_flash(key, val, kcv, vcv, slots, "fp8", ks, vs)
        else:
            kw.write(key, val, kcv, vcv, slots, ks, vs)
    torch.cuda.synchronize()
    d = int((kv != kv2).sum()); tot += d
    log({"test": "kvw_bits", "NT": NT, "diff": d})
log({"summary": "kvw", "bit_diffs_total": tot})
print("TEST_KVW_DONE", flush=True)
