"""N3 round 3: cache write path by layout, image stack. A layout choice for the read kernel also moves the write. Arms are vLLM's own writers:
roc_native ops.reshape_and_cache on LHBNC (ROCM_ATTN at page 16), roc_triton triton_reshape_and_cache_flash on the same views (ROCM_ATTN's fallback, used under LBHNC), tri_n the Triton writer on LBNHC,
tri_b the Triton writer on LBHNC. Writes per step: decode 64 tokens, 8 tokens, prefill chunk 2048 and 16384 tokens, at scattered slots of a 1 GB cache. Each arm is checked by reading back. usage: n3r3w.py ROT cycles"""
import json, math, sys, torch
ROT, cycles = int(sys.argv[1]), int(sys.argv[2]); reps = 50
from vllm import _custom_ops as ops
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash as tw
from vllm.v1.attention.ops.paged_attn import PagedAttention
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
Hkv, D, PAGE = 4, 256, 16; L = 4096; B = 64; MAXP = L // PAGE; NBLK = B * MAXP
torch.manual_seed(7)
KV = torch.zeros(NBLK, Hkv, PAGE, 2 * D, device="cuda", dtype=torch.float16); kb, vb = KV.transpose(1, 2)[..., :D], KV.transpose(1, 2)[..., D:]
NH = torch.zeros(NBLK, PAGE, Hkv, 2 * D, device="cuda", dtype=torch.float16); kn, vn = NH[..., :D], NH[..., D:]
C2 = torch.zeros(2, NBLK, PAGE, Hkv * D, device="cuda", dtype=torch.float16); kc2, vc2 = PagedAttention.split_kv_cache(C2, Hkv, D)
print(json.dumps({"t": "meta", "kc2": list(kc2.shape), "vc2": list(vc2.shape), "native_ok": True}), flush=True)
sc = torch.tensor(1.0, device="cuda", dtype=torch.float32)
for T in (8, 64, 2048, 16384):
    key = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16); val = torch.randn(T, Hkv, D, device="cuda", dtype=torch.float16)
    if T <= 64:   # one token per sequence, each in its own page: decode
        slots = torch.randperm(NBLK, device="cuda")[:T] * PAGE + torch.randint(0, PAGE, (T,), device="cuda")
    else:         # whole pages at scattered blocks, consecutive slots inside a page: prefill
        np_ = -(-T // PAGE); blk = torch.randperm(NBLK, device="cuda")[:np_]; slots = (blk[:, None] * PAGE + torch.arange(PAGE, device="cuda")).flatten()[:T]
    A = {"roc_native": lambda: ops.reshape_and_cache(key, val, kc2, vc2, slots.flatten(), "auto", sc, sc),
         "roc_triton": lambda: tw(key, val, kc2, vc2, slots, "auto", sc, sc),
         "tri_n": lambda: tw(key, val, kn, vn, slots, "auto", sc, sc),
         "tri_b": lambda: tw(key, val, kb, vb, slots, "auto", sc, sc)}
    names = list(A); bl, sl = slots // PAGE, slots % PAGE
    for n in names:
        for c_ in (KV, NH, C2): c_.zero_()
        try:
            A[n]()
            if n in ("roc_native", "roc_triton"):
                kg = kc2[bl, :, :, sl]; vg = vc2[bl, :, :, sl]; g = (kg.reshape(T, Hkv, D) == key).all().item() and (vg == val).all().item()
            elif n == "tri_n":
                g = (kn[bl, sl] == key).all().item() and (vn[bl, sl] == val).all().item()
            else:
                g = (kb[bl, sl] == key).all().item() and (vb[bl, sl] == val).all().item()
            print(json.dumps({"t": "check", "T": T, "impl": n, "ok": bool(g)}), flush=True)
        except Exception as e:
            print(json.dumps({"t": "check", "T": T, "impl": n, "err": (type(e).__name__ + ": " + str(e))[-200:]}), flush=True); A.pop(n)
    names = list(A)
    for c in range(cycles):
        for ai in range(len(names)):
            n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); print(json.dumps({"t": "time", "T": T, "cycle": c, "pos": ai, "impl": n, "ms": round(ms, 5), "ns_per_token": round(ms * 1e6 / T, 1)}), flush=True)
print("DONE", flush=True)
