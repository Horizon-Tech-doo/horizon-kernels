"""N4 lever D: 128-byte K and V shifts on N5's landed prefill kernel (gqa_flash_fast, P32 bN96, 768 threads) and on CK, same process. Offsets are relative to Q and live inside one spare allocation per tensor, refilled before each arm.
argv: tag rounds. Arm order rotates per round and flips by tag. Ratio convention in the analysis: ours / CK, throughput, higher is better."""
import json, os, sys, time
os.environ["TILELANG_DEFAULT_TARGET"] = json.dumps(dict(kind="hip", mcpu="gfx90a", mtriple="amdgcn-amd-amdhsa-hcc", thread_warp_size=64))
import torch
import gqa_fa
from n5lib import CASES, inputs
tag, rounds = sys.argv[1], int(sys.argv[2])
try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None
print(json.dumps(dict(info=1, tag=tag, torch=torch.__version__, stack=os.environ.get("N5_STACK"), ck=flash_attn_func is not None, gqa_fa=gqa_fa.__file__)), flush=True)


def bench(fn, warm, rep):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / rep


class Slot:
    def __init__(self, t):
        self.t, self.sz = t, t.element_size()
        self.sp = torch.empty(t.numel() + 512 // self.sz, device=t.device, dtype=t.dtype)

    def at(self, ref, off):
        st = ((ref.data_ptr() - self.sp.data_ptr() - off) & 255) // self.sz
        v = self.sp[st:st + self.t.numel()].view(self.t.shape)
        v.copy_(self.t)
        assert (ref.data_ptr() - v.data_ptr()) & 255 == off
        return v


TL_OFF = [("k0v0", 0, 0), ("k128v0", 128, 0), ("k0v128", 0, 128), ("k128v128", 128, 128)]
CK_OFF = [("k0v0", 0, 0), ("k128v0", 128, 0), ("k128v128", 128, 128)]
for case, rep in (("d128_8192", 200), ("d128_16384", 60)):
    s = CASES[case]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    flops = 4.0 * B * Hkv * G * L * L * D * (0.5 if causal else 1.0)
    Q, K, V = inputs("randn0", s)
    k = gqa_fa.gqa_flash_fast(B, L, Hkv, G, D, causal, 32, 96, 768, 2, True, True, True, True, 0, 0)
    sk, sv = Slot(K), Slot(V)
    arms = [("tl", *a) for a in TL_OFF] + ([("ck", *a) for a in CK_OFF] if flash_attn_func else [])
    ref_o = {}
    for impl, n, ko, vo in arms:
        kk, vv = sk.at(Q, ko), sv.at(Q, vo)
        O = k(Q, kk, vv) if impl == "tl" else flash_attn_func(Q, kk, vv, causal=causal)
        same = None
        if impl in ref_o:
            same = bool(torch.equal(ref_o[impl], O))
        else:
            ref_o[impl] = O.clone()
        print(json.dumps(dict(kind="check", tag=tag, case=case, impl=impl, arm=n, bit_same_as_first=same)), flush=True)
        del O
    if "ck" in ref_o:
        print(json.dumps(dict(kind="check", tag=tag, case=case, rel_tl_vs_ck=((ref_o["tl"].float() - ref_o["ck"].float()).norm() / ref_o["ck"].float().norm()).item())), flush=True)
    order = arms[::-1] if tag.endswith("b") else arms
    for r in range(rounds):
        sh = r * len(order) // max(rounds, 1)
        for pos, (impl, n, ko, vo) in enumerate(order[sh:] + order[:sh]):
            kk, vv = sk.at(Q, ko), sv.at(Q, vo)
            fn = (lambda kk=kk, vv=vv: k(Q, kk, vv)) if impl == "tl" else (lambda kk=kk, vv=vv: flash_attn_func(Q, kk, vv, causal=causal))
            ms = bench(fn, 5, rep)
            print(json.dumps(dict(kind="time", tag=tag, case=case, impl=impl, arm=n, round=r, pos=pos, ms=round(ms, 4), tflops=round(flops / ms * 1e-9, 2))), flush=True)
    del Q, K, V, k, sk, sv, ref_o
    torch.cuda.empty_cache()
