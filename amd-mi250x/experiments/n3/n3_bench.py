"""N3 final: our paged decode (old, landed v2) and vLLM's Triton unified attention, same harness, per batch. Run per stack.
usage: n3_bench.py lumi|image cycles"""
import json, math, sys, importlib.util, torch
import torch.nn.functional as F
stack = sys.argv[1]; cycles = int(sys.argv[2]); reps = 50
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py")
gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
unified = None
if stack == "image":
    for mod in ("vllm.attention.ops.triton_unified_attention", "vllm.v1.attention.ops.triton_unified_attention", "vllm.v1.attention.backends.triton_unified_attention"):
        try:
            unified = __import__(mod, fromlist=["unified_attention"]).unified_attention; print(json.dumps({"unified": mod}), flush=True); break
        except Exception as e: print(json.dumps({"import_fail": mod, "e": str(e)[-120:]}), flush=True)
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
Hkv, G, D, PAGE = 4, 6, 256, 16; H = Hkv * G
OLDS = {1: 220, 8: 55, 16: 27, 32: 13, 64: 5}
for B, L in [(1, 131072), (8, 32768), (16, 16384), (32, 8192), (64, 4096)]:
    torch.manual_seed(B)
    lens = torch.full((B,), L, dtype=torch.int32, device="cuda")
    lens[1::2] = L * 3 // 4
    MAXP = L // PAGE; NBLK = B * MAXP
    Q = torch.randn(B, H, D, device="cuda", dtype=torch.float16)
    Kc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    Vc = torch.randn(NBLK, PAGE, Hkv, D, device="cuda", dtype=torch.float16)
    BT = torch.randperm(NBLK, device="cuda").to(torch.int32).view(B, MAXP)
    R = gq.ref64(Q, Kc[BT.long()].view(B, L, Hkv, D), Vc[BT.long()].view(B, L, Hkv, D), lens, G)
    nbytes = 2 * 2 * Hkv * D * int(lens.sum())
    arms = {}
    So = OLDS[B]; k1o = gq.decode_partial_paged(B, NBLK, MAXP, Hkv, G, D, So, 128, PAGE, 32); k2o = gq.decode_combine(B, H, So, D)
    arms["old"] = (lambda: k2o(*k1o(Q, Kc, Vc, BT, lens)), lambda: k1o(Q, Kc, Vc, BT, lens))
    Sv, tp = gq.paged_v2_config(MAXP)
    k1v = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, Sv, 128, PAGE, 32, "if", True, True, tp); k2v = gq.decode_combine(B, H, Sv, D); k2vf = gq.decode_combine_v2(B, H, Sv, D); k2of = gq.decode_combine_v2(B, H, So, D)
    k1u = gq.decode_partial_paged_v2(B, NBLK, MAXP, Hkv, G, D, So, 128, PAGE, 32, "if", False, True, None)
    arms["v2f"] = (lambda: k2vf(*k1v(Q, Kc, Vc, BT, lens)), lambda: k1v(Q, Kc, Vc, BT, lens))
    arms["v2_uniformf"] = (lambda: k2of(*k1u(Q, Kc, Vc, BT, lens)), lambda: k1u(Q, Kc, Vc, BT, lens))
    if unified is not None:
        scale = 1.0 / math.sqrt(D)
        cu_q = torch.arange(B + 1, device="cuda", dtype=torch.int32); out_u = torch.empty_like(Q)
        call = lambda: unified(q=Q, k=Kc, v=Vc, out=out_u, cu_seqlens_q=cu_q, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True,
                               window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None)
        arms["vllm_triton"] = (call, None)
    names = list(arms)
    for c in range(cycles):
        for ai in range(len(names)):
            n = names[(ai + c) % len(names)]; fn, fk = arms[n]
            r = {"stack": stack, "cycle": c, "pos": ai, "impl": n, "B": B, "L": L}
            if c == 0:
                o = fn()
                if n == "vllm_triton": o = out_u
                r["rel_vs_fp64"] = round(((o.double() - R).norm() / R.norm()).item(), 6)
            ms = bench(fn); r.update(ms=round(ms, 4), gbps=round(nbytes / ms / 1e6, 1), pct=round(nbytes / ms / 1e6 / 16, 1))
            if fk is not None: r["k1"] = round(bench(fk, 5, 30), 4)
            print(json.dumps(r), flush=True)
    del Kc, Vc, Q, BT, k1o, k1v, k1u, arms, R
    torch.cuda.empty_cache()
print("DONE", flush=True)
