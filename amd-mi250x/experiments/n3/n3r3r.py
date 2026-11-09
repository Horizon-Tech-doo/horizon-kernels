"""N3 round 3: real KV data. Query, K, V of 3 full-attention layers captured from a live vLLM 0.30 run of Qwen3.8-27B at the first decode step (n3r3k.py), BF16 as served, 8 sequences of 4.5k-8k tokens on real pages.
B=8 as captured; B=32 repeats each sequence 4 times on the same pages. Arms as n3r3h.py plus "rand", the same arms on N(0,1) data with the same lengths and table, same process. bf16 and fp16 (the capture cast). Ratio vLLM ms / ours ms. usage: n3r3r.py image ROT cycles [compile I N]"""
import json, math, sys, importlib.util, torch
stack, ROT, cycles = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); compile_only = len(sys.argv) > 4
CI, CN = (int(sys.argv[5]), int(sys.argv[6])) if compile_only else (0, 1)
sys.argv = ["x"]; reps = 50
spec = importlib.util.spec_from_file_location("gq", "/scratch/project_XXXXXXXXX/tilelang/exp/N3/gqa_decode.py"); gq = importlib.util.module_from_spec(spec); spec.loader.exec_module(gq)
Hkv, G, D, PAGE, N = 4, 6, 256, 16, 784; H = Hkv * G; RK = "/scratch/project_XXXXXXXXX/tilelang/exp/N3/realkv"
LANDS = {"rh": ("rocm_hybrid", "joint", "xj"), "rh_vk": ("rocm_hybrid", "v_k", "xj"), "n": ("lbnhc", "joint", "jx")}
DT = {"fp16": torch.float16, "bf16": torch.bfloat16}; L = 8192; MAXB = -(-L // N); MAXP = L // PAGE
if compile_only:
    i = 0
    d0 = torch.load(f"{RK}/layer0.pt", map_location="cpu"); nb = sum(s["K"].shape[0] for s in d0["seqs"])
    for B in (8, 32):
        for dt in DT:
            for n, (lay, st, ko) in LANDS.items():
                if i % CN == CI: gq.paged_kernels(B, nb, MAXP, Hkv, G, D, PAGE, lay, st, ko, N, dt)
                i += 1
    print("COMPILED", CI, flush=True); sys.exit(0)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention as unified
from vllm.v1.attention.ops.chunked_prefill_paged_decode import chunked_prefill_paged_decode as chunked
from vllm.v1.attention.ops.paged_attn import PagedAttention
def bench(fn, warmup=10, rep=reps):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(rep): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep
def P(**k): print(json.dumps({"stack": stack, **k}), flush=True)
def rel(o, R): return float(((o.double() - R).norm() / R.norm()).item())
for layer in (0, 8, 15):
    d = torch.load(f"{RK}/layer{layer}.pt", map_location="cpu")
    Kp = torch.cat([s["K"] for s in d["seqs"]]).cuda(); Vp = torch.cat([s["V"] for s in d["seqs"]]).cuda(); Q0 = d["Q"].cuda(); lens0 = [s["n"] for s in d["seqs"]]
    NBLK = Kp.shape[0]; assert Kp.shape[3] == N
    off = 0; BT0 = torch.zeros(8, MAXB, dtype=torch.int32, device="cuda")
    for b, s in enumerate(d["seqs"]):
        m = s["K"].shape[0]; BT0[b, :m] = torch.arange(off, off + m, device="cuda", dtype=torch.int32); off += m
    kreal = Kp.permute(0, 3, 1, 2, 4).reshape(NBLK, N, Hkv, D).float(); vreal = Vp.permute(0, 3, 1, 2).float()
    nz = [BT0[b, :(lens0[b] + N - 1) // N].long() for b in range(8)]
    for src in ("real", "rand"):
        for dt, dty in DT.items():
            torch.manual_seed(layer + 7)
            if src == "real": kv, vv, Q1 = kreal.to(dty), vreal.to(dty), Q0.float().to(dty)
            else: kv, vv, Q1 = torch.randn_like(kreal).to(dty), torch.randn_like(vreal).to(dty), torch.randn_like(Q0.float()).to(dty)
            if dt == "bf16" and src == "real": P(t="data", layer=layer, k_rms=kreal.norm().item() / math.sqrt(kreal.numel()), v_rms=vreal.norm().item() / math.sqrt(vreal.numel()), q_rms=Q0.float().norm().item() / math.sqrt(Q0.numel()), k_max=kreal.abs().amax().item(), v_max=vreal.abs().amax().item(), v_zero=float((vreal == 0).float().mean()))
            C = torch.empty(NBLK, 2, N, Hkv * D, device="cuda", dtype=dty); K6 = C.view(NBLK, 2, Hkv, D // 8, N, 8); V5 = C.view(NBLK, 2, Hkv, D, N)
            K6[:, 0].copy_(kv.reshape(NBLK, N, Hkv, D // 8, 8).permute(0, 2, 3, 1, 4)); V5[:, 1].copy_(vv.permute(0, 2, 3, 1))
            NH = torch.cat([kv, vv], -1); knh, vnh = NH[..., :D], NH[..., D:]; kc2, vc2 = PagedAttention.split_kv_cache(C.transpose(0, 1), Hkv, D)
            for B in (8, 32):
                rp = B // 8; Q = Q1.repeat(rp, 1, 1); BT = BT0.repeat(rp, 1); lens = torch.tensor(lens0 * rp, dtype=torch.int32, device="cuda")
                KS = {n: gq.paged_kernels(B, NBLK, MAXP, Hkv, G, D, PAGE, lay, st, ko, N, dt) for n, (lay, st, ko) in LANDS.items()}
                scale = 1.0 / math.sqrt(D); cu = torch.arange(B + 1, device="cuda", dtype=torch.int32); out = torch.empty_like(Q)
                A = {}
                for n in KS:
                    a, b = (K6, V5) if n.startswith("rh") else (NH, NH)
                    A[n] = (lambda k=KS[n], a=a, b=b: (lambda: k[1](*k[0](Q, a, b, BT, lens))))()
                def tr():
                    unified(q=Q, k=knh, v=vnh, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=lens, max_seqlen_k=L, softmax_scale=scale, causal=True, window_size=(-1, -1), block_table=BT, softcap=0, q_descale=None, k_descale=None, v_descale=None); return out
                def rc():
                    chunked(query=Q, key=None, value=None, output=out, kv_cache_dtype="auto", key_cache=kc2, value_cache=vc2, block_table=BT, query_start_loc=cu, seq_lens=lens, max_seq_len=L, max_query_len=1, k_scale=1.0, v_scale=1.0, alibi_slopes=None, sliding_window=0, sm_scale=scale, output_scale=None, sinks=None, causal=True); return out
                A["tri"] = tr; A["roc"] = rc
                # reference in fp64, per sequence
                Kseq = [kv[BT[b][:(int(lens[b]) + N - 1) // N].long()].reshape(-1, Hkv, D)[:int(lens[b])] for b in range(B)]
                Vseq = [vv[BT[b][:(int(lens[b]) + N - 1) // N].long()].reshape(-1, Hkv, D)[:int(lens[b])] for b in range(B)]
                R = torch.empty(B, H, D, dtype=torch.float64, device="cuda")
                for b in range(B):
                    kk = Kseq[b].double().repeat_interleave(G, 1); vv_ = Vseq[b].double().repeat_interleave(G, 1)
                    s_ = torch.einsum("hd,nhd->hn", Q[b].double(), kk) / math.sqrt(D); R[b] = torch.einsum("hn,nhd->hd", torch.softmax(s_, -1), vv_)
                ref_tri = None
                for n in A:
                    o = A[n]().clone(); P(t="err", layer=layer, src=src, dt=dt, B=B, impl=n, rel=rel(o, R), finite=bool(torch.isfinite(o.float()).all()))
                names = list(A); nb = 2 * 2 * Hkv * D * int(lens.sum())
                for c in range(cycles):
                    for ai in range(len(names)):
                        n = names[(ai + c + ROT) % len(names)]; ms = bench(A[n]); P(t="time", layer=layer, src=src, dt=dt, B=B, cycle=c, pos=ai, impl=n, ms=round(ms, 4), pct=round(nb / ms / 1e6 / 16, 1))
                del KS, A
            del C, K6, V5, NH, knh, vnh, kc2, vc2, kv, vv; torch.cuda.empty_cache()
print("DONE", flush=True)
