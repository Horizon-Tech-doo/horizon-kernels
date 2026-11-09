# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# A1 standalone gate: horizon_ops.attn_decode vs FlashInfer XQA (what vLLM runs) vs an fp64 reference, on a paged FP8
# KV cache with vLLM's layout (kernel page 64; allocator block size is separate, 4 KV heads, head 256, 24 query heads), q_len 1 and 4 (causal drafts);
# determinism, CUDA-graph capture; speed vs XQA at 1k..128k context for 1 / 8 / 32 requests.
# v2: supported kernel pages, capture-safe metadata, explicit correctness gates.
# usage: python test_attn_v2.py OUT.jsonl [quick] [pkg]
import importlib, json, math, os, sys, time
import torch

OUT = sys.argv[1]
QUICK = len(sys.argv) > 2 and sys.argv[2] == "quick"
PKG = sys.argv[3] if len(sys.argv) > 3 else "horizon_ops"
ad = importlib.import_module(f"tilelang.contrib.{PKG}.attn_decode")
from flashinfer.decode import xqa_batch_decode_with_kv_cache
from vllm.v1.attention.backends.flashinfer import _make_xqa_draft_block_mask

dev = torch.device("cuda")
KVH, G, D = 4, 6, 256
PS = int(os.environ.get("HORIZON_TEST_PAGE_SIZE", "64"))
assert PS in (16, 32, 64), "Use a supported SM120 attention kernel page size"
HQ = KVH * G
SCALE = D ** -0.5
WS = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=dev)


def log(rec):
    print(json.dumps(rec), flush=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


def make(B, lens, QL, gen, k_scale=0.05, v_scale=0.05):
    pages = [math.ceil(l / PS) for l in lens]
    NP = sum(pages) + 3
    perm = torch.randperm(NP - 1, generator=torch.Generator().manual_seed(len(lens) * 7 + QL)) + 1
    MP = max(pages)
    bt = torch.zeros(B, MP, dtype=torch.int32)
    o = 0
    for b, p in enumerate(pages):
        bt[b, :p] = perm[o:o + p]; o += p
    # vLLM's FlashInfer cache on SM120: HND with K | V interleaved per token: (pages, kv heads, page, 2 D)
    kv = (torch.randn(NP, KVH, PS, 2 * D, device=dev, generator=gen) * 2).to(torch.float8_e4m3fn)
    kc, vc = kv[..., :D], kv[..., D:]
    q = (torch.randn(B * QL, HQ, D, device=dev, generator=gen) * 1.0).to(torch.bfloat16)
    return q, kc, vc, bt.to(dev), torch.tensor(lens, dtype=torch.int32, device=dev), k_scale, v_scale


def reference(q, kc, vc, bt, sl, QL, ks, vs):
    B = sl.shape[0]
    out = torch.empty(q.shape, dtype=torch.float64, device=dev)
    for b in range(B):
        L = int(sl[b])
        pg = bt[b, : math.ceil(L / PS)].long()
        K = kc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * ks
        V = vc[pg].permute(0, 2, 1, 3).reshape(-1, KVH, D)[:L].double() * vs
        for t in range(QL):
            lim = L - QL + t + 1
            qq = q[b * QL + t].double().view(KVH, G, D)
            s = torch.einsum("hgd,lhd->hgl", qq, K[:lim]) * SCALE
            p = torch.softmax(s, dim=-1)
            out[b * QL + t] = torch.einsum("hgl,lhd->hgd", p, V[:lim]).reshape(HQ, D)
    return out


def make_mask(B, QL):
    return (_make_xqa_draft_block_mask(QL, True, dev).unsqueeze(0)
            .expand(B, -1, -1).contiguous()) if QL > 1 else None


def run_xqa(q, kc, vc, bt, sl, QL, ks, vs, out, max_len, mask):
    xqa_batch_decode_with_kv_cache(query=q, kv_cache=(kc, vc), workspace_buffer=WS, block_tables=bt, seq_lens=sl,
                                   max_seq_len=max_len, bmm1_scale=SCALE * ks, bmm2_scale=vs, window_left=-1,
                                   out=out, sinks=None, kv_layout="HND", q_len_per_req=QL, mask=mask)


def run_ours(q, kc, vc, bt, sl, QL, ks, vs, out, max_len=None, pipe=0):
    ad.decode(q, kc, vc, bt, sl, max_len if max_len is not None else 0, SCALE * ks, vs, QL, out=out, kv_layout="HND",
              pipe=pipe, block_N=32 if pipe else 64)


BASELINE_OK = True

def correctness():
    global BASELINE_OK
    gen = torch.Generator(device=dev); gen.manual_seed(1)
    cases = [(1, [37], 1), (1, [1000], 4), (1, [1601], 4), (3, [5, 3300, 900], 4), (8, [4000] * 8, 4),
             (2, [20000, 7], 1), (1, [65536], 4)]
    for B, lens, QL in cases:
        q, kc, vc, bt, sl, ks, vs = make(B, lens, QL, gen)
        ref = reference(q, kc, vc, bt, sl, QL, ks, vs)
        o1 = torch.empty_like(q); o2 = torch.empty_like(q); o3 = torch.empty_like(q)
        mask = make_mask(B, QL)
        run_xqa(q, kc, vc, bt, sl, QL, ks, vs, o1, max(lens), mask)
        run_ours(q, kc, vc, bt, sl, QL, ks, vs, o2)
        run_ours(q, kc, vc, bt, sl, QL, ks, vs, o3)
        o4 = torch.empty_like(q)
        run_ours(q, kc, vc, bt, sl, QL, ks, vs, o4, pipe=1)
        torch.cuda.synchronize()
        e_x = (o1.double() - ref).abs().max().item(); e_o = (o2.double() - ref).abs().max().item()
        r_x = ((o1.double() - ref).norm() / ref.norm()).item(); r_o = ((o2.double() - ref).norm() / ref.norm()).item()
        log({"test": "attn_correct", "page_size": PS, "B": B, "lens": lens[:4], "QL": QL, "xqa_max_abs": e_x, "ours_max_abs": e_o,
             "xqa_rel_l2": r_x, "ours_rel_l2": r_o, "ours_vs_xqa_max_abs": (o1.float() - o2.float()).abs().max().item(),
             "deterministic": bool(torch.equal(o2, o3)),
             "pipe_rel_l2": ((o4.double() - ref).norm() / ref.norm()).item()})
        baseline_ok = bool(torch.isfinite(o1).all()) and math.isfinite(r_x) and r_x <= 0.0025
        BASELINE_OK = BASELINE_OK and baseline_ok
        log({"test": "baseline_gate", "B": B, "KVH": KVH, "QL": QL,
             "lens": lens, "relative_l2_limit": 0.0025, "ok": baseline_ok})
        assert torch.isfinite(o2).all() and torch.isfinite(o4).all()
        assert r_o <= 0.0025
        assert ((o4.double() - ref).norm() / ref.norm()).item() <= 0.0025
        assert torch.equal(o2, o3), "decode nondeterminism"
        for pipe in (0, 1):
            target = o4 if pipe else o2
            expected = target.clone()
            graph = graph_of(lambda: run_ours(q, kc, vc, bt, sl, QL, ks, vs, target, max(lens), pipe))
            for _ in range(2):
                target.fill_(float("nan")); graph.replay(); torch.cuda.synchronize()
                assert torch.equal(target, expected), "graph replay differs from eager"
            del graph
        log({"test": "attn_gate", "B": B, "KVH": KVH, "QL": QL, "page_size": PS, "ok": True})
        del kc, vc


def graph_of(fn, *a):
    fn(*a); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    for attempt in range(5):
        try:
            with torch.cuda.graph(g, capture_error_mode="thread_local"):
                fn(*a)
            return g
        except Exception as e:  # noqa: BLE001
            print("capture retry", attempt, e, flush=True); g = torch.cuda.CUDAGraph()
    raise RuntimeError("capture failed")


def bench():
    gen = torch.Generator(device=dev); gen.manual_seed(2)
    ctxs = (1024, 32768) if QUICK else (1024, 4096, 16384, 32768, 131072)
    for B in (1, 8, 32):
        for L in ctxs:
            if B * L > 1_100_000:      # keep the fp8 cache <= ~2.3 GB per copy
                continue
            for QL in (4, 1):
                layers = max(2, math.ceil(513 * 1024**2 / ((B * math.ceil(L / PS) + 3) * PS * KVH * D * 2)))
                sets = [make(B, [L] * B, QL, gen) for _ in range(layers)]
                o = torch.empty_like(sets[0][0])

                mask = make_mask(B, QL)

                def gx():
                    for s in sets: run_xqa(*s[:4], s[4], QL, s[5], s[6], o, L, mask)

                def go():
                    for s in sets: run_ours(*s[:4], s[4], QL, s[5], s[6], o)

                def gp():
                    for s in sets: run_ours(*s[:4], s[4], QL, s[5], s[6], o, pipe=1)
                gs = {"xqa": graph_of(gx), "ours": graph_of(go), "ours_pipe": graph_of(gp)}
                best = {}
                for rnd in range(3):
                    names = list(gs)
                    for n in names[rnd:] + names[:rnd]:
                        g = gs[n]
                        g.replay(); torch.cuda.synchronize()
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        ts = []
                        for _ in range(10):
                            e0.record(); g.replay(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1) * 1000 / layers)
                        ts.sort(); best[n] = min(best.get(n, 1e30), ts[len(ts) // 2])
                floor = B * L * KVH * D * 2 / 1.63e6
                log({"test": "attn_bench", "B": B, "ctx": L, "QL": QL, "page_size": PS, "layers": layers, "rotated_cache_bytes": layers * sets[0][1].untyped_storage().nbytes(), "xqa_us": round(best["xqa"], 2),
                     "ours_us": round(best["ours"], 2), "ours_pipe_us": round(best["ours_pipe"], 2),
                     "floor_us": round(floor, 2), "baseline_quality_pass": BASELINE_OK,
                     "comparison_scope": "same_tolerance" if BASELINE_OK else "timing_only_baseline_failed_quality"})
                del sets, gs
                torch.cuda.empty_cache()


if __name__ == "__main__":
    log({"test": "configuration", "page_size": PS, "kernel_package": PKG, "quick": QUICK})
    correctness()
    # MoE model shape (Qwen3.6-35B-A3B: 16 q heads, 2 kv heads)
    KVH, G = 2, 8
    HQ = KVH * G
    correctness()
    KVH, G = 4, 6
    HQ = KVH * G
    bench()
    log({"test": "complete", "ok": True, "baseline_quality_pass": BASELINE_OK, "deployment_eligible": False})
    print("TEST_ATTN_DONE", flush=True)
