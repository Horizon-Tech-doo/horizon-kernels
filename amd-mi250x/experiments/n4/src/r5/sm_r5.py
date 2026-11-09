# N4 round 5: small-M rerun. argv: fwd|rev rounds. randn, hb through torch in the same process, rotated order, B at its own allocation (unplaced) and at A - B = 128 mod 256 (placed). Compiled TileLang, not hand asm.
# precompile: argv "pre" builds kernels only.
import json, sys
sys.path.insert(0, "../../scripts")
import torch, gemm_prefetch as g
SH = [(64, 5120, 5120), (32, 5120, 5120), (16, 17408, 5120), (64, 17408, 5120), (64, 5120, 17408), (1, 5120, 5120)]
def cfgs(bM):
    th = 64 if bM == 16 else 128
    return [(64, 64, th, 1), (64, 64, th, 4), (64, 128, th, 4), (64, 64, th, 8)]
def bench(fn, it, w):
    return g.bench(fn, it, w)
mode = sys.argv[1]; rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
torch.manual_seed(0)
for M, N, K in SH:
    bM = 16 if M <= 16 else (32 if M <= 32 else 64); Mp = max(M, bM)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16); b0 = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b0.float().T
    ap = a if M == Mp else torch.cat([a, torch.zeros(Mp - M, K, device="cuda", dtype=a.dtype)])
    bp = g.place_b(ap, b0)
    bh = g.place_b(a, b0)
    print(json.dumps({"M": M, "N": N, "K": K, "unplaced_mod256": (ap.data_ptr() - b0.data_ptr()) & 255, "placed_mod256": (ap.data_ptr() - bp.data_ptr()) & 255}), flush=True)
    arms = [("hb|u", lambda: a @ b0.T), ("hb|p", lambda: a @ bh.T)]
    for bN, bK, th, S in cfgs(bM):
        if K % (bK * S):
            continue
        tag = f"S{S}n{bN}k{bK}t{th}"
        if S == 1:
            k = g.gemm(Mp, N, K, bM, bN, bK, th, 1, "base")
            mk = lambda bb, k_=k: (lambda: k_(ap, bb)[:M])
        else:
            k = g.gemm_splitk(Mp, N, K, bM, bN, bK, th, 1, S); c32 = torch.zeros(Mp, N, device="cuda", dtype=torch.float32)
            def mk(bb, k_=k, c_=c32):
                def run():
                    c_.zero_(); k_(ap, bb, c_); return c_[:M].half()
                return run
        for pl, bb in (("u", b0), ("p", bp)):
            run = mk(bb)
            rel = ((run().float() - ref).norm() / ref.norm()).item()
            print(json.dumps({"M": M, "N": N, "K": K, "arm": f"{tag}|{pl}", "rel_err": round(rel, 7)}), flush=True)
            arms.append((f"{tag}|{pl}", run))
    relhb = ((arms[0][1]().float() - ref).norm() / ref.norm()).item()
    print(json.dumps({"M": M, "N": N, "K": K, "arm": "hb|u", "rel_err": round(relhb, 7)}), flush=True)
    if mode == "pre":
        continue
    if mode == "rev":
        arms = arms[::-1]
    it, w = (2000, 200) if M == 1 else (800, 80)
    for r in range(rounds):
        for i in range(len(arms)):
            nm, fn = arms[(i + r) % len(arms)]
            ms = bench(fn, it, w)
            print(json.dumps({"impl": nm, "round": r, "pos": i, "M": M, "N": N, "K": K, "ms": round(ms, 5)}), flush=True)
