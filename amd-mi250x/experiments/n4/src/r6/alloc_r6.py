# N4 round 6: landed gemm_small_m against torch hb on the SAME B allocation, over fresh allocations. argv: fwd|rev. Ratio ours over hb, throughput, higher is better. Hand asm.
# hb here reads bp[:N], the data rows the kernel reads, so A and B addresses match between arms. Junk tensors move the allocator between draws.
import json, sys
sys.path.insert(0, "../../scripts")
import torch, gemm_prefetch as g
order = sys.argv[1]
for M, N, K in [(16, 17408, 5120), (32, 17408, 5120), (64, 17408, 5120)]:
    k = g.gemm_small_m(M, N, K); keep = []
    for d, mb in enumerate([0, 3, 17, 64, 129, 300]):
        torch.manual_seed(d)
        keep.append(torch.empty(mb * 512 * 1024 + 1, device="cuda", dtype=torch.float16))
        a = torch.randn(M, K, device="cuda", dtype=torch.float16); b0 = torch.randn(N, K, device="cuda", dtype=torch.float16)
        bp = g.small_m_b(M, b0); view = bp[:N]
        ref = a.float() @ b0.float().T
        chk = lambda c: round(((c.float() - ref).norm() / ref.norm()).item(), 7)
        arms = [("hb", lambda: a @ view.T), ("rule", lambda: k(a, bp))]
        print(json.dumps({"M": M, "draw": d, "rel": chk(arms[1][1]()), "rel_hb": chk(arms[0][1]()), "a_b_low": (a.data_ptr() - bp.data_ptr()) & 0xfffff}), flush=True)
        seq = arms[::-1] if order == "rev" else arms
        for r in range(2):
            for i in (0, 1):
                nm, fn = seq[(i + r) % 2]
                ms = g.bench(fn, 1500, 150)
                print(json.dumps({"M": M, "draw": d, "impl": nm, "round": r, "pos": i, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)
        del a, b0, bp, view, ref, arms
    del keep; torch.cuda.empty_cache()
