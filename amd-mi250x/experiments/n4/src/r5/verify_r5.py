# N4 round 5: landed rule through scripts/gemm_prefetch.py functions in one process. argv: fwd|rev set rounds secs. Ratio ours over torch hb, same process, rotated. Hand asm.
import json, sys
sys.path.insert(0, "../../scripts")
import torch, gemm_prefetch as g
order, which, rounds, secs = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
C = [(3072,)*3, (6144,)*3, (12288,)*3, (4608,)*3, (9216,)*3, (2304,)*3, (1536,)*3, (2048,)*3, (2816,)*3, (1024, 8192, 2048), (8192, 1024, 2048), (2048, 4096, 2048), (4096, 2048, 2048), (1024, 7680, 2048), (1536, 6400, 2048), (1280, 8448, 2048),
     (2048, 6144, 2048), (6144, 2048, 2048), (4096, 4096, 192), (2560, 2560, 256), (3072, 3072, 256), (6144, 6144, 1024), (8192,)*3, (16384,)*3, (32768, 1024, 2048), (1024, 4096, 2048), (3072, 6144, 4096), (5376, 5376, 4096), (2048, 12288, 2048), (7168, 3584, 4096)]
SETS = {"all": C, "cap": [(6144,)*3, (12288,)*3, (3072,)*3, (2048, 4096, 2048)]}
print(json.dumps({"md5note": "see md5.txt", "n": len(SETS[which])}), flush=True)
for M, N, K in SETS[which]:
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16); b0 = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b0.float().T
    bp = g.place_b(a, b0)
    t_s = 2 * M * N * K / 1e14 * 1e3
    it = min(6000, max(4, int(secs * 1e3 / t_s))); w = it // 20 + 1
    modes = [("placed", bp, True), ("unplaced", b0, False)]
    if order == "rev": modes = modes[::-1]
    for nm, b, pl in modes:
        k = g.gemm_auto(M, N, K, None, pl)
        if k is None:
            print(json.dumps({"M": M, "N": N, "K": K, "mode": nm, "rule": None}), flush=True); continue
        chk = lambda c: round(((c.float() - ref).norm() / ref.norm()).item(), 7)
        print(json.dumps({"M": M, "N": N, "K": K, "mode": nm, "tile": g.hand_tile(M, N, K, pl), "rel": chk(k(a, b)), "rel_hb": chk(a @ b.T), "mod256": (a.data_ptr() - b.data_ptr()) & 255}), flush=True)
        arms = [("hb", lambda: a @ b.T), ("rule", lambda k_=k, b_=b: k_(a, b_))]
        for r in range(rounds):
            for i in range(2):
                n, fn = arms[(i + r + (order == "rev")) % 2]
                ms = g.bench(fn, it, w)
                print(json.dumps({"impl": n, "round": r, "pos": i, "mode": nm, "M": M, "N": N, "K": K, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)
    del a, b0, bp, ref; torch.cuda.empty_cache()
