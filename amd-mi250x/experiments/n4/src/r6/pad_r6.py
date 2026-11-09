# N4 round 6: zero-padded N so hb's own full-fill tile runs in the hand loop. argv: fwd|rev rounds [only]. Ratio ours over torch hb, throughput, true 2MNK, same process, rotated. Hand-scheduled asm except arm S1 (compiled TileLang).
# B is padded once with zero rows (long-lived B, like place_b); C is trimmed with a view. cat = landed A padding per call; buf = reused A buffer, 16 rows copied per call; st = A padded once, no per-call copy.
import json, sys
sys.path.insert(0, "../../scripts")
import torch, gemm_prefetch as g
order, rounds = sys.argv[1], int(sys.argv[2])
only = sys.argv[3] if len(sys.argv) > 3 else None
D = dict(device="cuda", dtype=torch.float16)

def pad_b(a, b, Np, placed=False):
    """B with Np - N zero rows after its data; placed: A - B = 128 mod 256 bytes."""
    N, K = b.shape
    if placed:
        spare = torch.zeros(Np * K + 128, **D)
        start = ((a.data_ptr() - spare.data_ptr() - 128) & 255) // 2
        out = spare[start:start + Np * K].view(Np, K)
    else:
        out = torch.zeros(Np, K, **D)
    out[:N].copy_(b)
    return out

def bench_shape(tag, M, N, K, secs, rds, build):
    torch.manual_seed(0)
    a = torch.randn(M, K, **D); b0 = torch.randn(N, K, **D)
    ref = a.float() @ b0.float().T
    arms = build(a, b0)
    chk = lambda c: round(((c.float() - ref).norm() / ref.norm()).item(), 7)
    for nm, fn, mod in arms:
        c = fn(); assert c.shape == (M, N), (nm, c.shape)
        print(json.dumps({"tag": tag, "arm": nm, "M": M, "N": N, "K": K, "rel": chk(c), "mod256": mod}), flush=True)
    t = max(g.bench(fn, 3, 1) for _, fn, _ in arms)
    it = min(8000, max(4, int(secs / (t * 1e-3)))); w = it // 10 + 1
    print(json.dumps({"tag": tag, "it": it, "w": w, "order": [x[0] for x in (arms[::-1] if order == "rev" else arms)]}), flush=True)
    seq = arms[::-1] if order == "rev" else arms
    for r in range(rds):
        for i in range(len(seq)):
            nm, fn, _ = seq[(i + r) % len(seq)]
            ms = g.bench(fn, it, w)
            print(json.dumps({"tag": tag, "impl": nm, "round": r, "pos": i, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)
    del a, b0, ref, arms; torch.cuda.empty_cache()

def mod(x, y):
    return (x.data_ptr() - y.data_ptr()) & 255

def s16(a, b0):
    M, N, K = 16, 17408, 5120; Np = 17440
    k256, k160 = g.gemm_hand(32, N, K, 32, 256, pad=32), g.gemm_hand(32, Np, K, 32, 160, pad=32)
    bp = pad_b(a, b0, Np); land = g.gemm_small_m(M, N, K); assert land is not None
    z = lambda: torch.cat([a, torch.zeros(32 - M, K, device=a.device, dtype=a.dtype)])
    buf = torch.zeros(32, K, **D); st = z()
    def f256buf():
        buf[:M].copy_(a); return k256(buf, b0)[:M]
    def f160buf():
        buf[:M].copy_(a); return k160(buf, bp)[:M, :N]
    return [("hb", lambda: a @ b0.T, mod(a, b0)), ("256cat", lambda: land(a, b0), None), ("256buf", f256buf, mod(buf, b0)), ("256st", lambda: k256(st, b0)[:M], mod(st, b0)),
            ("160cat", lambda: k160(z(), bp)[:M, :N], None), ("160buf", f160buf, mod(buf, bp)), ("160st", lambda: k160(st, bp)[:M, :N], mod(st, bp))]

def s64(a, b0):
    N, K = 17408, 5120; Np = 17440
    k256, k160, s1 = g.gemm_hand(64, N, K, 64, 256, pad=32), g.gemm_hand(64, Np, K, 64, 160, pad=32), g.gemm(64, N, K, 64, 64, 64, 128, 1, "base")
    bp = pad_b(a, b0, Np)
    return [("hb", lambda: a @ b0.T, mod(a, b0)), ("256", lambda: k256(a, b0), mod(a, b0)), ("S1", lambda: s1(a, b0), mod(a, b0)), ("160", lambda: k160(a, bp)[:64, :N], mod(a, bp))]

def s32(a, b0):
    N, K = 17408, 5120; Np = 17440
    k256, k160 = g.gemm_hand(32, N, K, 32, 256, pad=32), g.gemm_hand(32, Np, K, 32, 160, pad=32)
    bp = pad_b(a, b0, Np)
    return [("hb", lambda: a @ b0.T, mod(a, b0)), ("256", lambda: k256(a, b0), mod(a, b0)), ("160", lambda: k160(a, bp)[:32, :N], mod(a, bp))]

def s7168(a, b0):
    M, N, K = 7168, 3584, 4096
    bh = g.place_b(a, b0)
    ku, kp, q = g.gemm_auto(M, N, K, None, False), g.gemm_auto(M, N, K, None, True), g.gemm_hand(M, N, K, 128, 128, pad=32)
    print(json.dumps({"tag": "7168", "pick_u": g.hand_tile(M, N, K, False), "pick_p": g.hand_tile(M, N, K, True)}), flush=True)
    return [("hb|u", lambda: a @ b0.T, mod(a, b0)), ("hb|p", lambda: a @ bh.T, mod(a, bh)), ("pick|u", lambda: ku(a, b0), mod(a, b0)), ("pick|p", lambda: kp(a, bh), mod(a, bh)), ("128x128|u", lambda: q(a, b0), mod(a, b0)), ("128x128|p", lambda: q(a, bh), mod(a, bh))]

def s3072(a, b0):
    M = N = K = 3072; Np = 3200
    bh = g.place_b(a, b0)
    ku, kp, d = g.gemm_auto(M, N, K, None, False), g.gemm_auto(M, N, K, None, True), g.gemm_hand(M, Np, K, 192, 160, pad=32)
    bu, bq = pad_b(a, b0, Np), pad_b(a, b0, Np, True)
    print(json.dumps({"tag": "3072", "pick_u": g.hand_tile(M, N, K, False), "pick_p": g.hand_tile(M, N, K, True)}), flush=True)
    return [("hb|u", lambda: a @ b0.T, mod(a, b0)), ("hb|p", lambda: a @ bh.T, mod(a, bh)), ("256x192|u", lambda: ku(a, b0), mod(a, b0)), ("256x192|p", lambda: kp(a, bh), mod(a, bh)),
            ("192x160|u", lambda: d(a, bu)[:M, :N], mod(a, bu)), ("192x160|p", lambda: d(a, bq)[:M, :N], mod(a, bq))]

P = [("16w", 16, 17408, 5120, 0.25, rounds + 2, s16), ("64w", 64, 17408, 5120, 0.25, rounds + 2, s64), ("32w", 32, 17408, 5120, 0.25, rounds + 2, s32), ("7168", 7168, 3584, 4096, 0.5, rounds, s7168), ("3072", 3072, 3072, 3072, 0.5, rounds, s3072)]
for tag, M, N, K, secs, rds, build in P:
    if only and tag != only:
        continue
    bench_shape(tag, M, N, K, secs, rds, build)
