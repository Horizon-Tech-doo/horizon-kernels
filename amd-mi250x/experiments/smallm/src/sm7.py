# SMALLM: small-M GEMM, all arms on one allocation, fresh draws. argv: fwd|rev. Ratio ours over torch hipBLASLt, throughput, higher is better. Every ours arm is hand asm (g.hand_source), none compiled.
# New "rng" form: A and B get a buffer resource limited to their true size, so rows beyond M or N come back as zeros from the range check, with no padded copy and no memory read of those rows.
import json, sys
sys.path.insert(0, "../../scripts")
import torch, tilelang
import tilelang.rocm.language as T
import gemm_prefetch as g
order = sys.argv[1]
D = dict(device="cuda", dtype=torch.float16)
ITERS, WARM, DRAWS, ROUNDS = 1500, 150, 5, 3

def up(x, b):
    return -(-x // b) * b

@tilelang.jit(out_idx=[-1])
def rng(M, N, K, bM, bN, Mt, Nt):
    dtype = T.float16
    src, dyn = g.hand_source(Mt, Nt, K, bM, bN, "c8", 1, 32)
    old = "make_wave_buffer_resource(A), rb=make_wave_buffer_resource(B)"
    assert old in src
    src = src.replace(old, f"make_wave_buffer_resource(A,{2 * M * K}u), rb=make_wave_buffer_resource(B,{2 * N * K}u)")

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((Mt, Nt), dtype)):
        with T.Kernel(T.ceildiv(Nt, bN), T.ceildiv(Mt, bM), threads=256, prelude=src):
            lds = T.alloc_shared((dyn,), T.uint8)
            T.call_extern("handle", "hand_body", T.address_of(A[0, 0]), T.address_of(B[0, 0]), T.address_of(C[0, 0]), T.address_of(lds[0]))
    return main

def pad(b, Np):
    out = torch.zeros(Np, b.shape[1], **D); out[:b.shape[0]].copy_(b); return out

def rform(M, N, K, bM, bN):
    Mt, Nt = up(M, bM), up(N, bN)
    k = rng(M, N, K, bM, bN, Mt, Nt)
    return k, Mt, Nt

K16r, _, _ = rform(16, 17408, 5120, 32, 160)
K16land = g.gemm_small_m(16, 17408, 5120)
K16st = g.gemm_hand(32, 17440, 5120, 32, 160, pad=32)
K64land = g.gemm_small_m(64, 5120, 5120)
K64r5280, _, _ = rform(64, 5120, 5120, 32, 96)
K64p5280 = g.gemm_hand(64, 5280, 5120, 32, 96, pad=32)
print(json.dumps({"compiled": True}), flush=True)

def arms16(a, b0):
    M, N = 16, 17408
    bp = g.small_m_b(M, b0); st = pad(a, 32); view, stv = bp[:N], st[:M]
    return [("hb_own", lambda: a @ b0.T), ("rng", lambda: K16r(a, b0)[:M, :N]), ("hb_padb", lambda: a @ view.T), ("land", lambda: K16land(a, bp)), ("hb_st", lambda: stv @ view.T), ("st", lambda: K16st(st, bp)[:M, :N])], {"rng": "hb_own", "land": "hb_padb", "st": "hb_st"}, (K16r, a, b0, M)

def arms64(a, b0):
    M, N = 64, 5120
    bp = pad(b0, 5280); view = bp[:N]
    return [("hb_own", lambda: a @ b0.T), ("land128", lambda: K64land(a, b0)), ("rng96", lambda: K64r5280(a, b0)[:M, :N]), ("hb_padb", lambda: a @ view.T), ("pad96", lambda: K64p5280(a, bp)[:M, :N])], {"land128": "hb_own", "rng96": "hb_own", "pad96": "hb_padb"}, (K64r5280, a, b0, M)

for tag, M, N, K, build in [("16", 16, 17408, 5120, arms16), ("64", 64, 5120, 5120, arms64)]:
    keep = []
    for d, mb in enumerate([0, 3, 17, 64, 129][:DRAWS]):
        torch.manual_seed(d)
        keep.append(torch.empty(mb * 512 * 1024 + 1, **D))
        a = torch.randn(M, K, **D); b0 = torch.randn(N, K, **D)
        ref = a.float() @ b0.float().T
        arms, pairs, (kr, ra, rb, mm) = build(a, b0)
        chk = lambda c: round(((c.float() - ref).norm() / ref.norm()).item(), 7)
        rel = {nm: chk(fn()) for nm, fn in arms}
        full = kr(ra, rb)
        print(json.dumps({"tag": tag, "draw": d, "rel": rel, "beyond_M_max": full[mm:].abs().max().item() if full.shape[0] > mm else None, "a_b_low": (a.data_ptr() - b0.data_ptr()) & 0xfffff}), flush=True)
        seq = arms[::-1] if order == "rev" else arms
        for r in range(ROUNDS):
            for i in range(len(seq)):
                nm, fn = seq[(i + r) % len(seq)]
                ms = g.bench(fn, ITERS, WARM)
                print(json.dumps({"tag": tag, "draw": d, "impl": nm, "round": r, "pos": i, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)
        print(json.dumps({"tag": tag, "pairs": pairs}), flush=True)
        del a, b0, ref, arms, ra, rb, full, kr
    del keep; torch.cuda.empty_cache()
