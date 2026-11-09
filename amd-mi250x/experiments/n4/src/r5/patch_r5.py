import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep(' or dyn > 65536 or sorted((bM, bN)) != [128, 256]:', ' or dyn > 65536 or bM % 32 or bN % 32:')
i = s.index('# Below 200 hand-kernel blocks'); j = s.index('def place_b(a, b):')
s = s[:i] + '''# Round 5 (evidence/N4/RESULTS.md, ratios over hipBLASLt): the rule scores each tile by how evenly its blocks fill the 110 CUs, fill = B / (110 * ceil(B / 110)), times a per-tile factor.
# TF follows fill with the tile fixed: 128x256 at K 2048 reads 88, 101, 107, 112 TF at fill 0.73, 0.85, 0.91, 1.0. The exponent and factors are fitted to about 30 shapes, not derived. Why fill matters is a hypothesis.
HAND_CUS = 110
HAND_TILE_FACTOR = {(256, 192): 1.02, (192, 256): 1.02, (128, 128): 0.95, (64, 128): 0.88}
HAND_FILL_EXP = 0.7
# Below these hipBLASLt is used. 1536^3 reads 1.04-1.04 at 64x128; 1024^3 reads 1.00-1.01 at best and 128x5120x5120 needs a short side the rule does not take.
HAND_MIN_MN = 1536 * 1536
HAND_MIN_SIDE = 1024
# Below this K the 256x128 tile beats 128x256 on a placed B too: 4096x4096x192 1.244-1.252 against 1.220-1.221, 2560x2560x256 1.267-1.276 against 1.235-1.242. At K 512 and 1024 they tie within 1%.
HAND_LOW_K = 512


def hand_fill(blocks):
    return blocks / (HAND_CUS * -(-blocks // HAND_CUS))


def hand_pad(bM, bN):
    """LDS row padding in bytes: 16 is the largest that fits 64 KiB on 256x192 and 192x256, 32 elsewhere."""
    return 16 if 192 in (bM, bN) else 32


def hand_tile(M, N, K, placed=False):
    """(bM, bN) of the hand kernel for this shape, or None if it does not take the shape (then select_tile or hipBLASLt).
    placed: the caller has A - B = 128 mod 256 bytes (place_b). Candidates are 256x192, 192x256, 128x128, 64x128 and one of 256x128 or 128x256; the 128x256 form needs placed or HAND_128X256 and K >= HAND_LOW_K.
    The highest factor * fill ** HAND_FILL_EXP wins, first listed on a tie. The score predicts our tile, not hipBLASLt: 1536x6400x2048 still reads 0.97-0.99.
    voa and vob are 32-bit byte offsets and the C index is an int, so 2*M*K, 2*N*K at 2^32 or M*N at 2^31 would wrap and return wrong results silently."""
    if K % 64 or K < 192 or M * N < HAND_MIN_MN or min(M, N) < HAND_MIN_SIDE or 2 * max(M, N) * K >= 2**32 or M * N >= 2**31:
        return None
    fits = lambda t: M % t[0] == 0 and N % t[1] == 0
    o128 = (placed or (M, N, K) in HAND_128X256) and K >= HAND_LOW_K
    big = [(128, 256), (256, 128)] if o128 else [(256, 128), (128, 256)]
    big = next((t for t in big if fits(t)), None)
    if big and not o128 and big == (128, 256) and not (placed or (M, N, K) in HAND_128X256):
        big = None  # as before: without placement 128x256 only at the listed shapes
    cand = [t for t in [(256, 192), (192, 256), big, (128, 128), (64, 128)] if t and fits(t)]
    if not cand:
        return None
    return max(cand, key=lambda t: HAND_TILE_FACTOR.get(t, 1.0) * hand_fill((M // t[0]) * (N // t[1])) ** HAND_FILL_EXP)


# Decode shapes read with a hand tile. M below bM is padded with zero rows by gemm_small_m. Exact shapes only: no rule fits these reads.
SMALL_M_HAND = {(64, 5120, 5120): (32, 128), (32, 5120, 5120): (32, 64), (16, 17408, 5120): (32, 256), (1, 5120, 5120): (32, 64), (64, 17408, 5120): (64, 256), (64, 5120, 17408): (32, 128)}


def gemm_small_m(M, N, K):
    """Callable (a, b) -> C for a shape in SMALL_M_HAND, else None. Hand-scheduled asm."""
    t = SMALL_M_HAND.get((M, N, K))
    if t is None:
        return None
    Mp = max(M, t[0])
    k = gemm_hand(Mp, N, K, *t, pad=hand_pad(*t))
    if Mp == M:
        return k

    def run(a, b):
        return k(torch.cat([a, torch.zeros(Mp - M, K, device=a.device, dtype=a.dtype)]), b)[:M]
    return run


''' + s[j:]
rep('        return gemm_hand(M, N, K, *t) if t else None\n', '        return gemm_hand(M, N, K, *t, pad=hand_pad(*t)) if t else None\n')
rep('''if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "handsrc":''', '''if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "smallm":
    run_small_m()
    sys.exit(0)

if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "handsrc":''')
rep('''def run_handsrc():''', '''def run_small_m():
    """argv: smallm M N K rounds iters warmup [placed]. gemm_small_m against hipBLASLt through torch, same process, rotated order. Placed uses place_b against the padded A's address only when no padding is needed."""
    M, N, K, rounds, iters, warm = map(int, sys.argv[2:8])
    placed = len(sys.argv) > 8 and sys.argv[8] == "placed"
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(N, K, device="cuda", dtype=torch.float16)
    ref = a.float() @ b.float().T
    if placed:
        b = place_b(a, b)
    k = gemm_small_m(M, N, K)

    def check(c):
        return round(((c.float() - ref).norm() / ref.norm()).item(), 7)

    print(json.dumps({"M": M, "N": N, "K": K, "tile": SMALL_M_HAND.get((M, N, K)), "placed": placed, "rel_err": check(k(a, b)), "rel_err_hb": check(a @ b.T)}), flush=True)
    arms = [("hipblaslt", lambda: a @ b.T), ("rule", lambda: k(a, b))]
    for r in range(rounds):
        for i in range(2):
            nm, fn = arms[(i + r) % 2]
            ms = bench(fn, iters, warm)
            print(json.dumps({"impl": nm, "round": r, "pos": i, "M": M, "N": N, "K": K, "ms": round(ms, 5), "TF": round(2 * M * N * K / ms * 1e-9, 2)}), flush=True)


def run_handsrc():''')
open(p, 'w').write(s)
