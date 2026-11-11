"""BF16 form of gemm_prefetch.py's hand-scheduled GEMM kernels (track PROFILE, 10 Nov 2026). gemm_prefetch.py is not edited: this file loads a second copy of it and rewrites the HIP source its
hand_source / hand_source_pgr return, in two places only:
  1. the MFMA: v_mfma_f32_16x16x16f16 -> v_mfma_f32_16x16x16bf16_1k (gfx90a; same 16x16 tile, same K 16 per instruction, same operand registers, so loads, LDS layout and schedule are unchanged);
  2. the epilogue: fp32 -> bf16 by bits, round to nearest even (the form of gqa_decode._bf16_bits), in place of the (half_t) casts.
The compiled kernel still declares fp16 pointers; callers hand it bf16 tensors reinterpreted as fp16 (same 16-bit storage, no copy) and reinterpret the result. Why: the served 27B is BF16 end to end,
and the fp16 kernel wired through casts (bf16 -> fp16 activations, fp16 weight copy, fp16 -> bf16 result) reads 0.93-1.02 of vLLM's own path at the server's shapes (job 23620226), slower in 14 of 15.
Only the hand forms are taken (select_mode 'fill' or 'hand', and gemm_small_m's tables): the compiled T.gemm forms would run an fp16 MFMA on bf16 bits.
Use a TILELANG_CACHE_DIR of its own: the edit is applied to the prelude source, and a cache keyed without it would return the fp16 binary."""
import importlib.util, os, re
import torch

MFMA = os.environ.get("GEMM_BF16_MFMA", "v_mfma_f32_16x16x16bf16_1k")
_CAST = re.compile(r"\*\(\(half_t\*\)\(&\(\(\(half2\*\)\(&\(u\.[xy]\)\)\)->[xy]\)\)\)=\(half_t\)\(v_\[\d\]\);[ \t]*")
_HELPER = ("__device__ inline __attribute__((always_inline)) uint hb16_(float x_) { uint b_=__builtin_bit_cast(uint, x_); return (b_+((b_>>16)&1u)+0x7fffu)>>16; }\n")
_m = None


def edit(src):
    """The two rewrites; raises if the source is not the form this file knows (exactly one K loop with fp16 MFMAs and the four (half_t) casts)."""
    n = src.count("v_mfma_f32_16x16x16f16")
    casts = _CAST.findall(src)
    if n == 0 or len(casts) != 4 or src.count("float32x4 v_=cc[i5];") != 1:
        raise RuntimeError(f"gemm_bf16: unexpected hand source ({n} MFMAs, {len(casts)} casts)")
    src = src.replace("v_mfma_f32_16x16x16f16", MFMA)
    first = [True]

    def sub(m):
        if first[0]:
            first[0] = False
            return "u.x=hb16_(v_[0])|(hb16_(v_[1])<<16); u.y=hb16_(v_[2])|(hb16_(v_[3])<<16);"
        return ""
    src = _CAST.sub(sub, src)
    i = src.index("__device__ inline __attribute__((always_inline)) void hand_body(") if "void hand_body(" in src else src.index('extern "C" __global__')
    return src[:i] + _HELPER + src[i:]


def module():
    """The private copy of gemm_prefetch with the rewrite installed. GEMM_PREFETCH names the file; default: next to this one."""
    global _m
    if _m is None:
        p = os.environ.get("GEMM_PREFETCH") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "gemm_prefetch.py")
        spec = importlib.util.spec_from_file_location("gemm_prefetch_bf16", p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        for nm in ("hand_source", "hand_source_pgr"):
            orig = getattr(m, nm)

            def w(*a, _o=orig, **k):
                src, dyn = _o(*a, **k)
                return edit(src), dyn
            setattr(m, nm, w)
        _m = m
    return _m


def _wrap(k):
    f16, bf = torch.float16, torch.bfloat16
    return lambda a, b: k(a.view(f16), b.view(f16)).view(bf)


def gemm_auto(M, N, K, placed=False):
    """(a bf16 [M, K], w bf16 [N, K]) -> bf16 [M, N] = a @ w^T, or None where gemm_prefetch has no hand form for the shape."""
    m = module()
    mode = m.select_mode(M, N, K, placed)
    if mode not in ("fill", "hand"):
        return None
    k = m.gemm_auto(M, N, K, mode, placed)
    return None if k is None else _wrap(k)


def gemm_small_m(M, N, K):
    """As gemm_prefetch.gemm_small_m, bf16. The weight must come from small_m_b (zero rows at the SMALL_M_PAD_N shapes; bf16 zero is fp16 zero)."""
    k = module().gemm_small_m(M, N, K)
    return None if k is None else _wrap(k)


def small_m_b(M, w):
    return module().small_m_b(M, w.view(torch.float16)).view(torch.bfloat16)


# ---- rows form with a tile per (rows, shape): the dispatch the serving plugin uses (track G2, 11 Nov 2026) ----
# Schedules of gemm_prefetch.gemm_hand_rows by the names gm_tiles.py uses. cn 1 everywhere: C has the true N columns and is contiguous.
SCHEDS = {"xs4": dict(pad=0, plw=1, xs=1, cn=1), "plw": dict(pad=32, plw=1, cn=1), "plw16": dict(pad=16, plw=1, cn=1),
          "x8_42": dict(pad=0, plw=1, nw=8, wg=(4, 2), xs=1, cn=1), "x8_24": dict(pad=0, plw=1, nw=8, wg=(2, 4), xs=1, cn=1)}
# (M, N, K) -> (bM, bN, schedule): only entries whose kernel beat vLLM's own path (torch F.linear on bf16 = hipBLASLt) above the identical-code noise in two opposite-order one-GCD jobs.
# A shape that is not here has no landed form: gemm_rows returns None and the caller keeps its own path. Evidence: evidence/G2/RESULTS.md.
ROWS_FORMS = {
    # BEGIN ROWS_FORMS
    (784, 5120, 3072): (160, 256, "xs4"),
    (784, 5120, 8704): (160, 256, "xs4"),
    (784, 7168, 5120): (128, 128, "xs4"),
    (784, 8192, 5120): (224, 160, "plw"),
    (1568, 5120, 3072): (160, 256, "xs4"),
    (1568, 7168, 5120): (224, 160, "xs4"),
    (1568, 8192, 5120): (160, 256, "xs4"),
    (1568, 17408, 5120): (224, 224, "xs4"),
    (1600, 8192, 5120): (160, 256, "xs4"),
    (2352, 5120, 3072): (224, 256, "xs4"),
    (2352, 5120, 8704): (224, 256, "xs4"),
    (2352, 7168, 5120): (160, 256, "xs4"),
    (2352, 8192, 5120): (192, 256, "xs4"),
    (2352, 17408, 5120): (224, 256, "xs4"),
    (3136, 5120, 3072): (224, 224, "xs4"),
    (3136, 5120, 8704): (224, 224, "xs4"),
    (3136, 7168, 5120): (224, 192, "xs4"),
    (3136, 8192, 5120): (192, 256, "xs4"),
    (3136, 17408, 5120): (224, 224, "xs4"),
    (3920, 5120, 3072): (192, 256, "x8_24"),
    (3920, 5120, 8704): (192, 256, "x8_24"),
    (3920, 7168, 5120): (128, 128, "xs4"),
    (3920, 17408, 5120): (192, 256, "xs4"),
    (4704, 5120, 3072): (224, 256, "xs4"),
    (4704, 5120, 8704): (224, 256, "xs4"),
    (4704, 7168, 5120): (128, 128, "xs4"),
    (4704, 8192, 5120): (224, 160, "xs4"),
    (4704, 17408, 5120): (224, 256, "xs4"),
    (5488, 5120, 3072): (128, 128, "xs4"),
    (5488, 5120, 8704): (256, 128, "xs4"),
    (5488, 7168, 5120): (160, 256, "xs4"),
    (5488, 8192, 5120): (192, 256, "xs4"),
    (5488, 17408, 5120): (192, 256, "xs4"),
    (6272, 5120, 3072): (192, 256, "x8_24"),
    (6272, 5120, 8704): (192, 256, "xs4"),
    (6272, 7168, 5120): (128, 128, "xs4"),
    (6272, 8192, 5120): (224, 192, "xs4"),
    (6272, 17408, 5120): (224, 224, "xs4"),
    (7056, 5120, 3072): (192, 256, "x8_24"),
    (7056, 5120, 8704): (192, 256, "x8_24"),
    (7056, 7168, 5120): (192, 256, "x8_24"),
    (7056, 8192, 5120): (192, 256, "x8_24"),
    (7056, 17408, 5120): (192, 256, "x8_24"),
    (7840, 5120, 3072): (160, 256, "xs4"),
    (7840, 5120, 8704): (160, 256, "xs4"),
    (7840, 7168, 5120): (224, 256, "xs4"),
    (7840, 8192, 5120): (192, 256, "x8_24"),
    (7840, 17408, 5120): (224, 224, "xs4"),
    (8624, 5120, 3072): (160, 256, "xs4"),
    (8624, 5120, 8704): (160, 256, "xs4"),
    (8624, 7168, 5120): (224, 256, "xs4"),
    (8624, 8192, 5120): (160, 256, "xs4"),
    (8624, 17408, 5120): (192, 256, "xs4"),
    (9408, 5120, 3072): (192, 256, "x8_24"),
    (9408, 5120, 8704): (192, 256, "xs4"),
    (9408, 7168, 5120): (224, 256, "xs4"),
    (9408, 8192, 5120): (192, 256, "x8_24"),
    (9408, 17408, 5120): (224, 256, "xs4"),
    (10192, 5120, 3072): (192, 256, "x8_24"),
    (10192, 5120, 8704): (192, 256, "xs4"),
    (10192, 7168, 5120): (224, 256, "xs4"),
    (10192, 8192, 5120): (224, 192, "xs4"),
    (10192, 17408, 5120): (224, 224, "xs4"),
    (10976, 5120, 3072): (224, 256, "xs4"),
    (10976, 5120, 8704): (224, 256, "xs4"),
    (10976, 7168, 5120): (192, 256, "xs4"),
    (10976, 8192, 5120): (192, 256, "x8_24"),
    (10976, 17408, 5120): (224, 224, "xs4"),
    (11760, 5120, 3072): (224, 256, "xs4"),
    (11760, 5120, 8704): (224, 256, "xs4"),
    (11760, 7168, 5120): (192, 256, "xs4"),
    (11760, 8192, 5120): (224, 224, "xs4"),
    (11760, 17408, 5120): (224, 256, "xs4"),
    (12544, 5120, 3072): (192, 256, "x8_24"),
    (12544, 5120, 8704): (192, 256, "xs4"),
    (12544, 7168, 5120): (192, 256, "x8_24"),
    (12544, 8192, 5120): (224, 224, "xs4"),
    (12544, 17408, 5120): (224, 224, "xs4"),
    (13328, 5120, 3072): (224, 256, "xs4"),
    (13328, 5120, 8704): (224, 256, "xs4"),
    (13328, 7168, 5120): (192, 256, "xs4"),
    (13328, 8192, 5120): (224, 256, "xs4"),
    (13328, 17408, 5120): (224, 224, "xs4"),
    (14112, 5120, 3072): (192, 256, "x8_24"),
    (14112, 5120, 8704): (224, 256, "xs4"),
    (14112, 7168, 5120): (192, 256, "xs4"),
    (14112, 8192, 5120): (192, 256, "x8_24"),
    (14112, 17408, 5120): (224, 256, "xs4"),
    (14896, 5120, 3072): (192, 256, "x8_24"),
    (14896, 5120, 8704): (192, 256, "xs4"),
    (14896, 7168, 5120): (192, 256, "xs4"),
    (14896, 8192, 5120): (192, 256, "x8_24"),
    (14896, 17408, 5120): (224, 256, "xs4"),
    (15680, 5120, 3072): (192, 256, "x8_24"),
    (15680, 5120, 8704): (192, 256, "xs4"),
    (15680, 7168, 5120): (192, 256, "xs4"),
    (15680, 8192, 5120): (192, 256, "xs4"),
    (15680, 17408, 5120): (192, 256, "xs4"),
    # END ROWS_FORMS
}


def gemm_rows(M, N, K, form=None):
    """(a bf16 [M, K] contiguous, w bf16 [N, K] contiguous) -> bf16 [M, N] = a @ w^T on the landed tile for this shape, or None if ROWS_FORMS has none.
    form (bM, bN, schedule name) overrides the table (for races). M need not be a tile multiple: rows past M read zeros through the buffer range check; a must have exactly M rows."""
    f = form or ROWS_FORMS.get((M, N, K))
    if f is None:
        return None
    bM, bN, sched = f
    kw = dict(SCHEDS[sched])
    k = module().gemm_hand_rows(M, N, K, bM, bN, kw.pop("pad"), **kw)
    f16, bf = torch.float16, torch.bfloat16
    if M % bM == 0:
        return lambda a, w: k(a.view(f16), w.view(f16)).view(bf)
    return lambda a, w: k(a.view(f16), w.view(f16))[:M].view(bf)
