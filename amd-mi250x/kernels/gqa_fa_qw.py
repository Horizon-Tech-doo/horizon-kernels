"""gqa_fa.py with a query window (track P1, 11 Nov 2026): causal attention of the LAST qw positions of a length-L sequence over all L keys. gqa_fa.py is not edited: this file imports a private
copy of its source with a few text changes, and adds gqa_fa_ps's weight scaling to the device source that copy compiles.

Why. vLLM prefills a long prompt in chunks. A chunk of n tokens after ctx cached tokens needs the attention of n query rows over ctx + n keys. gqa_fa's kernel is square: all L query positions.
Running it at L = ctx + n computes ctx rows that are thrown away (the plugin-only form of horizon_lumi_prefill, HLP_CTX), so its cost grows with L^2 and it loses to vLLM's kernel beyond about
9,400 tokens. The kernel's grid is one block per P query positions, and with rev=True block index 0 is the LAST P positions: launching only the first ceil(qw / P) block indices computes exactly
the last rows. Nothing inside a block changes (same tile, same K loop over the keys up to the block's diagonal, same MFMAs, same source edits), so the cost is that of the blocks that run.

Changes to the copy (each must match exactly once, else load() raises):
  make(..., qw=0): the grid's block count is ceil(qw / P) instead of ceil(L / P) when qw is set; the position of a block still counts from the full count.
  _gqa_flash_fast_jit, gqa_flash_fast: pass qw through; compiled kernels get the cache flags -DGQA_QW_<qw> and a PS marker.
  qw needs rev=True and rows=0 (d256_tile's form) and positions L - ceil(qw / P) * P .. L - 1 are computed, so qw need not be a multiple of P.
The output tensor has all L positions; only the window's rows are written, the others are uninitialised memory. Weight scaling: gqa_fa_ps.shift on the device source (2^10; the fp16 MFMA of
gfx90a drops subnormal operands), always on.

Use: m = gqa_fa_qw.load(); k = m.gqa_flash_fast(1, L, Hkv, G, D, True, qw=n, **m.d256_tile(L, 1)); O = k(Q, K, V); rows O[0, L - n:] are the result."""
import importlib.util, os, tempfile

_m = None
_EDITS = [
    ("def make(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack=2, edge=False, rev=False, recip=False, swap=False, rows=0):",
     "def make(B, L, Hkv, G, D, causal, P, bN, threads, q_regs, k_pack=2, edge=False, rev=False, recip=False, swap=False, rows=0, qw=0):"),
    ("    d0, d1 = (B * Hkv, NB) if swap else (NB, B * Hkv)\n",
     "    if qw and (rows or not rev or not causal):\n        raise ValueError('gqa_fa_qw: the query window needs rev=True, rows=0, causal')\n"
     "    NBW = min(NB, -(-qw // P)) if qw else NB\n    d0, d1 = (B * Hkv, NBW) if swap else (NBW, B * Hkv)\n"),
    ("def _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows):\n    return make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)",
     "def _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows, qw=0):\n    return make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows, qw)"),
    ("def gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, threads, k_pack=2, edge=True, rev=True, recip=False, swap=False, rows=0, lb=0, sched=\"\", form=\"\", vt=False):",
     "def gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, threads, k_pack=2, edge=True, rev=True, recip=False, swap=False, rows=0, lb=0, sched=\"\", form=\"\", vt=False, qw=0):"),
    ("            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)\n            return tilelang.compile(f, out_idx=[3], compile_flags=[\"-DGQA_FAST_N5_V2\", \"-DGQA_VT_1\"])",
     "            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows, qw)\n            return tilelang.compile(f, out_idx=[3], compile_flags=[\"-DGQA_FAST_N5_V2\", \"-DGQA_VT_1\", f\"-DGQA_QW_{qw}\"])"),
    ("            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows)\n            flags = [\"-DGQA_FAST_N5_V2\", f\"-DGQA_LB_{lb}\"]",
     "            f = make(B, L, Hkv, G, D, causal, P, bN, threads, True, k_pack, edge, rev, recip, swap, rows, qw)\n            flags = [\"-DGQA_FAST_N5_V2\", f\"-DGQA_QW_{qw}\", f\"-DGQA_LB_{lb}\"]"),
    ("        return _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows)\n",
     "        return _gqa_flash_fast_jit(B, L, Hkv, G, D, causal, P, bN, threads, k_pack, edge, rev, recip, swap, rows, qw)\n"),
]


def load(path=None):
    global _m
    if _m is None:
        here = os.path.dirname(os.path.abspath(__file__))
        p = path or os.environ.get("GQA_FA") or os.path.join(here, "gqa_fa.py")
        bits = int(os.environ.get("GQA_PS_BITS", "10"))
        src = open(p).read()
        for a, b in _EDITS:
            if src.count(a) != 1:
                raise RuntimeError("gqa_fa_qw: gqa_fa.py is not the source this file was written for: " + a[:60])
            src = src.replace(a, b)
        if src.count("-DGQA_FAST_N5_V2") != 3:
            raise RuntimeError("gqa_fa_qw: cache flags not found as expected")
        src = src.replace("-DGQA_FAST_N5_V2", f"-DGQA_FAST_N5_V2_QWPSB{bits}")
        s = importlib.util.spec_from_file_location("gqa_fa_ps_for_qw", os.path.join(os.path.dirname(p) if path else here, "gqa_fa_ps.py"))
        ps = importlib.util.module_from_spec(s)
        s.loader.exec_module(ps)
        d = tempfile.mkdtemp(prefix="gqa_fa_qw_")            # TileLang reads a kernel's Python source back from its file
        f = os.path.join(d, f"gqa_fa_qwpsb{bits}_impl.py")
        with open(f, "w") as o:
            o.write(src)
        spec = importlib.util.spec_from_file_location(f"gqa_fa_qwpsb{bits}_impl", f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        fe, ve = m._fast_edit, m._vt_edit
        m._fast_edit = lambda s_, c: fe(s_, c) if c.get("vt") else ps.shift(fe(s_, c), bits)
        m._vt_edit = lambda s_, c: ps.shift(ve(s_, c), bits)
        from tilelang.contrib import hipcc
        cur = hipcc.compile_hip
        hipcc.compile_hip = lambda code, *a, **k: cur(code, *a, **k)     # no marker: the copy installs its own hook on top (see gqa_fa_ps)
        m._install_hook()
        m.PS_BITS = bits
        _m = m
    return _m
