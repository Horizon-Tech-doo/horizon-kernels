"""Experiment only: gqa_fa.make with V transposed in LDS, for the 4-row staging patch. Generates gqa_vt_gen.py from gqa_fa.py."""
import os, re
s = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa_fa.py")).read()
a = s.index("def make(")
b = s.index("def _block_end(")
src = s[a:b]
src = src.replace("V_s = T.alloc_shared((bN, D), dtype)", "V_t = T.alloc_shared((D, bN), dtype)")
src = src.replace("T.copy(V[b, kv0:kv0 + bN, hk, :], V_s)", "for d, kk in T.Parallel(D, bN):\n                    V_t[d, kk] = V[b, kv0 + kk, hk, d]")
src = src.replace("T.gemm(p_cast, V_s, acc_o, policy=GemmWarpPolicy.FullRow)", "T.gemm(p_cast, V_t, acc_o, transpose_B=True, policy=GemmWarpPolicy.FullRow)")
assert "V_s" not in src
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gqa_vt_gen.py"), "w").write(
    "import tilelang\nimport tilelang.rocm.language as T\nfrom tilelang.tileop.base import GemmWarpPolicy\n\n" + src + "\n\n@tilelang.jit(out_idx=[3])\ndef fwd(B, L, Hkv, G, D, causal, P, bN, th, edge, rev, recip, swap, rows):\n    return make(B, L, Hkv, G, D, causal, P, bN, th, True, 2, edge, rev, recip, swap, rows)\n")
