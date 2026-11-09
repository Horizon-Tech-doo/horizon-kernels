"""Landed gqa_flash_fast through the real module, correctness only (compile + fp64 + repeat). Usage: land_test.py"""
import json, torch, gqa_fa
from n5lib import inputs, ref
torch.manual_seed(0)
for name, B, L, Hkv, G, D, causal, P, bN, th, edge, rev, recip, swap, rows in [
    ("d256_8k", 1, 8192, 4, 6, 256, True, 32, 32, 768, True, True, True, True, 0),
    ("d128_4k_r128", 1, 4096, 4, 6, 128, True, 0, 32, 512, True, True, True, True, 128),
    ("d128_4k_P32", 1, 4096, 4, 6, 128, True, 32, 96, 768, True, True, True, True, 0),
    ("mha_4k_P160", 1, 4096, 8, 1, 128, False, 160, 32, 640, True, False, False, False, 0),
    ("mha_4k_P192", 1, 4096, 8, 1, 128, False, 192, 96, 768, True, False, False, False, 0),
    ("d128_b2_r128", 2, 1024, 4, 6, 128, True, 0, 32, 512, True, True, True, True, 128),
    ("d256_b2", 2, 2048, 4, 6, 256, True, 32, 32, 768, True, True, True, True, 0),
    ("d128_L4000_P32", 1, 4000, 4, 6, 128, True, 32, 96, 768, True, True, True, True, 0),
    ("d128_L4000_r128", 1, 4000, 4, 6, 128, True, 0, 32, 512, True, True, True, True, 128),
    ("mha_L1000", 1, 1000, 8, 1, 128, False, 160, 32, 640, True, False, False, False, 0),
]:
    g = torch.Generator().manual_seed(7)
    f = dict(device="cuda", dtype=torch.float16)
    Q = torch.randn(B, L, Hkv * G, D, generator=g).to(**f); K = torch.randn(B, L, Hkv, D, generator=g).to(**f); V = torch.randn(B, L, Hkv, D, generator=g).to(**f)
    R = ref(Q, K, V, causal, G)
    r = dict(name=name)
    try:
        k = gqa_fa.gqa_flash_fast(B, L, Hkv, G, D, causal, P, bN, th, 2, edge, rev, recip, swap, rows)
        O = k(Q, K, V)
        r["rel_vs_fp64"] = round(((O.double() - R).norm() / R.norm()).item(), 7)
        r["repeat_equal"] = sum(int(torch.equal(k(Q, K, V), O)) for _ in range(20))
        # plain vs fast must not share a cache entry
        k0 = gqa_fa.gqa_flash_fwd(B, L, Hkv, G, D, causal, P, bN, th, True, 2, edge, rev, recip, swap, rows)
        O0 = k0(Q, K, V)
        r["plain_rel"] = round(((O0.double() - R).norm() / R.norm()).item(), 7)
        r["max_diff_plain_fast"] = round((O0.float() - O.float()).abs().max().item(), 6)
    except Exception as e:  # noqa: BLE001
        r["err"] = (type(e).__name__ + ": " + str(e))[-300:]
    print(json.dumps(r), flush=True)
