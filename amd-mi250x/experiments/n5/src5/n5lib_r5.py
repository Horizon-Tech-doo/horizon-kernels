"""Round 5 cases: round 4's plus the d128 grid lengths and the MHA rule-hole cases."""
from n5lib_r4 import extend as _e4


def extend(CASES):
    _e4(CASES)
    for L in (2816, 3584):
        CASES[f"d128_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=128, causal=True)
    for L in (512, 1536, 2048, 2560, 3072, 3456, 3584):
        CASES[f"mha_{L}"] = dict(B=1, L=L, Hkv=8, G=1, D=128, causal=False)
    for B, L in ((2, 1024), (2, 2048), (4, 512)):
        CASES[f"mhaB{B}_{L}"] = dict(B=B, L=L, Hkv=8, G=1, D=128, causal=False)
