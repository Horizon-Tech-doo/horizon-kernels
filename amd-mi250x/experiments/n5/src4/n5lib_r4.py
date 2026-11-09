"""Round 4 cases on top of n5lib.CASES: off-grid lengths for d128 and d256, MHA at the rule-hole lengths, MHA batch."""


def extend(CASES):
    for L in (1536, 2048, 2560):
        CASES[f"d128_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=128, causal=True)
    for L in (3072, 4160, 5000, 5120, 6144, 10240, 12288):
        CASES[f"d256_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=256, causal=True)
        CASES[f"d128_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=128, causal=True)
    for L in (4160, 4320, 4608, 5120, 6144, 7168):
        CASES[f"mha_{L}"] = dict(B=1, L=L, Hkv=8, G=1, D=128, causal=False)
    for B, L in ((4, 1024), (16, 1024), (4, 4096), (2, 8192)):
        CASES[f"mhaB{B}_{L}"] = dict(B=B, L=L, Hkv=8, G=1, D=128, causal=False)
