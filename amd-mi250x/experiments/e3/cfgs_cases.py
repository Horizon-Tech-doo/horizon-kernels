CASES = {
    "d256": dict(B=1, L=8192, Hkv=4, G=6, D=256, causal=True),
    "d128": dict(B=1, L=4096, Hkv=4, G=6, D=128, causal=True),
    "mha": dict(B=1, L=4096, Hkv=8, G=1, D=128, causal=False),
}
