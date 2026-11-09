"""Round 3 cases on top of n5lib.CASES: MHA at grid-fill lengths, batch cases, real inputs by name."""
import torch

REAL = "/scratch/project_XXXXXXXXX/tilelang/exp/N5/real"


def real_inputs(kind, s):
    """kind realN: captured Q, K, V of full-attention layer N of Qwen3.8-27B on real tokens (capture_real.py). D 256 G 6 is the model's own shape. D 128 keeps the first 128 dims and is real-derived, not a model shape. MHA (Hkv 8, G 1): Q heads 0-7 and K, V from this layer's 4 kv heads and layer 15's or 3's 4 kv heads: statistics only. Batch: B consecutive slices."""
    n = int(kind[4:])
    B, L, Hkv, G, D = s["B"], s["L"], s["Hkv"], s["G"], s["D"]
    src = next(x for x in (4096, 8192, 16384) if x >= B * L)
    x = torch.load(f"{REAL}/real_L{src}_l{n}.pt")
    Q, K, V = x["Q"][0, :B * L], x["K"][0, :B * L], x["V"][0, :B * L]
    if Hkv == 8:
        y = torch.load(f"{REAL}/real_L{src}_l{3 if n != 3 else 15}.pt")
        Q, K, V = Q[:, :8], torch.cat([K, y["K"][0, :B * L]], 1), torch.cat([V, y["V"][0, :B * L]], 1)
    f = dict(device="cuda", dtype=torch.float16)
    sh = lambda t: t[..., :D].reshape(B, L, t.shape[1], D).contiguous().to(**f).clone(memory_format=torch.contiguous_format)  # noqa: E731
    return sh(Q), sh(K), sh(V)


def extend(CASES):
    for L in (4160, 4320, 4608, 5120):
        CASES[f"mha_{L}"] = dict(B=1, L=L, Hkv=8, G=1, D=128, causal=False)
    for L in (2048, 6144, 12288):
        CASES[f"mha_{L}"] = dict(B=1, L=L, Hkv=8, G=1, D=128, causal=False)
    for B, L in ((4, 1024), (16, 1024), (4, 4096), (2, 8192)):
        CASES[f"d128B{B}_{L}"] = dict(B=B, L=L, Hkv=4, G=6, D=128, causal=True)
        CASES[f"d256B{B}_{L}"] = dict(B=B, L=L, Hkv=4, G=6, D=256, causal=True)
        CASES[f"mhaB{B}_{L}"] = dict(B=B, L=L, Hkv=8, G=1, D=128, causal=False)
