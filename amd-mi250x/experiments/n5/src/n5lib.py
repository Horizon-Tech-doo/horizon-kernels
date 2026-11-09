"""N5 shared parts: cases, inputs, arms. Inputs are built on the CPU with a fixed generator, so every arm and both stacks see the same data."""
import torch

CASES = {}
for L in (1024, 4096, 8192, 16384):
    CASES[f"d256_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=256, causal=True)
    CASES[f"d128_{L}"] = dict(B=1, L=L, Hkv=4, G=6, D=128, causal=True)
    CASES[f"mha_{L}"] = dict(B=1, L=L, Hkv=8, G=1, D=128, causal=False)


def inputs(kind, s):
    B, L, Hkv, G, D = s["B"], s["L"], s["Hkv"], s["G"], s["D"]
    H = Hkv * G
    seed = {"randn0": 0, "randn1": 1, "self": 2, "big": 3, "ar": 4, "ramp": 5}[kind]
    g = torch.Generator().manual_seed(seed)
    r = lambda sh: torch.randn(*sh, generator=g)  # noqa: E731
    Q, K, V = r((B, L, H, D)), r((B, L, Hkv, D)), r((B, L, Hkv, D))
    if kind == "self":        # queries equal their kv head's keys: sharply peaked diagonal
        Q = K.repeat_interleave(G, 2)
    elif kind == "big":       # large scores: very peaked softmax, max moves late
        Q, K = Q * 8, K * 8
    elif kind == "ar":        # smooth along position, correlated rows, like real embeddings more than randn
        a = 0.99
        b = (1 - a * a) ** 0.5
        for x in (Q, K, V):
            for t in range(1, L):
                x[:, t] = a * x[:, t - 1] + b * x[:, t]
    elif kind == "ramp":      # scores grow with key position: the row max moves on nearly every K tile
        w = (1 + 6 * torch.arange(L, dtype=torch.float32) / L).view(1, L, 1, 1)
        K = K * w
    f = dict(device="cuda", dtype=torch.float16)
    return Q.to(**f), K.to(**f), V.to(**f)


def ref(Q, K, V, causal, G, dt=torch.float64):
    F = torch.nn.functional
    q, k, v = Q.to(dt), K.to(dt).repeat_interleave(G, 2), V.to(dt).repeat_interleave(G, 2)
    s_ = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    if causal:
        n = s_.size(-1)
        s_ = s_.masked_fill(torch.ones(n, n, device=s_.device).tril() == 0, float("-inf"))
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(s_, -1), v)


def arm(cases, P, bN, th, flags="", form="land", mode="full", vp=1, rows=0):
    """flags: E edge, R reverse, D reciprocal, T transposed V (e3 form only)."""
    return [dict(name=f"{c}_{'r' + str(rows) if rows else 'P' + str(P)}_b{bN}_t{th}_{form}_{mode}{flags}" + (f"vp{vp}" if vp != 1 else ""), case=c, P=P, bN=bN, th=th,
                 edge="E" in flags, rev="R" in flags, recip="D" in flags, vt="T" in flags, swap="W" in flags, rows=rows, form=form, mode=mode, vp=vp) for c in cases]


def build(c, compile_only=False):
    """Returns (callable(Q,K,V) -> O, or for compile_only the raw kernel function)."""
    s = CASES[c["case"]]
    B, L, Hkv, G, D, causal = s["B"], s["L"], s["Hkv"], s["G"], s["D"], s["causal"]
    if c["form"] == "vt":
        import gqa_vt  # noqa: F401  (writes gqa_vt_gen.py)
        import gqa_vt_gen
        return gqa_vt_gen.fwd(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
    if c["form"] == "land":
        import gqa_fa
        if compile_only:
            raise NotImplementedError
        k = gqa_fa.gqa_flash_fwd(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], True, 2, c["edge"], c["rev"], c["recip"], c["swap"], c["rows"])
        return k
    import gqa_e3b
    k = gqa_e3b.fwd(B, L, Hkv, G, D, causal, c["P"], c["bN"], c["th"], 2, c["edge"], c["recip"], c["mode"], c["vt"], c["vp"], c["rev"])
    return lambda q, kk, v: k(q, kk, v, 0)
