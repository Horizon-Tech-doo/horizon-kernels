# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Summarise xbuild_race.jsonl: fastest round per (kernel, shape, M, build), next/cur ratio per cell, and the
# layer-weighted total (Qwen3.8-27B: 48 GDN layers (qkvz, o/out_proj), 16 attention (qkv, o_proj), 64 MLP (gate_up,
# down)) per kernel family and M.   xbuild_race_summary.py <jsonl> [cur-tag] [next-tag]
import json, sys
from collections import defaultdict
path = sys.argv[1]
A, B = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("cur", "next")
W = {"qkvz": 48, "qkv": 16, "o": 64, "gate_up": 64, "down": 64}
best = defaultdict(dict)
nround = defaultdict(set)
for l in open(path):
    r = json.loads(l)
    k = (r["kernel"], r["shape"], r["M"])
    best[k][r["build"]] = min(best[k].get(r["build"], 1e18), r["us"])
    nround[r["build"]].add(r["round"])
print(f"rounds: {dict((b, sorted(v)) for b, v in nround.items())}")
kernels = sorted({k[0] for k in best})
worst = {}
for kern in kernels:
    cells = {(s, m): v for (kk, s, m), v in best.items() if kk == kern and A in v and B in v}
    ms = sorted({m for _, m in cells})
    print(f"== {kern}: us {B} / {A} (ratio)")
    for m in ms:
        parts, to, tn = [], 0.0, 0.0
        for (s, mm), v in sorted(cells.items()):
            if mm != m:
                continue
            ratio = v[B] / v[A]
            worst[kern] = max(worst.get(kern, 0), ratio)
            parts.append(f"{s} {v[B]:.2f}/{v[A]:.2f} ({ratio:.3f})")
            w = W.get(s, 1)
            to += w * v[A]; tn += w * v[B]
        print(f"  M={m}: " + " | ".join(parts) + f" || weighted {tn / to:.4f}")
print("worst cell per kernel:", {k: round(v, 4) for k, v in worst.items()})
