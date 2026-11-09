# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Summarise rega_race.jsonl: per arm, layer-weighted time / K3 per M (48 GDN: qkvz + o; 16 attention: qkv + o;
# 64 MLP: gate_up + down -> weights qkvz 48, qkv 16, o 64, gate_up 64, down 64), worst cell, and bit-identity vs K3.
#   rega_summary.py <jsonl> [kind=gemm|full]
import json, sys
W = {"qkvz": 48, "qkv": 16, "o": 64, "gate_up": 64, "down": 64}
kind = sys.argv[2] if len(sys.argv) > 2 else "gemm"
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{"shape')]
cells = {(r["shape"], r["M"]): r for r in rows}
ms = sorted({m for _, m in cells})
arms = sorted({k[:-len(f"_{kind}_us")] for r in rows for k in r if k.endswith(f"_{kind}_us")} - {"k3"})
for arm in arms:
    print(f"== {arm} ({kind}), time / K3")
    worst, wc, allw, bad = 0.0, None, [], []
    for m in ms:
        to = tn = 0.0
        parts = []
        for sh in W:
            r = cells.get((sh, m))
            if not r or f"{arm}_{kind}_us" not in r:
                continue
            o, n = r[f"k3_{kind}_us"], r[f"{arm}_{kind}_us"]
            to += W[sh] * o
            tn += W[sh] * n
            if n / o > worst:
                worst, wc = n / o, (sh, m)
            if not r.get("bit_identical_vs_k3", {}).get(arm, True):
                bad.append((sh, m))
            parts.append(f"{sh} {n / o:.3f}")
        if to:
            allw.append(tn / to)
            print(f"  M={m:5d}: weighted {tn / to:.4f}   " + " | ".join(parts))
    if allw:
        print(f"  weighted {min(allw):.4f}..{max(allw):.4f}; worst cell {worst:.4f} {wc}; not bit-identical: {bad or 'none'}")
