# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Summarise rs_race.jsonl: per cell arm/K3 and the layer-weighted total per M (48 GDN: qkvz + o; 16 attention: qkv
# + o; 64 MLP: gate_up + down -> weights qkvz 48, qkv 16, o 64, gate_up 64, down 64), for gemm-only and full linear().
#   rs_summary.py <jsonl> [start_line]
import json, sys
W = {"qkvz": 48, "qkv": 16, "o": 64, "gate_up": 64, "down": 64}
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip() and l.startswith("{\"shape")]
if len(sys.argv) > 2:
    rows = rows[int(sys.argv[2]):]
cells = {(r["shape"], r["M"]): r for r in rows}
ms = sorted({m for _, m in cells})
for kind in ("gemm", "full"):
    for arm in ("rf", "rs1", "rs0", "rs1i"):
        if not any(f"{arm}_{kind}_us" in r for r in cells.values()):
            continue
        print(f"== {arm} {kind}: us {arm} / k3 (ratio)")
        worst, allw = 0.0, []
        for m in ms:
            to = tn = 0.0
            parts = []
            for sh in W:
                r = cells.get((sh, m))
                if not r or f"{arm}_{kind}_us" not in r:
                    continue
                o, n = r[f"k3_{kind}_us"], r[f"{arm}_{kind}_us"]
                to += W[sh] * o; tn += W[sh] * n
                worst = max(worst, n / o)
                ident = r.get("bit_identical_vs_k3", {}).get(arm, True)
                parts.append(f"{sh} {n:.1f}/{o:.1f} ({n / o:.3f}){'' if ident else ' NOT-IDENTICAL'}")
            if to:
                allw.append(tn / to)
                print(f"  M={m}: " + " | ".join(parts) + f" || weighted {tn / to:.4f}")
        print(f"  worst cell {worst:.4f}; weighted range {min(allw):.4f}..{max(allw):.4f}")
