#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""GSM8K screen comparison, using the established -1pp / paired-sign gates."""
import argparse
import json
from math import comb
from pathlib import Path


def compare(p, x):
    pa, xa = p.get("correct", []), x.get("correct", [])
    if len(pa) != 1319 or len(xa) != 1319 or any(type(v) is not bool for v in pa + xa):
        return {"result": "fail", "observed": "requires 1319 paired boolean scores per arm"}
    worse = sum(a and not b for a, b in zip(pa, xa))
    better = sum(b and not a for a, b in zip(pa, xa))
    n = worse + better
    sign_p = min(1, 2 * sum(comb(n, k) for k in range(min(worse, better) + 1)) / 2 ** n) if n else 1
    drop = (sum(pa) - sum(xa)) / len(pa)
    ok = drop <= 0.010 and not (worse > better and sign_p < 0.05)
    return {"result": "pass" if ok else "fail", "params": {"max_drop": 0.010, "alpha": 0.05},
            "observed": {"P_correct": sum(pa), "X_correct": sum(xa), "n": len(pa),
                         "worse": worse, "better": better, "sign_p": sign_p, "accuracy_drop": drop}}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--p", default="P")
    p.add_argument("--x", default="X")
    a = p.parse_args()
    try:
        result = compare(json.loads((a.data / f"gsm-{a.p}.json").read_text()),
                         json.loads((a.data / f"gsm-{a.x}.json").read_text()))
    except (OSError, ValueError):
        result = {"result": "fail", "observed": "GSM8K results missing or invalid"}
    a.out.write_text(json.dumps({"gates": {"gsm8k_load": result},
                                "summary_md": f"GSM8K under load: {result['result']}"}, indent=2) + "\n")
