# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Completeness + gate verifier for test_gdn_state_gpu.py results (JSON lines). CPU only.

usage: python verify_gdn_state_results.py RESULTS.jsonl   (exit 0 = complete and every correctness gate passed)
Gates (all required):
  window        our G1S kernel vs vLLM's kernel, float32 and bfloat16 state, nr 1/2/8/32 x QL 2/4: out and every state
                slot bit-identical (+0/-0 treated equal)
  window_fp16   float16 state vs our float32 kernel + torch RNE/saturating rounding of the written states: bit-identical
  replay_chain  G3 float32 chained vs vLLM's kernel chained, nr 2/8/32: outputs bit-identical every step, checkpoint
                after materialize bit-identical to vLLM's slot state_idx[r, n_acc - 1], at least one flush and one
                materialize per row
  graph         determinism (3 runs) + CUDA-graph replay == eager for window f32/bf16/f16 and replay f32/bf16
  bench         nr 1/8/32 x QL 2/4, every arm a finite positive time
  end           terminal record complete = true
Speed and drift rows are reported, never gated here (no automatic promotion).
"""
import json
import math
import sys

WINDOW_NR = (1, 2, 8, 32)
QLS = (2, 4)
CHAIN_NR = (2, 8, 32)
GRAPH_KERNELS = ("window_float32", "window_bfloat16", "window_float16", "replay_float32", "replay_bfloat16")
BENCH_NR = (1, 8, 32)
BENCH_ARMS = ("vllm_f32", "vllm_bf16", "g1s_f32", "g1s_bf16", "g1s_f16", "g3_f32", "g3_bf16", "g3_f16")


def verify(recs):
    by = {}
    for r in recs:
        by.setdefault(r.get("test"), []).append(r)
    for sdt in ("float32", "bfloat16"):
        for nr in WINDOW_NR:
            for ql in QLS:
                rows = [r for r in by.get("window", []) if r.get("sdt") == sdt and r.get("nr") == nr
                        and r.get("ql") == ql]
                if not rows:
                    return False, f"missing window row {sdt} nr={nr} ql={ql}"
    for r in by.get("window", []):
        if r.get("out_diff") != 0 or r.get("state_diff") != 0:
            return False, f"window mismatch {r.get('case')}: out {r.get('out_diff')} state {r.get('state_diff')}"
    f16 = by.get("window_fp16", [])
    if len(f16) < 4:
        return False, "fewer than 4 window_fp16 rows"
    for r in f16:
        if r.get("out_diff") != 0 or r.get("state_diff") != 0:
            return False, f"window_fp16 mismatch {r.get('case')}"
    for nr in CHAIN_NR:
        rows = [r for r in by.get("replay_chain", []) if r.get("nr") == nr]
        if not rows:
            return False, f"missing replay_chain nr={nr}"
        for r in rows:
            if r.get("out_diff") != 0 or r.get("ckpt_diff") != 0:
                return False, f"replay_chain mismatch nr={nr}: out {r.get('out_diff')} ckpt {r.get('ckpt_diff')}"
            if not (r.get("steps", 0) > 0 and r.get("flushes", 0) > 0 and r.get("materializes", 0) > 0):
                return False, f"replay_chain nr={nr} did not exercise steps/flushes/materializes"
    for k in GRAPH_KERNELS:
        rows = [r for r in by.get("graph", []) if r.get("kernel") == k]
        if not rows:
            return False, f"missing graph row {k}"
        if not all(r.get("deterministic") is True and r.get("graph_replay_equal") is True for r in rows):
            return False, f"graph/determinism failed for {k}"
    for nr in BENCH_NR:
        for ql in QLS:
            rows = [r for r in by.get("bench", []) if r.get("nr") == nr and r.get("ql") == ql]
            if not rows:
                return False, f"missing bench nr={nr} ql={ql}"
            us = rows[-1].get("us", {})
            for arm in BENCH_ARMS:
                v = us.get(arm)
                if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                    return False, f"bench nr={nr} ql={ql} arm {arm} invalid: {v}"
    ends = by.get("end", [])
    if not ends or ends[-1].get("complete") is not True:
        return False, "no complete terminal record"
    return True, "complete, all correctness gates passed"


def synthetic_complete():
    recs = []
    for sdt in ("float32", "bfloat16"):
        for nr in WINDOW_NR:
            for ql in QLS:
                recs.append({"test": "window", "case": f"{sdt}_nr{nr}_ql{ql}", "sdt": sdt, "nr": nr, "ql": ql,
                             "out_diff": 0, "state_diff": 0})
    for i in range(4):
        recs.append({"test": "window_fp16", "case": f"f16_{i}", "out_diff": 0, "state_diff": 0})
    for nr in CHAIN_NR:
        recs.append({"test": "replay_chain", "nr": nr, "half": 16, "steps": 100, "out_diff": 0, "ckpt_diff": 0,
                     "flushes": 10, "materializes": 3})
    for k in GRAPH_KERNELS:
        recs.append({"test": "graph", "kernel": k, "deterministic": True, "graph_replay_equal": True})
    for nr in BENCH_NR:
        for ql in QLS:
            recs.append({"test": "bench", "nr": nr, "ql": ql, "us": {a: 10.0 + i for i, a in enumerate(BENCH_ARMS)}})
    recs.append({"test": "end", "complete": True})
    return recs


if __name__ == "__main__":
    recs = []
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    ok, why = verify(recs)
    print(("PASS: " if ok else "FAIL: ") + why)
    sys.exit(0 if ok else 1)
