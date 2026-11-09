# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K4 runtime v2 GPU job (short): counter hand-over latency per variant (strict chain: task i waits for task i-1, on
# another CTA) and the random-DAG stress (10^6 tasks) for the default variant (pure spin, release-add without fence).
# usage: python rt_latency.py OUT.jsonl
import json, sys, time
import torch
from tilelang.contrib.horizon_k4 import selftest as st

out = open(sys.argv[1], "w"); T0 = time.time()


def emit(d):
    d["t"] = round(time.time() - T0, 1); print(json.dumps(d), flush=True); out.write(json.dumps(d) + "\n"); out.flush()


NCTA = torch.cuda.get_device_properties(0).multi_processor_count
W = 8
bad = torch.zeros(1, dtype=torch.int32, device="cuda")


def launch(k, ts, B, seq):
    k(ts["desc"], ts["sigtab"], ts["ntask"], ts["counters"], ts["done"], ts["err"], B, bad, seq)


for ncta in (NCTA, 2 * NCTA):
    for sleep_ns, fence in ((0, 0), (0, 1), (64, 0)):
        k = st.dag_kernel(ncta, W, 1, 128, 2_000_000_000, sleep_ns, fence)
        n = 20000
        sc = st.build_dag(n, ncta, chain=True); ts = sc.tensors()
        B = torch.zeros(n, W, dtype=torch.int32, device="cuda"); bad.zero_()
        launch(k, ts, B, 9); torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        best = 1e9
        for _ in range(5):
            e0.record(); launch(k, ts, B, 9); e1.record(); torch.cuda.synchronize()
            best = min(best, e0.elapsed_time(e1) * 1000)
        emit({"test": "chain", "ncta": ncta, "sleep_ns": sleep_ns, "fence": fence, "ns_per_hop": round(best * 1000 / n, 1),
              "bad": int(bad[0]), "err": int(ts["err"][0])})
        if (sleep_ns, fence) == (0, 0):
            tot = bb = ee = nz = 0
            for it in range(10):
                sc = st.build_dag(50000, ncta, window=4096, seed=it); ts = sc.tensors()
                B = torch.zeros(50000, W, dtype=torch.int32, device="cuda"); bad.zero_()
                launch(k, ts, B, 300 + it); launch(k, ts, B, 400 + it); torch.cuda.synchronize()
                tot += 100000; bb += int(bad[0]); ee += int(ts["err"][0]); nz += int((ts["counters"] != 0).sum())
            emit({"test": "stress", "ncta": ncta, "tasks": tot, "bad": bb, "err": ee, "not_reset": nz,
                  "ok": bb == 0 and ee == 0 and nz == 0})
print("RT_LATENCY_DONE")
