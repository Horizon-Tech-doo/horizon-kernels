# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# K4 phase 2a GPU job (short, under the GPU lock): the persistent runtime's counters on 188 co-resident CTAs.
#   1. timeout: a task waiting for a signal that never comes raises err and the kernel returns (no hang)
#   2. stress: random task DAGs, 10^6 tasks over 20 launches (seq changes every launch), L2 reads (rd=1): zero bad
#      values, zero errors, counters back at zero after every launch; the same through a CUDA graph replay
#   3. the hazard the rules exist for: the same with plain (L1-cached) loads (rd=0): bad values expected
#   4. hand-over latency: a strict chain (task i waits for task i-1, consecutive tasks on different CTAs)
# usage: run_p1_job.sh rt rt_stress.py OUT.jsonl
import json, sys, time
import torch
from tilelang.contrib.horizon_k4 import selftest as st, runtime as rt

out = open(sys.argv[1], "w"); T0 = time.time()


def emit(d):
    d["t"] = round(time.time() - T0, 1); print(json.dumps(d), flush=True); out.write(json.dumps(d) + "\n"); out.flush()


NCTA = torch.cuda.get_device_properties(0).multi_processor_count
emit({"sms": NCTA})
W = 8


def launch(k, ts, B, bad, seq):
    k(ts["desc"], ts["sigtab"], ts["ntask"], ts["counters"], ts["done"], ts["err"], B, bad, seq)


# 1. timeout
k_to = st.dag_kernel(NCTA, W, 1, 128, 50_000_000)
sc = st.build_dag(2000, NCTA, unsatisfiable=True)
ts = sc.tensors(); B = torch.zeros(2000, W, dtype=torch.int32, device="cuda"); bad = torch.zeros(1, dtype=torch.int32, device="cuda")
t = time.time(); launch(k_to, ts, B, bad, 1); torch.cuda.synchronize(); dt = time.time() - t
emit({"test": "timeout", "err": int(ts["err"][0]), "seconds": round(dt, 3), "ok": int(ts["err"][0]) == 1 and dt < 5})

# 2./3. stress
for rd in (1, 0):
    k = st.dag_kernel(NCTA, W, rd)
    tot_tasks = tot_bad = tot_err = 0; nonzero = 0; times = []
    for it in range(20):
        sc = st.build_dag(50000, NCTA, window=4096, seed=it)
        ts = sc.tensors(); B = torch.zeros(50000, W, dtype=torch.int32, device="cuda"); bad.zero_()
        launch(k, ts, B, bad, 100 + it); torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record(); launch(k, ts, B, bad, 200 + it); ev1.record(); torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1) * 1000)
        tot_tasks += 2 * 50000; tot_bad += int(bad[0]); tot_err += int(ts["err"][0])
        nonzero += int((ts["counters"] != 0).sum()) + int(ts["done"][0] != 0)
        if rd == 0 and it >= 3:
            break
    emit({"test": "stress", "rd": rd, "tasks": tot_tasks, "bad_values": tot_bad, "errors": tot_err,
          "counters_not_reset": nonzero, "us_per_launch_med": round(sorted(times)[len(times) // 2], 1),
          "tasks_per_cta": sc.stats()["max_per_cta"],
          "ok": (tot_bad == 0 and tot_err == 0 and nonzero == 0) if rd == 1 else None})

# graph replay (rd=1)
k = st.dag_kernel(NCTA, W, 1)
sc = st.build_dag(50000, NCTA, window=4096, seed=77); ts = sc.tensors()
B = torch.zeros(50000, W, dtype=torch.int32, device="cuda"); bad.zero_()
launch(k, ts, B, bad, 5); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
for _ in range(4):
    try:
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            launch(k, ts, B, bad, 5)
        break
    except RuntimeError:
        torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
gb = 0
for r in range(50):
    B.zero_(); g.replay()
torch.cuda.synchronize()
emit({"test": "graph_replay", "replays": 50, "bad_values": int(bad[0]), "errors": int(ts["err"][0]),
      "counters_not_reset": int((ts["counters"] != 0).sum()), "ok": int(bad[0]) == 0 and int(ts["err"][0]) == 0})

# 4. chain latency
for n in (2000, 20000):
    sc = st.build_dag(n, NCTA, chain=True); ts = sc.tensors()
    B = torch.zeros(n, W, dtype=torch.int32, device="cuda"); bad.zero_()
    launch(k, ts, B, bad, 9); torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    best = 1e9
    for _ in range(5):
        ev0.record(); launch(k, ts, B, bad, 9); ev1.record(); torch.cuda.synchronize(); best = min(best, ev0.elapsed_time(ev1) * 1000)
    emit({"test": "chain", "hops": n, "us_total": round(best, 1), "ns_per_hop": round(best * 1000 / n, 1),
          "bad_values": int(bad[0]), "errors": int(ts["err"][0])})
print("RT_STRESS_DONE")
