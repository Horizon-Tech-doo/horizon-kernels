# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Compile K4 segment-kernel variants into the TileLang JIT disk cache with N processes in parallel (the GPU is only
# used to load each module). usage: python seg_compile.py JOBS_JSON [procs]
#   JOBS_JSON: list of [ncta, WM, [s_out, s_gu, s_down, s_q], stages, stamps, pf_kt, nq, hw]
import json, sys, time


def _compile(job):
    from tilelang.contrib.horizon_k4 import segment as sg
    ncta, WMb, splits, stages, st_, pf_kt, nq, hw = job
    t0 = time.time()
    sg.segment_kernel(ncta, WMb, *splits, sg.SK_CFG[0], sg.SK_CFG[1], stages, 1, 1e-6, 1, st_, 2_000_000_000, pf_kt,
                      0, nq, hw)
    return job, round(time.time() - t0, 1)


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = json.loads(sys.argv[1])
    with mp.get_context("spawn").Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 4) as pool:
        for job, dt in pool.imap_unordered(_compile, jobs):
            print(json.dumps({"compiled": job, "s": dt}), flush=True)
    print("SEG_COMPILE_DONE")
