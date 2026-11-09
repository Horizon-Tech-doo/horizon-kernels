# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Does a GPU sync from ANOTHER thread break a CUDA-graph capture? (what gpushare's release_early thread does)
#   test_capture_mode.py global|thread_local|relaxed      one mode per process, so a broken capture can't leak
import sys, threading, time, torch

mode = sys.argv[1]
x = torch.randn(1024, 1024, device="cuda")
ref = x @ x
torch.cuda.synchronize()
g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
go, res = threading.Event(), {}


def other():
    go.wait()
    try:
        torch.cuda.synchronize()
        res["other_thread_sync"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["other_thread_sync"] = f"{type(e).__name__}: {str(e).splitlines()[0][:70]}"


t = threading.Thread(target=other)
t.start()
try:
    with torch.cuda.stream(s):
        g.capture_begin(capture_error_mode=mode)
        go.set()
        time.sleep(0.3)        # the other thread syncs now: capture has begun, no kernel launched yet
        y = x @ x
        g.capture_end()
    g.replay()
    torch.cuda.synchronize()
    res["capture"] = "ok, replay correct" if torch.allclose(y, ref, rtol=1e-2, atol=1e-2) else "replay WRONG"
except Exception as e:  # noqa: BLE001
    res["capture"] = f"FAILED {type(e).__name__}: {str(e).splitlines()[0][:70]}"
t.join()
print(f"{mode:12s} {res}", flush=True)
