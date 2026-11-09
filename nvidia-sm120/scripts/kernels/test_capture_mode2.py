# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Which CUDA calls from ANOTHER thread break a CUDA-graph capture, per capture mode?
# The vLLM A/B (26-09-2026) showed thread_local fixes the gpushare startup crash, but test_capture_mode.py showed a full
# sync from another thread breaks every mode, so the culprit must be a call that only "global" forbids.
#   test_capture_mode2.py MODE ACTION      MODE: global|thread_local   ACTION: sync|memgetinfo|malloc|eventquery|none
import ctypes, glob, sys, threading, time, torch

mode, action = sys.argv[1], sys.argv[2]
rt = ctypes.CDLL(glob.glob("$HK_VENV/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so.13")[0])
x = torch.randn(1024, 1024, device="cuda")
ref = x @ x
ev = torch.cuda.Event()
ev.record()
torch.cuda.synchronize()
g, s = torch.cuda.CUDAGraph(), torch.cuda.Stream()
go, res = threading.Event(), {}


def other():
    go.wait()
    try:
        if action == "sync":
            torch.cuda.synchronize()
        elif action == "memgetinfo":            # what gpushare's gpu_watcher thread does (cudaMemGetInfo)
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            rc = rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            if rc:
                raise RuntimeError(f"cudaMemGetInfo rc={rc}")
        elif action == "malloc":                # a raw allocation, as gpushare's memory hooks see them
            p = ctypes.c_void_p()
            rc = rt.cudaMalloc(ctypes.byref(p), 64 << 20)
            if rc:
                raise RuntimeError(f"cudaMalloc rc={rc}")
            rt.cudaFree(p)
        elif action == "eventquery":
            ev.query()
        res["other"] = "ok"
    except Exception as e:  # noqa: BLE001
        res["other"] = f"{type(e).__name__}: {str(e).splitlines()[0][:60]}"


t = threading.Thread(target=other)
t.start()
try:
    with torch.cuda.stream(s):
        g.capture_begin(capture_error_mode=mode)
        go.set()
        time.sleep(0.3)   # the other thread acts now: capture has begun, no kernel launched yet
        y = x @ x
        g.capture_end()
    g.replay()
    torch.cuda.synchronize()
    res["capture"] = "ok" if torch.allclose(y, ref, rtol=1e-2, atol=1e-2) else "replay WRONG"
except Exception as e:  # noqa: BLE001
    res["capture"] = f"BROKEN ({type(e).__name__})"
t.join()
print(f"RESULT {mode:12s} {action:10s} capture={res.get('capture')} other_thread={res.get('other')}", flush=True)
