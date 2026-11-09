# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Cross-build kernel race (27-09-2026): ONE timing round of every live horizon_kernels TileLang kernel with whichever
# TileLang build is on PYTHONPATH. run_tlnext_race.sh alternates processes (cur, next, cur, next, cur, next) = 3
# interleaved rounds per build; xbuild_race_summary.py takes the fastest round per cell and build.
# Same fair method as k2_race_xl.py / k3rf_race.py: CUDA-graph replay (thread_local capture, retried), >= 512 MB of
# rotated weight copies per shape (defeats the 128 MB L2), median of 5 replays, a tiny separator kernel between calls
# (measured and subtracted), timings only while vLLM is settled and idle, must run under $HK_WORK/gpu_lock.sh.
#   xbuild_race.py <build-tag> <round> <out.jsonl>
import json, os, re, statistics, subprocess as _sp, sys, time, urllib.request
import torch
import tilelang, tilelang.language as T
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rfm
from tilelang.contrib.sm120_nvfp4 import prefill_rf as prf
from tilelang.contrib.sm120_nvfp4.kernels import nvfp4_gemm_kernel
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.horizon_fused import skinny_bf16 as sk, add_rmsnorm as an

BUILD, ROUND, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
ONLY = set(os.environ.get("RACE_ONLY", "xl,k3,rf,k5,k1a").split(","))
LOCK = "$HK_WORK/GPU_BUSY"
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
XL_MS = (1, 4, 8, 16, 32, 33, 48, 64, 96, 128, 192, 256, 512, 1024)
K3_MS = (512, 1024, 2048, 4096, 8192)


def vllm_busy():
    try:
        m = urllib.request.urlopen("http://localhost:8000/metrics", timeout=3).read().decode()
    except Exception:  # noqa: BLE001
        return 0.0
    return sum(float(v) for v in re.findall(r'vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.e+]+)', m))


def _vllm_sig():
    # (site-specific: return your vLLM service state here as (main pid, active state, sub state, restart count);
    # this stub reports "no service", so the waits below only check that the GPU is free)
    return ("0", "inactive", "dead", "0")


def _vllm_health():
    try:
        return urllib.request.urlopen("http://localhost:8000/health", timeout=3).status == 200
    except Exception:  # noqa: BLE001
        return False


_STABLE = {"sig": None, "since": 0.0}


def wait_vllm_stable(min_up=60.0, min_down=10.0, max_wait=300.0):
    n, t0 = 0, time.time()
    while True:
        sig = _vllm_sig(); now = time.time()
        if sig != _STABLE["sig"]:
            _STABLE["sig"], _STABLE["since"] = sig, now
        age = now - _STABLE["since"]
        up = sig[1] == "active" and sig[2] == "running"
        down = sig[1] in ("inactive", "failed") and sig[0] in ("0", None)
        if (up and age >= min_up and _vllm_health() and vllm_busy() == 0) or (down and age >= min_down):
            return sig
        if now - t0 > max_wait:
            print(json.dumps({"vllm_not_settled_proceeding": list(sig)}), flush=True)
            return sig
        if n % 24 == 0:
            print(json.dumps({"waiting_for_vllm_stable": time.strftime('%d-%m-%Y %H:%M:%S'), "vllm": sig}), flush=True)
        n += 1
        time.sleep(5)


def vllm_unchanged(sig):
    return _vllm_sig() == sig and (sig[1] != "active" or (_vllm_health() and vllm_busy() == 0))


def guarded(fn, tries=4):
    last = None
    for _ in range(tries):
        sig = wait_vllm_stable()
        try:
            v = fn()
        except Exception as ex:  # noqa: BLE001
            last = ex
            try:
                torch.cuda.synchronize()
            except Exception:  # noqa: BLE001
                pass
            print(json.dumps({"retry_after_error": f"{type(ex).__name__}: {str(ex)[:120]}"}), flush=True)
            time.sleep(5)
            continue
        if vllm_unchanged(sig):
            return v
        print(json.dumps({"retry_vllm_changed": list(sig), "now": list(_vllm_sig())}), flush=True)
    raise RuntimeError(f"timing failed {tries}x (last: {last})")


def acquire():
    try:
        holder = open(LOCK).read()
    except FileNotFoundError:
        holder = ""
    anc, pid = [], os.getppid()
    while pid > 1 and len(anc) < 16:
        anc.append(pid)
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    if not any(f"(pid {a}," in holder for a in anc):
        sys.exit(f"refusing to time: not running under $HK_WORK/gpu_lock.sh (GPU_BUSY: {holder.strip()!r})")


SEP = torch.zeros(1, device="cuda")


def graph_time_us(fns, iters=5, repeats=5, sep=True):
    if sep:
        fns = [lambda f=f: (f(), SEP.add_(1.0)) for f in fns]
    for fn in fns: fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, capture_error_mode="thread_local"):
        for _ in range(iters):
            for fn in fns: fn()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(repeats):
        s.record(); g.replay(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3 / (iters * len(fns)))
    del g
    return statistics.median(ts)


out = open(OUT, "a")


def emit(kernel, shape, M, us, **kw):
    row = {"build": BUILD, "round": ROUND, "kernel": kernel, "shape": shape, "M": M, "us": round(us, 3), **kw}
    out.write(json.dumps(row) + "\n"); out.flush()


acquire()
SEP_US = guarded(lambda: graph_time_us([lambda: None] * 8, sep=True))
print(json.dumps({"build": BUILD, "round": ROUND, "tilelang": tilelang.__file__, "separator_us": round(SEP_US, 2),
                  "start": time.strftime('%d-%m-%Y %H:%M')}), flush=True)
t0 = time.time()
for sname, (N, K) in SHAPES.items():
    copies = max(2, int(512e6 // (N * K // 2)) + 1)
    if "xl" in ONLY:                                   # regfrag_xl (all W4A16 decode)
        layers = [synthetic_layer(N, K, seed=s) for s in range(copies)]
        preps = [xl.prepare(*l) for l in layers]
        del layers
        for M in XL_MS:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            emit("xl", sname, M, guarded(lambda: graph_time_us([lambda p=p: xl.linear(x, p) for p in preps])) - SEP_US)
        del preps; torch.cuda.empty_cache()
    if ONLY & {"k3", "rf"}:                            # K3 two-pass FP4 prefill and prefill_rf (one weight copy)
        old, new = [], []
        for s in range(copies):
            g = torch.Generator(device="cuda").manual_seed(s)
            wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
            ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
            gs = torch.tensor([448.0 * 6], device="cuda")
            w0 = nv.prepare_weight(wp, ws, gs); old.append(w0)
            if "rf" in ONLY:
                prep = rfm.prepare(wp.view(torch.int8), ws, w0.alpha)
                new.append(prf.prepare_weight_rf(prep.Wg, ws, gs)); del prep
        for M in K3_MS:
            x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            A, SFA, rsinv = nv.quantize(x, 2)
            SFA3 = SFA.reshape(2, A.shape[0] * (K // 256), 4)
            C = torch.empty(A.shape[0], N, dtype=torch.bfloat16, device="cuda")
            if "k3" in ONLY:
                emit("k3_gemm", sname, M, guarded(lambda: graph_time_us(
                    [lambda w=w: nvfp4_gemm_kernel(A, w.packed, SFA3, w.sfb, w.alpha, rsinv, passes=2, out_dtype=T.bfloat16) for w in old])) - SEP_US)
                emit("k3_full", sname, M, guarded(lambda: graph_time_us([lambda w=w: nv.linear(x, w, passes=2) for w in old])) - SEP_US)
            if "rf" in ONLY:
                kern = prf.get_kernel(N, K, 2)
                emit("rf_gemm", sname, M, guarded(lambda: graph_time_us(
                    [lambda w=w: kern(A, w.wg, SFA3, w.sfb, w.alpha, rsinv, C) for w in new])) - SEP_US)
                emit("rf_full", sname, M, guarded(lambda: graph_time_us([lambda w=w: prf.linear_rf(x, w, passes=2) for w in new])) - SEP_US)
            del A, SFA, SFA3, rsinv, C, x
        del old, new; torch.cuda.empty_cache()
    print(json.dumps({"shape": sname, "secs": round(time.time() - t0)}), flush=True)
if "k5" in ONLY:                                       # K5: skinny in_proj_ba GEMM and add + RMSNorm
    Ws = [(torch.randn(96, 5120, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(256)]
    for M in (1, 4, 8, 16, 32, 48):
        x = torch.randn(M, 5120, device="cuda").to(torch.bfloat16)
        y1 = torch.empty(M, 96, device="cuda", dtype=torch.bfloat16)
        emit("k5_skinny", "in_proj_ba", M, guarded(lambda: graph_time_us([lambda w=w: sk.linear(x, w, out=y1) for w in Ws])) - SEP_US)
    del Ws
    w = (torch.randn(5120, device="cuda") * 0.1).to(torch.bfloat16)
    for M in (4, 32, 128):
        xs = [torch.randn(M, 5120, device="cuda").to(torch.bfloat16) for _ in range(64)]
        rs = [torch.randn(M, 5120, device="cuda").to(torch.bfloat16) for _ in range(64)]
        emit("k5_norm", "h5120", M, guarded(lambda: graph_time_us([lambda a=a, b=b: an.add_rmsnorm(a, b, w, 1e-6) for a, b in zip(xs, rs)])) - SEP_US)
    del xs, rs
if "k1a" in ONLY:                                      # K1a fused FP8 draft head, 48k vocabulary x 5120
    sys.path.insert(0, "$HK_DATA/horizon-pkgs")
    from horizon_kernels.draft_head import DraftHeadFP8
    heads = [DraftHeadFP8((torch.randn(49152, 5120, device="cuda") * 0.02).to(torch.bfloat16), act="fp8") for _ in range(3)]
    for M in (1, 4, 8):
        h = torch.randn(M, 5120, device="cuda").to(torch.bfloat16)
        emit("k1a", "draft48k", M, guarded(lambda: graph_time_us([lambda hd=hd: hd(h) for hd in heads])) - SEP_US)
    del heads
print(json.dumps({"ROUND_DONE": BUILD, "round": ROUND, "secs": round(time.time() - t0)}), flush=True)
