# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# #3257 race (27-09-2026): the current K3 (two weight copies), prefill_rf (one copy, shared-memory conversion) and
# prefill_rs (one copy, #3257 register operand; conv=1) plus its conv=0 ceiling arm, on the rebased build, same protocol
# as the prefill_rf NO-GO: 3 interleaved rounds in ONE process (K3, rf, rs1, rs0, K3, ...), fastest round per method.
# Same fair method as k2_race_xl.py / k3rf_race.py: CUDA-graph replay (thread_local capture, retried), >= 512 MB of
# rotated weight copies per shape (defeats the 128 MB L2), median of 5 replays, a tiny separator kernel between calls
# (measured and subtracted), timings only while vLLM is settled and idle, must run under $HK_WORK/gpu_lock.sh.
#   gpu_lock.sh rs_race python rs_race.py <out.jsonl>
import json, os, re, statistics, subprocess as _sp, sys, time, urllib.request
import torch
import tilelang, tilelang.language as T
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import decode_regfrag_xl as xl
from tilelang.contrib.sm120_nvfp4 import decode_regfrag as rfm
from tilelang.contrib.sm120_nvfp4 import prefill_rf as prf
from tilelang.contrib.sm120_nvfp4.kernels import nvfp4_gemm_kernel
from tilelang.contrib.sm120_nvfp4.decode_contract import synthetic_layer
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs

OUT = sys.argv[1]
ARMS = os.environ.get("RS_ARMS", "k3,rf,rs1,rs0").split(",")
ROUNDS = 3
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



acquire()
SEP_US = guarded(lambda: graph_time_us([lambda: None] * 8, sep=True))
print(json.dumps({"separator_us": round(SEP_US, 2), "arms": ARMS, "start": time.strftime('%d-%m-%Y %H:%M')}), flush=True)
out = open(OUT, "a")
if os.environ.get("RS_SHAPES"):
    SHAPES = {k: SHAPES[k] for k in os.environ["RS_SHAPES"].split(",")}
if os.environ.get("RS_MS"):
    K3_MS = tuple(int(v) for v in os.environ["RS_MS"].split(","))
for sname, (N, K) in SHAPES.items():
    copies = max(2, int(512e6 // (N * K // 2)) + 1)
    W = {a: [] for a in ARMS}
    for s in range(copies):
        g = torch.Generator(device="cuda").manual_seed(s)
        wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
        ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
        gs = torch.tensor([448.0 * 6], device="cuda")
        w0 = nv.prepare_weight(wp, ws, gs)
        if "k3" in W: W["k3"].append(w0)
        prep = rfm.prepare(wp.view(torch.int8), ws, w0.alpha)
        if "rf" in W: W["rf"].append(prf.prepare_weight_rf(prep.Wg, ws, gs))
        if "rs1" in W: W["rs1"].append(prs.prepare_weight_rs(prep.Wg, ws, gs))
        if "rs1i" in W: W["rs1i"].append(prs.prepare_weight_rs(prep.Wg, ws, gs))
        if "rs0" in W:
            w1 = prs.prepare_weight_rs(prep.Wg, ws, gs)
            W["rs0"].append(prs.NVFP4WeightRS(prs.fragment_order_weights(wp, ws), w1.sw, w1.alpha, N, K, 0))
        del prep
    for M in K3_MS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        A, SFA, rsinv = nv.quantize(x, 2)
        SFA3 = SFA.reshape(2, A.shape[0] * (K // 256), 4)
        Xr, SXr, rsr = prs.quantize_rs(x, 2)
        C = torch.empty(A.shape[0], N, dtype=torch.bfloat16, device="cuda")
        fns = {}
        if "k3" in W:
            fns["k3_gemm"] = [lambda w=w: nvfp4_gemm_kernel(A, w.packed, SFA3, w.sfb, w.alpha, rsinv, passes=2, out_dtype=T.bfloat16) for w in W["k3"]]
        if "rf" in W:
            kr = prf.get_kernel(N, K, 2)
            fns["rf_gemm"] = [lambda w=w: kr(A, w.wg, SFA3, w.sfb, w.alpha, rsinv, C) for w in W["rf"]]
        for a in ("rs1", "rs0", "rs1i"):
            if a in W:
                ks = prs.get_kernel(N, K, 2, conv=0 if a == "rs0" else 1, **({"ilv": 1} if a == "rs1i" else {}))
                fns[a + "_gemm"] = [lambda w=w, ks=ks: ks(Xr, w.w, SXr, w.sw, w.alpha, rsr, C) for w in W[a]]
        # identity of this exact setup (first copy) vs K3
        ident = {}
        if "k3" in W:
            y0 = nvfp4_gemm_kernel(A, W["k3"][0].packed, SFA3, W["k3"][0].sfb, W["k3"][0].alpha, rsinv, passes=2, out_dtype=T.bfloat16)
            for a in ("rs1", "rs0", "rs1i", "rf"):
                if a in W:
                    fns[a + "_gemm"][0](); torch.cuda.synchronize()
                    ident[a] = bool(torch.equal(y0.view(torch.int16), C.view(torch.int16)))
        if os.environ.get("RS_FULL", "1") == "1":
            if "k3" in W: fns["k3_full"] = [lambda w=w: nv.linear(x, w, passes=2) for w in W["k3"]]
            if "rf" in W: fns["rf_full"] = [lambda w=w: prf.linear_rf(x, w, passes=2) for w in W["rf"]]
            for a in ("rs1", "rs0", "rs1i"):
                opt = {"ilv": 1} if a == "rs1i" else {}
                if a in W: fns[a + "_full"] = [lambda w=w, opt=opt: prs.linear_rs(x, w, passes=2, **opt) for w in W[a]]
        rounds = {k: [] for k in fns}
        for _ in range(ROUNDS):
            for k, f in fns.items():
                rounds[k].append(guarded(lambda f=f: graph_time_us(f)) - SEP_US)
        row = {"shape": sname, "N": N, "K": K, "M": M, "bit_identical_vs_k3": ident}
        for k in fns:
            row[k + "_us"] = round(min(rounds[k]), 2)
            row[k + "_rounds"] = [round(v, 2) for v in rounds[k]]
        print(json.dumps(row), flush=True); out.write(json.dumps(row) + "\n"); out.flush()
        del A, SFA, SFA3, rsinv, C, x, Xr, SXr, rsr
    del W; torch.cuda.empty_cache()
print(json.dumps({"RS_RACE_DONE": time.strftime('%d-%m-%Y %H:%M')}), flush=True)
