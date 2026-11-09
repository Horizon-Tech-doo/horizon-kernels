# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Native-order FP4 prefill (HORIZON_NATIVE_PREFILL, 27-09-2026): standalone checks before the in-vLLM A/B.
#   gpu_lock.sh natpf python natpf_check.py OUT.jsonl [exact|graph|time|all]
# exact: prefill_rs.linear_native (rsN3) vs today's K3 (sm120_nvfp4.linear) on REAL checkpoint layers, all 5 fused
#        shapes (collapsed global scale, as the plugin does with HORIZON_EXACT_SHARD_SCALES=0), M 128 ... 8192 plus
#        odd sizes, passes 2 (and 1 for the lossy variant): bitwise equality of the bf16 outputs; plus the bytes of each
#        handle (the native copy must equal K3's).
# graph: CUDA-graph capture (capture_error_mode="thread_local", with retry) of linear_native at several M, replay with
#        new inputs copied into the static buffer, bitwise vs eager.
# time:  the full op (quantizer + GEMM) K3 vs native, CUDA-graph replay over > 512 MB of rotated weight copies,
#        3 interleaved rounds, fastest per arm, layer-weighted (48 qkvz, 16 qkv, 64 o, 64 gate_up, 64 down).
import json, sys, time
import torch
from safetensors import safe_open
from tilelang.contrib import sm120_nvfp4 as nv
from tilelang.contrib.sm120_nvfp4 import prefill_rs as prs

OUT = sys.argv[1]
WHAT = sys.argv[2] if len(sys.argv) > 2 else "all"
CK = "$HK_DATA/models/Qwen3.8-27B-NVFP4/model.safetensors"
P = "model.language_model.layers.%d.%s."
# fused shapes as vLLM builds them: (layer, [shard names]) -> N x K
REAL = {
    "qkvz": [(0, ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"]), (30, ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"])],
    "qkv": [(3, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]), (43, ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"])],
    "o": [(3, ["self_attn.o_proj"]), (0, ["linear_attn.out_proj"])],
    "gate_up": [(3, ["mlp.gate_proj", "mlp.up_proj"]), (60, ["mlp.gate_proj", "mlp.up_proj"])],
    "down": [(3, ["mlp.down_proj"]), (60, ["mlp.down_proj"])],
}
SHAPES = {"qkvz": (16384, 5120), "qkv": (14336, 5120), "o": (5120, 6144), "gate_up": (34816, 5120), "down": (5120, 17408)}
WEIGHT = {"qkvz": 48, "qkv": 16, "o": 64, "gate_up": 64, "down": 64}
fo = open(OUT, "a")


def emit(d):
    d["t"] = time.strftime("%d-%m-%Y %H:%M:%S")
    print(json.dumps(d), flush=True)
    fo.write(json.dumps(d) + "\n"); fo.flush()


def load_real(f, li, names):
    wps, wss, gss, sizes = [], [], [], []
    for nm in names:
        pre = P % (li, nm)
        wps.append(f.get_tensor(pre + "weight_packed")); wss.append(f.get_tensor(pre + "weight_scale"))
        gss.append(f.get_tensor(pre + "weight_global_scale").float().reshape(-1)); sizes.append(wps[-1].shape[0])
    gs = torch.cat(gss)
    gs = gs.max().expand(len(sizes)).clone()          # vLLM's max() collapse (plugin, HORIZON_EXACT_SHARD_SCALES=0)
    return torch.cat(wps).contiguous(), torch.cat(wss).contiguous(), gs, sizes


def act(m, k, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", generator=g)
    x[:, torch.randint(0, k, (16,), device="cuda", generator=g)] *= 30      # LLM-like outlier channels
    x[m // 3] = 0                                                           # an all-zero row (rsinv path)
    return x.to(torch.bfloat16)


def nbytes(*ts):
    return sum(t.numel() * t.element_size() for t in ts)


if WHAT in ("exact", "all"):
    MS = [128, 200, 256, 512, 1000, 1024, 2048, 3000, 4096, 8192]
    with safe_open(CK, "pt", device="cuda") as f:
        for sname, layers in REAL.items():
            for li, names in layers:
                wp, ws, gs, sizes = load_real(f, li, names)
                N, K = wp.shape[0], wp.shape[1] * 2
                assert (N, K) == SHAPES[sname], (sname, N, K)
                w3 = nv.prepare_weight(wp, ws, gs, shard_sizes=sizes)
                wn = prs.prepare_weight_native(wp, ws, gs, shard_sizes=sizes)
                alpha_eq = bool(torch.equal(w3.alpha, wn.alpha))
                # K3's handle = the checkpoint weight (a view) + swizzled scales; native = own copy + scale tiles
                emit({"check": "bytes", "shape": sname, "layer": li, "k3_bytes": nbytes(w3.packed, w3.sfb),
                      "native_bytes": nbytes(wn.w, wn.sw), "alpha_equal": alpha_eq,
                      "k3_packed_is_checkpoint": w3.packed.data_ptr() == wp.data_ptr()})
                for passes in (2, 1):
                    for M in (MS if passes == 2 else [512, 2048]):
                        x = act(M, K, 1000 * li + M)
                        y3 = nv.linear(x, w3, passes=passes)
                        yn = prs.linear_native(x, wn, passes=passes)
                        torch.cuda.synchronize()
                        same = bool(torch.equal(y3.view(torch.int16), yn.view(torch.int16)))
                        d = {"check": "exact", "shape": sname, "layer": li, "passes": passes, "M": M, "bit_identical": same}
                        if not same:
                            diff = y3.view(torch.int16) != yn.view(torch.int16)
                            d.update(ndiff=int(diff.sum()), max_abs=float((y3.float() - yn.float()).abs().max()))
                        emit(d)
                del w3, wn, wp, ws
                torch.cuda.empty_cache()

if WHAT in ("graph", "all"):
    for sname, (N, K) in SHAPES.items():
        g = torch.Generator(device="cuda").manual_seed(5)
        wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
        ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
        gs = torch.tensor([448.0 * 6], device="cuda")
        wn = prs.prepare_weight_native(wp, ws, gs)
        w3 = nv.prepare_weight(wp, ws, gs)
        for M in (512, 1024, 3000, 8192):
            xs = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
            prs.linear_native(xs, wn)                    # warm (JIT) outside capture
            torch.cuda.synchronize()
            ok, err = False, ""
            for attempt in range(3):
                try:
                    gr = torch.cuda.CUDAGraph()
                    st = torch.cuda.Stream()
                    with torch.cuda.stream(st):
                        with torch.cuda.graph(gr, stream=st, capture_error_mode="thread_local"):
                            ys = prs.linear_native(xs, wn)
                    torch.cuda.synchronize()
                    ok = True
                    break
                except Exception as ex:  # noqa: BLE001
                    err = f"{type(ex).__name__}: {str(ex)[:200]}"
                    torch.cuda.synchronize()
            res = []
            if ok:
                for seed in (1, 2, 3):
                    x = act(M, K, seed)
                    xs.copy_(x)
                    gr.replay(); torch.cuda.synchronize()
                    ye = prs.linear_native(x, wn)
                    y3 = nv.linear(x, w3, passes=2)
                    res.append(bool(torch.equal(ys.view(torch.int16), ye.view(torch.int16)))
                               and bool(torch.equal(ye.view(torch.int16), y3.view(torch.int16))))
            emit({"check": "graph", "shape": sname, "M": M, "captured": ok, "attempts": attempt + 1,
                  "replay_bit_identical_vs_eager_and_k3": res, "error": err})
            del gr
        del wn, w3, wp, ws
        torch.cuda.empty_cache()

if WHAT in ("time", "all"):
    MS = [512, 1024, 2048, 4096, 8192]
    sep = torch.empty(1 << 20, device="cuda")
    res = {}
    for sname, (N, K) in SHAPES.items():
        copies = max(2, int(512e6 // (N * K // 2)) + 1)
        W3, WN = [], []
        for c in range(copies):
            g = torch.Generator(device="cuda").manual_seed(100 + c)
            wp = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda", generator=g)
            ws = (torch.rand(N, K // 16, device="cuda", generator=g) * 2 + 0.5).to(torch.float8_e4m3fn)
            gs = torch.tensor([448.0 * 6], device="cuda")
            W3.append(nv.prepare_weight(wp, ws, gs)); WN.append(prs.prepare_weight_native(wp, ws, gs))
        for M in MS:
            x = act(M, K, 9)
            arms = {"k3": lambda w: nv.linear(x, w, passes=2), "native": lambda w: prs.linear_native(x, w, passes=2)}
            handles = {"k3": W3, "native": WN}
            graphs = {}
            for a, fn in arms.items():
                for w in handles[a]:
                    fn(w)
                torch.cuda.synchronize()
                for attempt in range(3):
                    try:
                        gr = torch.cuda.CUDAGraph(); st = torch.cuda.Stream()
                        with torch.cuda.stream(st):
                            with torch.cuda.graph(gr, stream=st, capture_error_mode="thread_local"):
                                for w in handles[a]:
                                    sep.mul_(1.0)      # separator: keeps consecutive GEMMs from overlapping tails
                                    fn(w)
                        break
                    except Exception:  # noqa: BLE001
                        torch.cuda.synchronize()
                torch.cuda.synchronize()
                graphs[a] = gr
            times = {a: [] for a in arms}
            for r in range(3):
                for a in (list(arms) if r % 2 == 0 else list(arms)[::-1]):
                    gr = graphs[a]
                    gr.replay(); torch.cuda.synchronize()
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(5):
                        gr.replay()
                    e.record(); torch.cuda.synchronize()
                    times[a].append(s.elapsed_time(e) * 1e3 / (5 * copies))
            best = {a: min(t) for a, t in times.items()}
            res[(sname, M)] = best
            emit({"check": "time", "shape": sname, "M": M, "copies": copies, "k3_us": round(best["k3"], 2),
                  "native_us": round(best["native"], 2), "ratio": round(best["native"] / best["k3"], 4),
                  "rounds": {a: [round(v, 2) for v in t] for a, t in times.items()}})
            del graphs
        del W3, WN
        torch.cuda.empty_cache()
    for M in MS:
        k3 = sum(WEIGHT[s] * res[(s, M)]["k3"] for s in SHAPES)
        na = sum(WEIGHT[s] * res[(s, M)]["native"] for s in SHAPES)
        emit({"check": "time_weighted", "M": M, "ratio": round(na / k3, 4)})
print("NATPF_CHECK_DONE", WHAT, flush=True)
