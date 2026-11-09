# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# A2 gate: horizon_ops.qk_norm_rope vs vLLM's Triton _fused_qk_rmsnorm_rope_gate_kernel: bit-identity of q, k, gate
# on real q/k-norm weights (layer 3), random bf16 cos/sin cache, mRoPE positions; speed per call.
# usage: python test_qk.py OUT.jsonl [pkg]
import importlib, json, sys
import torch
from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate

OUT = sys.argv[1]
PKG = sys.argv[2] if len(sys.argv) > 2 else "horizon_ops"
qk = importlib.import_module(f"tilelang.contrib.{PKG}.qk_norm_rope")
dev = torch.device("cuda")
HQ, HK, D, ROT = 24, 4, 256, 64
MODEL = "$HK_DATA/models/Qwen3.8-27B-NVFP4"


def log(rec):
    print(json.dumps(rec), flush=True)
    open(OUT, "a").write(json.dumps(rec) + "\n")


def weights():
    from safetensors import safe_open
    idx = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
    p = "model.language_model.layers.3.self_attn."
    with safe_open(f"{MODEL}/{idx[p + 'q_norm.weight']}", "pt") as f:
        return f.get_tensor(p + "q_norm.weight").to(dev), f.get_tensor(p + "k_norm.weight").to(dev)


def ours(qg, k, qw, kw, cs, pos, eps):
    NT = qg.shape[0]
    qo = torch.empty(NT, HQ * D, dtype=torch.bfloat16, device=dev)
    ko = torch.empty(NT, HK * D, dtype=torch.bfloat16, device=dev)
    go = torch.empty_like(qo)
    kern = qk.qk_norm_rope_kernel(HQ, HK, D, ROT, qg.stride(0), k.stride(0), "bfloat16")
    kern(qg, k, qo, ko, go, qw, kw, cs, pos, eps)
    return qo, ko, go


def main():
    qw, kw = weights()
    gen = torch.Generator(device=dev); gen.manual_seed(0)
    cs = (torch.randn(300000, ROT, device=dev, generator=gen)).to(torch.bfloat16)
    tot = 0
    for NT in (1, 4, 32, 128, 1000, 8192):
        qkv = (torch.randn(NT, HQ * 2 * D + 2 * HK * D, device=dev, generator=gen) * 3).to(torch.bfloat16)
        qg, k, v = qkv.split([HQ * 2 * D, HK * D, HK * D], dim=-1)
        pos = torch.randint(0, 262144, (3, NT), device=dev, generator=gen, dtype=torch.int64)
        q1, k1, g1 = fused_qk_rmsnorm_rope_gate(qg, k, qw, kw, cs, pos, 1e-6, HQ, HK, D, ROT, norm_beta=1.0,
                                                 mrope_section=[11, 11, 10])
        q2, k2, g2 = ours(qg, k, qw, kw, cs, pos, 1e-6)
        torch.cuda.synchronize()
        d = [int((a.view(torch.int16) != b.view(torch.int16)).sum()) for a, b in ((q1, q2), (k1, k2), (g1, g2))]
        tot += sum(d)
        log({"test": "qk_bits", "NT": NT, "q_diff": d[0], "k_diff": d[1], "gate_diff": d[2],
             "q_max_abs": float((q1.float() - q2.float()).abs().max())})
        if NT in (4, 32, 128):
            ts = {}
            for name, fn in (("vllm", lambda: fused_qk_rmsnorm_rope_gate(qg, k, qw, kw, cs, pos, 1e-6, HQ, HK, D, ROT,
                                                                             norm_beta=1.0, mrope_section=[11, 11, 10])),
                             ("ours", lambda: ours(qg, k, qw, kw, cs, pos, 1e-6))):
                fn(); torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode="thread_local"):
                    for _ in range(20):
                        fn()
                best = 1e30
                for _ in range(5):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record(); g.replay(); e1.record(); e1.synchronize()
                    best = min(best, e0.elapsed_time(e1) * 1000 / 20)
                ts[name] = round(best, 2)
            log({"test": "qk_bench", "NT": NT, **ts})
    log({"summary": "qk_a2", "bit_diffs_total": tot})


main()
print("TEST_QK_DONE", flush=True)
# KV-cache write gate (horizon_ops.kv_write), run in the same slot
try:
    import runpy
    sys.argv = ["test_kvw.py", "$HK_WORK/ops/kvw.jsonl", PKG]
    runpy.run_path("$HK_WORK/ops/test_kvw.py", run_name="__main__")
except Exception as e:  # noqa: BLE001
    import traceback; traceback.print_exc()
    print(json.dumps({"test": "kvw_error", "error": str(e)[:500]}), flush=True)
