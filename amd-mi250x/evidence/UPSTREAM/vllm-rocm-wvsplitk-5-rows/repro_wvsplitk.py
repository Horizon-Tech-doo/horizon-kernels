"""Reproduction: vLLM's ROCm skinny GEMM (wvSplitK) against torch F.linear (hipBLASLt) for 1 to 8 activation rows.
Needs only a vLLM ROCm build and one GPU:   python repro_wvsplitk.py [N K ...]     (weight shapes as pairs; default 17408 5120  5120 8704  124160 5120)
For each weight [N, K] (bf16) and each row count n: time of
  dispatch   vllm.model_executor.layers.utils.rocm_unquantized_gemm_impl(x, w, None)   (what every unquantized nn.Linear runs on ROCm)
  wvSplitK   vllm._custom_ops.wvSplitK(w, x, cu_count, None) called directly           (the dispatch uses it for n <= 5)
  F.linear   torch.nn.functional.linear(x, w)                                          (hipBLASLt)
200 calls between two torch.cuda.synchronize(), median of 7 rounds, arm order rotated per round. Also prints the relative error of each result against an fp32 product."""
import statistics, sys, time
import torch, torch.nn.functional as F
import vllm, vllm._custom_ops as ops
from vllm.model_executor.layers import utils as U
a = [int(x) for x in sys.argv[1:]]
SHAPES = list(zip(a[0::2], a[1::2])) or [(17408, 5120), (5120, 8704), (124160, 5120)]
cu = torch.cuda.get_device_properties(0).multi_processor_count
print(f"vllm {vllm.__version__}  torch {torch.__version__}  hip {torch.version.hip}  device '{torch.cuda.get_device_name(0)}'  compute units {cu}")
def tm(f, it=200):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
print("N x K          rows  dispatch ms  wvSplitK ms  F.linear ms  dispatch/F.linear  err dispatch  err F.linear")
for N, K in SHAPES:
    w = (torch.randn(N, K, device="cuda") * 0.02).bfloat16()
    for n in (1, 2, 3, 4, 5, 6, 8):
        x = torch.randn(n, K, device="cuda").bfloat16(); ref = x.float() @ w.float().T
        arms = {"dispatch": lambda: U.rocm_unquantized_gemm_impl(x, w, None), "linear": lambda: F.linear(x, w)}
        try:
            ops.wvSplitK(w, x, cu, None); arms["wv"] = lambda: ops.wvSplitK(w, x, cu, None)
        except Exception:
            pass
        err = {k: float((f().reshape(n, N).float() - ref).norm() / ref.norm()) for k, f in arms.items()}
        for f in arms.values(): tm(f, 5)
        ts = {k: [] for k in arms}; names = list(arms)
        for rd in range(7):
            for i in range(len(names)):
                k = names[(i + rd) % len(names)]; ts[k].append(tm(arms[k]))
        m = {k: statistics.median(v) for k, v in ts.items()}
        print(f"{N} x {K:<6} {n:>4}  {m['dispatch']:>11.4f}  {(('%.4f' % m['wv']) if 'wv' in m else 'refused'):>11}  {m['linear']:>11.4f}  {m['dispatch'] / m['linear']:>17.2f}  {err['dispatch']:.2e}      {err['linear']:.2e}", flush=True)
