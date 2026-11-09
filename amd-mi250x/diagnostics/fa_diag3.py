"""FA diagnosis part 3: who is wrong at long seq, the TileLang kernel or the example's fp16 torch reference?
Compare both to a float64 reference. Also toggle rasterization (T.use_swizzle) and num_split_q."""
import json, os, re, runpy, sys, torch
import torch.nn.functional as F
src = sys.argv[1]
code = open(os.path.join(src, "examples/amd/example_amd_flash_attn_fwd.py")).read()
code, n = re.subn(r"^@tilelang\.autotune\(.*\)\n", "", code, flags=re.M); assert n == 1
dst = os.path.abspath("fa_fwd_noautotune.py"); open(dst, "w").write(code)
mod = runpy.run_path(dst, run_name="fa_example")

def ref64(Q, K, V):
    q, k, v = (x.double() for x in (Q, K, V))
    s = torch.einsum("bqhd,bkhd->bhqk", q, k) / q.size(-1) ** 0.5
    return torch.einsum("bhqk,bkhd->bqhd", F.softmax(s, -1), v)

def rel(a, b):
    return round(((a.double() - b).norm() / b.norm()).item(), 6)

torch.manual_seed(0)
for seq in [1024, 4096]:
    Q, K, V = (torch.randn(1, seq, 8, 128, device="cuda", dtype=torch.float16) for _ in range(3))
    R64 = ref64(Q, K, V)
    R16 = mod["ref_program"](Q, K, V, False, 1)
    out = {"seq": seq, "torch_fp16_ref_vs_fp64": rel(R16, R64),
           "sdpa_fp16_vs_fp64": rel(F.scaled_dot_product_attention(Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)).transpose(1, 2), R64)}
    for ras in (True, False):
        for nsq in (128, 16):
            cfg = dict(block_M=64, block_N=64, num_split_q=nsq, threads=256, num_stages=0, enable_rasterization=ras, k_pack=2,
                       panel_size=7, qk_coalesced_width=8, v_coalesced_width=4)
            try:
                O = mod["fast_flashattn"](1, 8, seq, 128, False, 1, **cfg)(Q, K, V)
                out[f"tilelang_vs_fp64 ras={int(ras)} split={nsq}"] = rel(O, R64)
            except Exception as e:  # noqa
                out[f"tilelang ras={int(ras)} split={nsq}"] = "ERR " + str(e)[-120:]
    print(json.dumps(out), flush=True)
