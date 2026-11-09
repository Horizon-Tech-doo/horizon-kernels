"""FA diagnosis part 2: run the upstream AMD example kernel (no autotune) at growing seq_len, so the number of KV
blocks (online-softmax rescale steps) and Q blocks grows. Characterise the error: relative norm, best scalar fit."""
import json, os, re, runpy, sys, torch
from functools import partial
src = sys.argv[1]
code = open(os.path.join(src, "examples/amd/example_amd_flash_attn_fwd.py")).read()
code, n = re.subn(r"^@tilelang\.autotune\(.*\)\n", "", code, flags=re.M); assert n == 1
dst = os.path.abspath("fa_fwd_noautotune.py"); open(dst, "w").write(code)
mod = runpy.run_path(dst, run_name="fa_example")
base = dict(num_split_q=128, threads=256, num_stages=0, enable_rasterization=True, k_pack=2, panel_size=7,
            qk_coalesced_width=8, v_coalesced_width=4)
torch.manual_seed(0)
for bm, bn in [(64, 64), (64, 32)]:
    for seq in [bn, 2 * bn, 4 * bn, 256, 1024, 4096]:
        for heads in [1, 8]:
            r = {"bM": bm, "bN": bn, "seq": seq, "heads": heads}
            try:
                k = mod["fast_flashattn"](1, heads, seq, 128, False, 1, block_M=bm, block_N=bn, **base)
                Q, K, V = (torch.randn(1, seq, heads, 128, device="cuda", dtype=torch.float16) for _ in range(3))
                O = k(Q, K, V).float(); R = mod["ref_program"](Q, K, V, False, 1).float()
                r["rel"] = round(((O - R).norm() / R.norm()).item(), 5)
                c = ((O * R).sum() / (O * O).sum()).item()
                r["scale_fit"] = round(c, 4)
                r["rel_after_scale"] = round(((O * c - R).norm() / R.norm()).item(), 5)
                rows = ((O - R).norm(dim=-1) / R.norm(dim=-1)).reshape(-1)
                r["bad_rows_frac"] = round((rows > 0.01).float().mean().item(), 4)
            except Exception as e:  # noqa
                r["err"] = (type(e).__name__ + ": " + str(e))[-200:]
            print(json.dumps(r), flush=True)
