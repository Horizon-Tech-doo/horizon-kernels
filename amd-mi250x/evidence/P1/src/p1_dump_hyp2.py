"""P1 follow-up, one GCD: is the prefill kernel's error on real activations the fp16 cast of the attention weights? python p1_dump_hyp.py <out.jsonl> <dump dir>
For each dumped verify step: float64 attention of the query rows, and the same computation with one change at a time, all else float64:
  e16   the unnormalised weights exp(s - rowmax) rounded to fp16 before the P x V product, the row sum taken from the unrounded weights (what a flash kernel with an fp16 P cast does)
  e16n  as e16 with the row sum taken from the rounded weights
  e16z  as e16, and weights below fp16's smallest normal number (6.1e-5) set to zero: what happens if the fp16 path drops subnormal values (a flush in the cast or in the fp16 MFMA)
  ebf   the weights rounded to bf16 instead (8 mantissa bits, but no underflow near 1e-5)
  o16   only the final output rounded to fp16 and then bf16 (the plugin's output path)
Relative Frobenius error of each against the exact float64 result, next to the measured error of our kernel's output and of vLLM's kernel's output; and the share of each row's weight mass that lies
below fp16's smallest normal number (6.1e-5 of the row maximum): mean and largest over rows."""
import glob, json, sys
import torch
out = open(sys.argv[1], "a"); G, D = 6, 256
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
for f in sorted(glob.glob(sys.argv[2] + "/hlp-dump-*.pt")):
    t = torch.load(f); q, k, v = t["q"].cuda(), t["k"].cuda(), t["v"].cuda(); ours, stock = t["ours"].cuda().double(), t["stock"].cuda().double(); ctx = int(t["ctx"])
    n, H, _ = q.shape; Hkv = k.shape[1]; Lk = k.shape[0]
    num = {a: 0.0 for a in ("ours", "stock", "e16", "e16n", "e16z", "ebf", "o16")}; den = 0.0; tail_sum = 0.0; tail_max = 0.0; rows = 0
    for r0 in range(0, n, 192):
        r1 = min(n, r0 + 192); ref = torch.empty(r1 - r0, H, D, dtype=torch.float64, device="cuda"); alt = {a: torch.empty_like(ref) for a in ("e16", "e16n", "e16z", "ebf")}
        for hk in range(Hkv):
            s = torch.einsum("rgd,nd->rgn", q[r0:r1, hk * G:(hk + 1) * G].double(), k[:, hk].double()) / D ** 0.5
            mask = torch.arange(Lk, device="cuda")[None, :] > (ctx + torch.arange(r0, r1, device="cuda"))[:, None]
            s.masked_fill_(mask[:, None, :], float("-inf"))
            e = torch.exp(s - s.max(-1, keepdim=True).values); l = e.sum(-1, keepdim=True); vv = v[:, hk].double(); sl = slice(hk * G, (hk + 1) * G)
            ref[:, sl] = torch.einsum("rgn,nd->rgd", e, vv) / l
            e16 = e.to(torch.float16).double(); alt["e16"][:, sl] = torch.einsum("rgn,nd->rgd", e16, vv) / l; alt["e16n"][:, sl] = torch.einsum("rgn,nd->rgd", e16, vv) / e16.sum(-1, keepdim=True)
            alt["e16z"][:, sl] = torch.einsum("rgn,nd->rgd", e16 * (e16 >= 6.103515625e-05), vv) / l
            alt["ebf"][:, sl] = torch.einsum("rgn,nd->rgd", e.to(torch.bfloat16).double(), vv) / l
            tm = (e * (e < 6.1e-5)).sum(-1) / l[..., 0]; tail_sum += float(tm.sum()); tail_max = max(tail_max, float(tm.max())); rows += tm.numel()
            del s, e, e16
        den += float((ref * ref).sum())
        for a, o in (("ours", ours[r0:r1]), ("stock", stock[r0:r1]), ("e16", alt["e16"]), ("e16n", alt["e16n"]), ("e16z", alt["e16z"]), ("ebf", alt["ebf"]), ("o16", ref.to(torch.float16).to(torch.bfloat16).double())):
            num[a] += float(((o - ref) ** 2).sum())
        del ref, alt
    P(kind="hyp", file=f.split("/")[-1], n=n, ctx=ctx, tail_mass_mean=tail_sum / rows, tail_mass_max=tail_max, **{"rel_" + a: (x / den) ** 0.5 for a, x in num.items()})
P(kind="done")
