"""P1 follow-up, one GCD: which kernel is closer to the truth on the server's own activations. python p1_dump_ref.py <out.jsonl> <dump dir>
Reads the tensors horizon_lumi_prefill saved in a verify step (HLP_DUMP): the step's queries q [n, H, D], all keys and values k, v [ctx + n, Hkv, D] (ctx = cached tokens before the chunk, 0 for a
fresh step), our output and vLLM's kernel's output for the n query rows. Computes the float64 causal GQA attention of those rows in chunks and reports each output's relative error, its worst row,
and the two outputs' difference."""
import glob, json, sys
import torch
out = open(sys.argv[1], "a"); G, D = 6, 256
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
for f in sorted(glob.glob(sys.argv[2] + "/hlp-dump-*.pt")):
    t = torch.load(f); q, k, v = t["q"].cuda(), t["k"].cuda(), t["v"].cuda(); ours, stock = t["ours"].cuda().double(), t["stock"].cuda().double(); ctx = int(t["ctx"])
    n, H, _ = q.shape; Hkv = k.shape[1]; Lk = k.shape[0]
    num = {"ours": 0.0, "stock": 0.0}; worst = {"ours": 0.0, "stock": 0.0}; den = 0.0
    for r0 in range(0, n, 256):
        r1 = min(n, r0 + 256); ref = torch.empty(r1 - r0, H, D, dtype=torch.float64, device="cuda")
        for hk in range(Hkv):
            s = torch.einsum("rgd,nd->rgn", q[r0:r1, hk * G:(hk + 1) * G].double(), k[:, hk].double()) / D ** 0.5
            mask = torch.arange(Lk, device="cuda")[None, :] > (ctx + torch.arange(r0, r1, device="cuda"))[:, None]
            s.masked_fill_(mask[:, None, :], float("-inf"))
            ref[:, hk * G:(hk + 1) * G] = torch.einsum("rgn,nd->rgd", torch.softmax(s, -1), v[:, hk].double()); del s
        den += float((ref * ref).sum())
        for nm, o in (("ours", ours), ("stock", stock)):
            d = o[r0:r1] - ref; num[nm] += float((d * d).sum()); worst[nm] = max(worst[nm], float((d.flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp_min(1e-300)).max()))
        del ref
    P(kind="dump", file=f.split("/")[-1], n=n, ctx=ctx, Lk=Lk, q_absmax=float(q.float().abs().max()), k_absmax=float(k.float().abs().max()), out_absmax=float(stock.abs().max()),
      rel_ours=(num["ours"] / den) ** 0.5, rel_stock=(num["stock"] / den) ** 0.5, worst_row_ours=worst["ours"], worst_row_stock=worst["stock"], ours_vs_stock=float((ours - stock).norm() / stock.norm()))
P(kind="done")
