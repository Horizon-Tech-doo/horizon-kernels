"""Scan data: every T in the list, same routing recipes as n1_mk.py, own seed, weights copied from data.pt. Output data_scan.pt."""
import sys, torch
H, E, topk = 2048, 256, 8
D = torch.load(sys.argv[1]); TS = [int(x) for x in sys.argv[3].split(",")]
out = {"W13": D["W13"], "W2": D["W2"]}
g = torch.Generator().manual_seed(7)
rank = torch.randperm(E, generator=g).float()
for Tn in TS:
    X = torch.randn(Tn, H, generator=g).half()
    wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, generator=g), -1), topk, dim=-1)
    out[(Tn, "uniform")] = {"X": X, "ids": ids.int(), "wts": (wts / wts.sum(-1, keepdim=True)).float()}
    for name, s in (("zipf1", 1.0), ("zipf2", 2.0)):
        gum = -torch.log(-torch.log(torch.rand(Tn, E, generator=g).clamp_min(1e-20)))
        v, ids = torch.topk(-s * torch.log(rank + 1) + gum, topk, dim=-1)
        out[(Tn, name)] = {"X": X, "ids": ids.int(), "wts": torch.softmax(v, -1).float()}
torch.save(out, sys.argv[2])
print("scan", len(TS), flush=True)
