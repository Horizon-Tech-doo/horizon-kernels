"""Same routing recipes as N1's n1_mk3.py, own seed and T list, weights copied from data.pt. argv: data.pt out.pt seed T,T,..."""
import sys, torch
H, E, topk = 2048, 256, 8
D = torch.load(sys.argv[1]); seed = int(sys.argv[3]); TS = [int(x) for x in sys.argv[4].split(",")]
out = {"W13": D["W13"], "W2": D["W2"]}
g = torch.Generator().manual_seed(seed)
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
print("data", seed, len(TS), flush=True)
