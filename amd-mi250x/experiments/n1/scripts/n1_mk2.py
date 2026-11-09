"""Adds T = 512 and T = 2048 to data.pt with their own seed, so existing T keep their bytes. Same routing recipes as n1_mk.py."""
import sys, torch
H, E, topk = 2048, 256, 8
D = torch.load(sys.argv[1])
g = torch.Generator().manual_seed(1)
rank = torch.randperm(E, generator=g).float()
for Tn in (512, 2048):
    X = torch.randn(Tn, H, generator=g).half()
    wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, generator=g), -1), topk, dim=-1)
    D[(Tn, "uniform")] = {"X": X, "ids": ids.int(), "wts": (wts / wts.sum(-1, keepdim=True)).float()}
    for name, s in (("zipf1", 1.0), ("zipf2", 2.0)):
        gum = -torch.log(-torch.log(torch.rand(Tn, E, generator=g).clamp_min(1e-20)))
        v, ids = torch.topk(-s * torch.log(rank + 1) + gum, topk, dim=-1)
        D[(Tn, name)] = {"X": X, "ids": ids.int(), "wts": torch.softmax(v, -1).float()}
torch.save(D, sys.argv[1])
print("added", flush=True)
