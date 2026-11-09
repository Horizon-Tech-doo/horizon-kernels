"""Same inputs for both stacks, generated on the CPU, top-8 of 256 experts.
uniform: softmax of random logits, as in E2. zipf1, zipf2: score = -s*log(rank+1) + Gumbel, so experts are drawn without
replacement with Zipf(s) popularity; weights are the softmax of the chosen scores. T=1024/4096 uniform match E2's recipe,
but not its bytes."""
import math, sys, torch
H, I, E, topk = 2048, 512, 256, 8
g = torch.Generator().manual_seed(0)
W13 = (torch.randn(E, 2 * I, H, generator=g) * (1 / math.sqrt(H))).half()
W2 = (torch.randn(E, H, I, generator=g) * (1 / math.sqrt(I))).half()
out = {"W13": W13, "W2": W2}
rank = torch.randperm(E, generator=g).float()
for Tn in (1, 16, 64, 256, 1024, 4096):
    X = torch.randn(Tn, H, generator=g).half()
    wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, generator=g), -1), topk, dim=-1)
    out[(Tn, "uniform")] = {"X": X, "ids": ids.int(), "wts": (wts / wts.sum(-1, keepdim=True)).float()}
    for name, s in (("zipf1", 1.0), ("zipf2", 2.0)):
        gum = -torch.log(-torch.log(torch.rand(Tn, E, generator=g).clamp_min(1e-20)))
        sc = -s * torch.log(rank + 1) + gum
        v, ids = torch.topk(sc, topk, dim=-1)
        wts = torch.softmax(v, -1).float()
        out[(Tn, name)] = {"X": X, "ids": ids.int(), "wts": wts}
torch.save(out, sys.argv[1])
print("saved")
