"""Same inputs for both stacks, generated on the CPU. Routing is the existing script's: softmax of random logits, top-8."""
import math, sys, torch
H, I, E, topk = 2048, 512, 256, 8
g = torch.Generator().manual_seed(0)
W13 = (torch.randn(E, 2 * I, H, generator=g) * (1 / math.sqrt(H))).half()
W2 = (torch.randn(E, H, I, generator=g) * (1 / math.sqrt(I))).half()
out = {"W13": W13, "W2": W2}
for Tn in (1024, 4096):
    X = torch.randn(Tn, H, generator=g).half()
    wts, ids = torch.topk(torch.softmax(torch.randn(Tn, E, generator=g), -1), topk, dim=-1)
    out[Tn] = {"X": X, "ids": ids.int(), "wts": (wts / wts.sum(-1, keepdim=True)).float()}
torch.save(out, sys.argv[1])
print("saved", {k: (v.shape if hasattr(v, "shape") else "") for k, v in out.items()})
