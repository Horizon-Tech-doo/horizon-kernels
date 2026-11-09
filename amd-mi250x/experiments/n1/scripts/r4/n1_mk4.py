"""More real-routing T: consecutive tokens taken from the 8192 slice of each layer's data file; the 2 layers keep their weights."""
import sys, torch
src, dst = sys.argv[1:3]
D = torch.load(src); out = {"W13": D["W13"], "W2": D["W2"]}
b = D[(8192, "prefill")]
for Tn in (8, 32, 128, 512, 513, 2048, 3072, 3584, 6144, 7168):
    out[(Tn, "prefill")] = {k: v[:Tn].clone() for k, v in b.items()}
torch.save(out, dst); print("real extra", flush=True)
