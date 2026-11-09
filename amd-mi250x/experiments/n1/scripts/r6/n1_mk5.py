"""Real slices: consecutive tokens taken from the 8192 slice of a layer's data file, weights kept. The 8192 case stays."""
import sys, torch
src, dst = sys.argv[1:3]
D = torch.load(src); out = {"W13": D["W13"], "W2": D["W2"]}
b = D[(8192, "prefill")]
for Tn in (1024, 1280, 1536, 2048, 3072, 4096, 6144, 7168):
    out[(Tn, "prefill")] = {k: v[:Tn].clone() for k, v in b.items()}
out[(8192, "prefill")] = b
torch.save(out, dst); print("real slices", flush=True)
