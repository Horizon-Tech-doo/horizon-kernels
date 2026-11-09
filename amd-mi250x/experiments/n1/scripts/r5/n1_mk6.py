"""Grid test T, real: consecutive tokens from the 8192 slice of a layer's data file. Output keeps weights and the new T."""
import sys, torch
src, dst = sys.argv[1:3]
D = torch.load(src); out = {"W13": D["W13"], "W2": D["W2"]}
b = D[(8192, "prefill")]
for Tn in (17, 33, 129, 513, 1025, 2049, 4097):
    out[(Tn, "prefill")] = {k: v[:Tn].clone() for k, v in b.items()}
torch.save(out, dst); print("real grid slices", flush=True)
