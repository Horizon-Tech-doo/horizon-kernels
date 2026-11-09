"""N4 lever D: 128-byte X and W2 shifts on landed FusedMoE(16) and on vLLM's fused_experts in the image, same process. Offsets: X relative to W13, W2 relative to its spare's 256-aligned start (inter's address is the kernel's own), inside one spare allocation per tensor, refilled before each arm.
argv: data tag rounds. Ratio convention in the analysis: ours / vLLM, time, lower is better, as in evidence/N1."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
data, tag, rounds = sys.argv[1], sys.argv[2], int(sys.argv[3])
import moe_fused as M
H, I, E, topk, Tn = 2048, 512, 256, 8, 16
D = torch.load(data)
W13, W20 = D["W13"].cuda(), D["W2"].cuda()
stack = os.environ.get("N5_STACK")
print(json.dumps(dict(info=1, tag=tag, torch=torch.__version__, stack=stack, moe_fused=M.__file__)), flush=True)


class Slot:
    def __init__(self, t):
        self.t, self.sz = t, t.element_size()
        self.sp = torch.empty(t.numel() + 512 // self.sz, device=t.device, dtype=t.dtype)

    def at(self, ref_ptr, off):
        st = ((ref_ptr - self.sp.data_ptr() - off) & 255) // self.sz
        v = self.sp[st:st + self.t.numel()].view(self.t.shape)
        v.copy_(self.t)
        assert (ref_ptr - v.data_ptr()) & 255 == off
        return v


def bench(fn, warm=20, rep=300):
    for _ in range(warm):
        fn()
    ps = []
    for _ in range(rep):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); ps.append((s, e))
    torch.cuda.synchronize()
    t = torch.tensor([s.elapsed_time(e) for s, e in ps], dtype=torch.float64)
    return round(t.median().item(), 4), round(t.mean().item(), 4)


fe = None
if stack == "image":
    from vllm.model_executor.layers.fused_moe import fused_moe as FM
    fe = FM.fused_experts
m = M.FusedMoE(Tn)
print(json.dumps(dict(kind="ship_cfg", cfg={k: str(v) for k, v in m.cfg.items()}, unroll=m.unroll, gu_flags=m.gu_flags, x=m.x)), flush=True)
OFF = [("x0w0", 0, 0), ("x128w0", 128, 0), ("x0w128", 0, 128), ("x128w128", 128, 128)]
for route in ("uniform", "zipf1"):
    d = D[(Tn, route)]
    X0, ids, wts = d["X"].cuda(), d["ids"].cuda(), d["wts"].cuda()
    sx, sw = Slot(X0), Slot(W20)
    w_ptr = sw.sp.data_ptr() & ~255
    arms = [("ours", *a) for a in OFF] + ([("vllm", *a) for a in OFF] if fe else [])
    first = {}
    for impl, n, xo, wo in arms:
        X, W2 = sx.at(W13.data_ptr(), xo), sw.at(w_ptr, wo)
        O = m(X, W13, W2, ids, wts) if impl == "ours" else fe(X, W13, W2, wts, ids.int())
        same = None
        if impl in first:
            same = bool(torch.equal(first[impl], O))
        else:
            first[impl] = O.clone()
        print(json.dumps(dict(kind="check", tag=tag, route=route, impl=impl, arm=n, bit_same_as_first=same, nonfinite=int((~torch.isfinite(O.float())).sum()))), flush=True)
    order = arms[::-1] if tag.endswith("b") else arms
    for r in range(rounds):
        sh = r * len(order) // max(rounds, 1)
        for pos, (impl, n, xo, wo) in enumerate(order[sh:] + order[:sh]):
            X, W2 = sx.at(W13.data_ptr(), xo), sw.at(w_ptr, wo)
            fn = (lambda X=X, W2=W2: m(X, W13, W2, ids, wts)) if impl == "ours" else (lambda X=X, W2=W2: fe(X, W13, W2, wts, ids.int()))
            med, mean = bench(fn)
            print(json.dumps(dict(kind="time", tag=tag, route=route, impl=impl, arm=n, round=r, pos=pos, med=med, mean=mean)), flush=True)
