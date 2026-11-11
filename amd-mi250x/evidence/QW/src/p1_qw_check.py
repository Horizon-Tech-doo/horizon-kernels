"""P1 gate for gqa_fa_qw.py, one GCD: the query-window kernel against the square kernel it replaces (the same module with qw=0, weights scaled as gqa_fa_ps) and against float64.
python p1_qw_check.py <out.jsonl> <dump dir> [more]      P1_REV=1 reverses the arm order (second job of a pair); HLP_QW=1 must be set so that horizon_lumi_prefill loads gqa_fa_qw.
Cases: (a) every dumped server step that has cached context (queries of one chunk, all keys and values, vLLM's kernel's output): float64 reference of the chunk rows, error of the window kernel,
of the square kernel and of vLLM's output; (b) random inputs at (total, chunk) = (3136, 1568), (9408, 1568), (15680, 1568), (4096, 1024), (6272, 1000: not a multiple of the 32-position block).
Per case also: are the window kernel's rows bitwise equal to the square kernel's rows (same block code, so they should be), and interleaved timings, 7 rounds, order rotated:
  qw (window kernel), sq (square kernel at the same total), sq_b (the square kernel again: identical-code control).
fail lists a non-finite result, an error above 1.25x the square kernel's (or vLLM's on dumps), or rows that are not bitwise equal."""
import glob, json, os, statistics, sys, time
import torch
import horizon_lumi_prefill as h
out = open(sys.argv[1], "a"); G, D, Hkv = 6, 256, 2; H = Hkv * G
REV = os.environ.get("P1_REV") == "1"
def P(**k): out.write(json.dumps(k) + "\n"); out.flush(); print(json.dumps(k), flush=True)
def tm(f, it=10):
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): f()
    torch.cuda.synchronize(); return (time.perf_counter() - t) * 1e3 / it
def ref_rows(q, k, v, ctx):
    n = q.shape[0]; Lk = k.shape[0]; o = torch.empty(n, H, D, dtype=torch.float64, device="cuda")
    for r0 in range(0, n, 256):
        r1 = min(n, r0 + 256)
        for hk in range(Hkv):
            sc = torch.einsum("rgd,nd->rgn", q[r0:r1, hk * G:(hk + 1) * G].double(), k[:, hk].double()) / D ** 0.5
            mask = torch.arange(Lk, device="cuda")[None, :] > (ctx + torch.arange(r0, r1, device="cuda"))[:, None]
            sc.masked_fill_(mask[:, None, :], float("-inf"))
            o[r0:r1, hk * G:(hk + 1) * G] = torch.einsum("rgn,nd->rgd", torch.softmax(sc, -1), v[:, hk].double()); del sc
    return o
m = h._mod()
P(kind="env", module=getattr(m, "__name__", "?"), rev=REV)
fail = []
def case(tag, q, k, v, ctx, stock=None):
    n = q.shape[0]; Lk = k.shape[0]
    Q = torch.zeros(1, Lk, H, D, device="cuda", dtype=torch.float16); K = k.to(torch.float16)[None].contiguous(); V = v.to(torch.float16)[None].contiguous(); Q[0, ctx:].copy_(q)
    t0 = time.time(); kq = h.build_qw(Lk, n, Hkv, G, D); ks = h.build(Lk, Hkv, G, D); tb = time.time() - t0
    ref = ref_rows(q, k, v, ctx); nr = ref.norm()
    oq = kq(Q, K, V)[0, ctx:]; os_ = ks(Q, K, V)[0, ctx:]
    row = dict(kind="cell", rev=REV, case=tag, n=n, ctx=ctx, Lk=Lk, build_s=round(tb, 1), bitwise_equal_to_square=bool(torch.equal(oq, os_)), finite=bool(torch.isfinite(oq).all()))
    for a, o in (("qw", oq), ("sq", os_)):
        d = o.to(torch.bfloat16).double() - ref
        row["rel_" + a] = float(d.norm() / nr); row["worst_row_" + a] = float((d.flatten(1).norm(dim=1) / ref.flatten(1).norm(dim=1).clamp_min(1e-300)).max())
    if stock is not None: row["rel_stock"] = float((stock - ref).norm() / nr)
    arms = {"qw": lambda: kq(Q, K, V), "sq": lambda: ks(Q, K, V), "sq_b": lambda: ks(Q, K, V)}
    for g in arms.values(): tm(g, 2)
    ts = {a: [] for a in arms}; names = list(arms)[::-1] if REV else list(arms)
    for rd in range(7):
        for i in range(3): a = names[(i + rd) % 3]; ts[a].append(tm(arms[a]))
    for a in arms: row[a + "_ms"] = round(statistics.median(ts[a]), 4)
    row["sq_over_qw"] = round(row["sq_ms"] / row["qw_ms"], 4); row["noise"] = round(abs(row["sq_b_ms"] / row["sq_ms"] - 1), 4)
    row["model_ratio"] = round(1.0 / (1.0 - (1.0 - (-(-n // 32) * 32) / Lk) ** 2), 3)      # square work / window work if cost were rows x keys below the diagonal
    lim = 1.25 * max(row["rel_sq"], row.get("rel_stock", 0.0))
    if not row["finite"] or row["rel_qw"] > lim or not row["bitwise_equal_to_square"]: fail.append([tag, row["rel_qw"], row["rel_sq"], row["bitwise_equal_to_square"]])
    P(**row); torch.cuda.empty_cache()
for f in sorted(f for d in sys.argv[2:] for f in glob.glob(d + "/hlp-dump-*.pt")):
    t = torch.load(f)
    if int(t["ctx"]) == 0: continue
    case("dump:" + f.split("/")[-1][9:-3], t["q"].cuda(), t["k"].cuda(), t["v"].cuda(), int(t["ctx"]), t["stock"].cuda().double()); del t
for Lk, n in ((3136, 1568), (9408, 1568), (15680, 1568), (4096, 1024), (6272, 1000)):
    g = torch.Generator(device="cuda").manual_seed(Lk + n)
    q = torch.randn(n, H, D, device="cuda", generator=g).bfloat16(); k = torch.randn(Lk, Hkv, D, device="cuda", generator=g).bfloat16(); v = torch.randn(Lk, Hkv, D, device="cuda", generator=g).bfloat16()
    case(f"random:{Lk}:{n}", q, k, v, Lk - n); del q, k, v
P(kind="summary", fail=fail, ok=not fail)
