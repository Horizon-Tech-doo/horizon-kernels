# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""G3v protocol vs vLLM 0.30 align machinery WITH the caching package's state traffic, CPU simulator (01-10-2026).

usage: python test_g3v_caching_sim_cpu.py       (torch; next to g3v_ref.py / gdn_state_ref.py; exit 0 = all pass)

test_g3v_cpu.py proves G3v against vLLM's decode / align copies for requests that only decode. The caching package
(HORIZON_PI + HORIZON_GDN_CKPT stop fill + --prefix-match-unit + --enable-mamba-fine-grained-prefix-cache) adds state
traffic around PREFILL: in-step checkpoint and fill exports into the prefill request's own blocks, the partial-tail
copy-on-write of the running block at the prompt's last hash boundary (queued, applied before the next forward),
prefix-cache hits that copy a cached state (ssm + conv) into a new request, block freeing / reuse with zeroing of
freshly allocated blocks, request-state indices reused by new requests. This simulator runs two worlds over the same
request stream (one GDN layer, block pool, prompts sharing prefixes, mixed prefill + decode steps, the 3/1 draft
schedule, random acceptance):
  world V : vLLM (window kernel writes every window token's state; conv state in slot 0's conv region)
  world G : G3v (ring in the conv regions of window slots 1..3, the plugin's H1 / H2 / H3 hooks)
Gates: decode outputs every step, every state read through a copy (pre-copy, post-copy, copy-on-write, cache hit, the
prefill's initial state and conv history) bit-identical, every cache entry's ssm AND conv state identical at the end,
committed states of live requests identical after a final expand. PI changes prefill kernels only (no state
traffic): modelled as the same prefill arithmetic in both worlds.
"""
import random
import sys

import torch

import g3v_ref as V
import gdn_state_ref as R

HV, H, DK, DV = 4, 2, 128, 128
REC = V.rec_floats(HV, H)
PPS = 3
RF = PPS * REC                         # conv region floats per block (= the conv state's size, as in production)
SW = 4
CWS = 6                                # conv state rows kept: width - 1 + drafts (3 + 3), stored at the region start
PPS_, HALF = V.ring_geometry(HV, H, RF, SW)
assert PPS_ == PPS and HALF == 4
FAILS = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def bits(x, y):
    return int(((x.view(torch.int32) != y.view(torch.int32)) & ~((x == 0) & (y == 0))).sum())


class Tok:
    """deterministic per-token inputs: the same token id at the same position -> the same inputs in both worlds"""

    def __init__(self, seed):
        self.seed, self.cache = seed, {}

    def get(self, key):
        if key not in self.cache:
            g = torch.Generator().manual_seed(hash((self.seed,) + key) % (2 ** 31))
            qkvz = torch.nn.functional.silu(torch.randn(1, 2 * H * DK + 2 * HV * DV, generator=g))
            ba = torch.randn(1, 2 * HV, generator=g) * 2
            conv = torch.randn(1, generator=g)
            self.cache[key] = (qkvz.bfloat16().float(), ba.bfloat16().float(), float(conv))
        return self.cache[key]


def inputs_for(keys, tok):
    rows = [tok.get(k) for k in keys]
    qkvz = torch.cat([r[0] for r in rows])
    ba = torch.cat([r[1] for r in rows])
    CQ = 2 * H * DK + HV * DV
    return qkvz[:, :CQ], ba[:, HV:], ba[:, :HV], qkvz[:, CQ:].view(-1, HV, DV), [r[2] for r in rows]


class Req:
    def __init__(self, rid, prompt, gen_len, rs):
        self.rid, self.prompt, self.gen_len, self.rs = rid, prompt, gen_len, rs
        self.blocks = []                  # column -> block id (0 = null)
        self.computed = 0
        self.col = 0
        self.acc = 1
        self.out_tokens = []
        self.tail_cow = None              # (prefix length, source block) queued at a prefill chunk end
        self.hit = 0

    @property
    def prompt_len(self):
        return len(self.prompt)

    def prefilling(self):
        return self.computed < self.prompt_len


class World:
    def __init__(self, g3, B, u, NB, A_log, dt_bias, norm_w):
        self.g3, self.B, self.u = g3, B, u
        self.st = V.Store(NB, HV, RF)
        self.st.ring.zero_()
        self.free = list(range(1, NB))
        self.cache = {}                   # prefix tuple -> block id (immutable once registered)
        self.cached_blocks = set()
        self.meta = torch.zeros(64, HV, DV // V.CH, dtype=torch.int32)
        self.A_log, self.dt_bias, self.norm_w = A_log, dt_bias, norm_w
        self.obs = []                     # (what, tensor) states / conv histories read through copies or kernels
        self.stats = {"flushes": 0, "steps": 0, "replayed": 0, "window_mode": 0, "expanded": 0, "ckpt": 0,
                      "fill": 0, "cow": 0, "hits": 0, "reused_blocks": 0, "precopy": 0, "postcopy": 0}
        self.zero_queue = []
        self.neg = None                   # negative controls: "no_h1_mixed", "cow_decode"

    # ---- block pool (vLLM: fresh blocks are zeroed in update_states, before the step's hooks and forward)
    def alloc(self):
        b = self.free.pop(0)
        self.zero_queue.append(b)
        return b

    def release(self, req):
        for b in req.blocks:
            if b > 0 and b not in self.cached_blocks:
                self.free.append(b)
                self.stats["reused_blocks"] += 1
        req.blocks = []

    def ensure_cols(self, req, upto):
        while len(req.blocks) <= upto:
            req.blocks.append(self.alloc())

    def apply_update_states(self, reqs):
        for b in self.zero_queue:          # zero freshly allocated blocks
            self.st.S[b].zero_()
            self.st.ring[b].zero_()
        self.zero_queue = []
        for r in reqs:                      # queued copy-on-write of partial tails (before the forward)
            if r.tail_cow is not None:
                L, src = r.tail_cow
                dst = self.free.pop(0)
                self.st.S[dst] = self.st.S[src]
                self.st.ring[dst] = self.st.ring[src]       # whole block page: ssm + conv region
                self.obs.append(("cow", self.st.S[src].clone(), self.st.ring[src, :CWS].clone()))
                self.cache[tuple(r.prompt[:L])] = dst
                self.cached_blocks.add(dst)
                self.stats["cow"] += 1
                r.tail_cow = None
            if self.neg == "cow_decode" and not r.prefilling() and len(r.blocks) > r.col + 1:
                src = r.blocks[r.col + 1]          # hypothetical: a whole-page copy of a DECODING request's scratch
                self.obs.append(("cow_decode", self.st.S[src].clone(), self.st.ring[src].clone()))

    def window(self, req, col):
        return [req.blocks[col + k] if col + k < len(req.blocks) else 0 for k in range(SW)]

    # ---- G3v plugin hooks
    def expand(self, rows):
        if not self.g3 or not rows:
            return
        sidx = torch.tensor([self.window(r, r.col) for r in rows], dtype=torch.int32)
        ridx = torch.tensor([r.rs for r in rows], dtype=torch.int32)
        self.stats["expanded"] += V.g3v_expand(sidx, ridx, torch.ones(len(rows), dtype=torch.int32), self.st,
                                               self.meta, HALF, PPS, H)

    def add_request(self, req, tok):
        if self.g3:
            self.meta[req.rs] = 0                                   # H3
        # prefix-cache hit: the longest cached prefix of the prompt (strictly shorter than the prompt)
        best = 0
        for L in range(req.prompt_len - 1, 0, -1):
            if tuple(req.prompt[:L]) in self.cache:
                best = L
                break
        if best:
            src = self.cache[tuple(req.prompt[:best])]
            req.col = (best - 1) // self.B
            req.blocks = [0] * req.col
            self.ensure_cols(req, req.col)
            self.apply_update_states([])                            # zero the fresh block, then the hit copy
            dst = req.blocks[req.col]
            self.st.S[dst] = self.st.S[src]
            self.st.ring[dst, :CWS] = self.st.ring[src, :CWS]
            self.obs.append(("hit", self.st.S[src].clone(), self.st.ring[src, :CWS].clone()))
            req.computed = best
            req.hit = best
            self.stats["hits"] += 1
        else:
            req.col = 0
            self.ensure_cols(req, 0)

    # ---- one step for a batch: prefill chunks (ends) and decode query lengths
    def step(self, batch, tok, step_no):
        """batch: [(req, kind, n_tokens, ckpt_pos or 0)]; returns the decode outputs (list of tensors)"""
        reqs = [b[0] for b in batch]
        dec = [b for b in batch if b[1] == "decode"]
        mixed = any(b[1] == "prefill" for b in batch)
        # update_states: new window / chunk columns are allocated now (zeroed), queued CoW copies applied
        for r, kind, n, _ in batch:
            end = r.computed + n
            self.ensure_cols(r, (end - 1) // self.B + (SW - 1 if kind == "decode" else 0))
        self.apply_update_states(reqs)
        # H1 (G3v): crossing rows, or every decode row when the batch has prefill
        if self.g3:
            rows = []
            for r, kind, n, _ in dec:
                new = (r.computed + n - 1) // self.B
                if (mixed and self.neg != "no_h1_mixed") or new != r.col:
                    rows.append(r)
            self.expand(rows)
        # vLLM pre-copy (every row whose running column moves)
        for r, kind, n, _ in batch:
            new = (r.computed + n - 1) // self.B
            if new != r.col:
                src = r.blocks[r.col + r.acc - 1]
                srcc = r.blocks[r.col]
                bias = r.acc - 1
                self.obs.append(("precopy", self.st.S[src].clone(), self.st.ring[srcc, bias:CWS].clone()))
                self.st.S[r.blocks[new]] = self.st.S[src]
                self.st.ring[r.blocks[new], :CWS - bias] = self.st.ring[srcc, bias:CWS].clone()
                r.col, r.acc = new, 1
                self.stats["precopy"] += 1
        outs = []
        # prefill rows (FLA in both worlds; PI = the same arithmetic in both worlds)
        for r, kind, n, ck in batch:
            if kind != "prefill":
                continue
            s0 = r.blocks[r.col]
            h = self.st.S[s0].clone()
            hist = self.st.ring[s0, :CWS].clone()
            self.obs.append(("prefill_init", h.clone(), hist.clone()))
            start = r.computed
            keys = [(tuple(r.prompt[:p + 1]),) for p in range(start, start + n)]
            qkv, a, b, z, cv = inputs_for(keys, tok)
            q, k, v, g, beta = R._prep(qkv, a, b, self.A_log, self.dt_bias, HV, H, DK ** -0.5)
            G = HV // H
            ck_col = -(-(start + n) // self.B) - 2
            for t in range(n):
                h, _, _ = R._token(h, k[t].repeat_interleave(G, 0), q[t].repeat_interleave(G, 0), v[t], g[t], beta[t])
                hist = torch.cat([hist[1:], torch.tensor([cv[t]])])
                pos = start + t + 1
                if ck and pos == ck and ck_col >= 0:                 # in-step checkpoint export (own block)
                    blk = r.blocks[ck_col]
                    self.st.S[blk] = h
                    self.st.ring[blk, :CWS] = hist
                    self.cache[tuple(r.prompt[:pos])] = blk
                    self.cached_blocks.add(blk)
                    self.stats["ckpt"] += 1
                if ck and pos % self.B == 0 and pos < start + n and pos // self.B - 1 < ck_col:   # fill export
                    blk = r.blocks[pos // self.B - 1]
                    if blk > 0 and blk not in self.cached_blocks:
                        self.st.S[blk] = h
                        self.st.ring[blk, :CWS] = hist
                        self.cache[tuple(r.prompt[:pos])] = blk
                        self.cached_blocks.add(blk)
                        self.stats["fill"] += 1
            end = start + n
            s1 = r.blocks[(end - 1) // self.B]
            self.st.S[s1] = h
            self.st.ring[s1, :CWS] = hist
            r.computed, r.acc = end, 1
            if end % self.B == 0:                                   # chunk ends on a block boundary: cached
                self.cache[tuple(r.prompt[:end])] = s1
                self.cached_blocks.add(s1)
            T = (r.prompt_len // self.u) * self.u                   # fine-grained partial tail
            if end == T and T % self.B and end < r.prompt_len + 1:
                r.tail_cow = (T, s1)
        # decode rows
        if dec:
            keys, cu = [], [0]
            for r, kind, n, _ in dec:
                keys += [(r.rid, len(r.out_tokens) + t) for t in range(n)]
                cu.append(cu[-1] + n)
            qkv, a, b, z, cv = inputs_for(keys, tok)
            sidx = torch.tensor([self.window(r, r.col) for r, _, _, _ in dec], dtype=torch.int32)
            nacc = torch.tensor([r.acc for r, _, _, _ in dec], dtype=torch.int32)
            cu_t = torch.tensor(cu, dtype=torch.int32)
            out = torch.full((len(keys), HV, DV), 7.0, dtype=torch.bfloat16)
            for i, (r, _, _, _) in enumerate(dec):                  # conv history read by the conv update
                s0 = r.blocks[r.col]
                self.obs.append(("conv_hist", self.st.ring[s0, r.acc - 1: r.acc + 2].clone(), torch.zeros(1)))
            if self.g3 and not mixed:
                ridx = torch.tensor([r.rs for r, _, _, _ in dec], dtype=torch.int32)
                V.g3v_step(qkv, a, b, self.A_log, self.dt_bias, sidx, cu_t, nacc, ridx, self.st, self.meta, z,
                           self.norm_w, out, HALF, PPS, stats=self.stats)
            else:
                R.window_step(qkv, a, b, self.A_log, self.dt_bias, sidx, cu_t, nacc, self.st.S, z, self.norm_w, out)
            for i, (r, _, n, _) in enumerate(dec):                  # conv update into slot 0 (both worlds)
                s0 = r.blocks[r.col]
                hist = self.st.ring[s0, r.acc - 1: r.acc + 2].clone()
                new = torch.cat([hist, torch.tensor(cv[cu[i]: cu[i + 1]])])
                self.st.ring[s0, :CWS] = 0
                self.st.ring[s0, :min(CWS, new.numel())] = new[-CWS:] if new.numel() >= CWS else new
            outs.append(out.float())
        return outs

    def post(self, dec_acc):
        """dec_acc: [(req, n_accepted)] for decode rows; vLLM's align post-copy, G3v's H2 before it"""
        cps = []
        for r, acc in dec_acc:
            r.acc = acc
            r.computed += acc
            r.out_tokens += [0] * acc
            running = r.computed - acc + 1
            aligned = (r.computed // self.B) * self.B
            if aligned >= running:
                cps.append((r, aligned - running, aligned // self.B - 1, aligned))
        self.expand([c[0] for c in cps])                            # H2
        for r, bias, dest, aligned in cps:
            if r.col == dest:
                r.acc = 1
            if r.col == dest and bias == 0:
                blk = r.blocks[dest]
            else:
                src = r.blocks[r.col + bias]
                self.obs.append(("postcopy", self.st.S[src].clone(), self.st.ring[r.blocks[r.col], bias:CWS].clone()))
                self.st.S[r.blocks[dest]] = self.st.S[src]
                self.st.ring[r.blocks[dest], :CWS - bias] = self.st.ring[r.blocks[r.col], bias:CWS].clone()
                blk = r.blocks[dest]
                self.stats["postcopy"] += 1
            # the block-boundary state of the request's full sequence so far (decode outputs are token 0s)
            key = tuple(r.prompt + [0] * (aligned - r.prompt_len)) if aligned > r.prompt_len else None
            if key is not None and key not in self.cache:
                self.cache[key] = blk
                self.cached_blocks.add(blk)

    def committed(self, r):
        return self.st.S[r.blocks[r.col + r.acc - 1]]


def run(seed, B=8, u=2, nreq=24, steps=220, max_running=6, neg=None):
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    A_log, dt_bias = torch.rand(HV) * 2 - 1, torch.randn(HV)
    norm_w = 1 + 0.1 * torch.randn(DV)
    tok = Tok(seed)
    NB = 4000
    WV, WG = World(False, B, u, NB, A_log, dt_bias, norm_w), World(True, B, u, NB, A_log, dt_bias, norm_w)
    WV.neg = WG.neg = neg
    bases = [[rnd.randrange(50) for _ in range(rnd.randrange(6, 30))] for _ in range(4)]
    pending = []
    for i in range(nreq):                  # prompts share prefixes (system prompts, follow-up turns)
        base = list(bases[rnd.randrange(len(bases))])
        if rnd.random() < 0.5:
            base = base[: rnd.randrange(1, len(base) + 1)]
        p = base + [rnd.randrange(50) for _ in range(rnd.randrange(1, 25))]
        pending.append((p, rnd.randrange(6, 40)))
    running = {}                           # rid -> (reqV, reqG)
    free_rs = list(range(40))
    rid = 0
    out_diff = 0
    for s in range(steps):
        while pending and len(running) < max_running and rnd.random() < 0.6:
            p, gl = pending.pop(0)
            rs = free_rs.pop(rnd.randrange(len(free_rs)))
            rv, rg = Req(rid, p, gl, rs), Req(rid, p, gl, rs)
            WV.add_request(rv, tok)
            WG.add_request(rg, tok)
            running[rid] = (rv, rg)
            rid += 1
        if not running:
            if not pending:
                break
            continue
        many = len(running) > 3                                     # the 3/1 draft schedule (scaled)
        bV, bG = [], []
        for k, (rv, rg) in list(running.items()):
            if rv.prefilling():
                rem = rv.prompt_len - rv.computed
                # chunk stops: a block boundary, the fine-grained tail, the prompt end
                T = (rv.prompt_len // u) * u
                cands = [rv.prompt_len - rv.computed]
                nb_ = ((rv.computed // B) + 1) * B - rv.computed
                if 0 < nb_ < rem:
                    cands.append(nb_)
                if rv.computed < T < rv.prompt_len and T % B:
                    cands.append(T - rv.computed)
                n = min(rnd.choice(cands), rem)
                ck = 0
                end = rv.computed + n
                if -(-end // B) - 2 >= 0 and rnd.random() < 0.7:     # in-step checkpoint at a u-aligned position
                    lo = max(rv.computed + 1, (-(-end // B) - 2) * B + 1)
                    opts = [x for x in range(lo, end) if x % u == 0 and (x - 1) // B == -(-end // B) - 2]
                    if opts:
                        ck = rnd.choice(opts)
                bV.append((rv, "prefill", n, ck))
                bG.append((rg, "prefill", n, ck))
            else:
                ql = 2 if many else 4
                bV.append((rv, "decode", ql, 0))
                bG.append((rg, "decode", ql, 0))
        oV, oG = WV.step(bV, tok, s), WG.step(bG, tok, s)
        out_diff += sum(bits(x, y) for x, y in zip(oV, oG))
        accs = [(rv, rnd.randint(1, n)) for rv, kind, n, _ in bV if kind == "decode"]
        WV.post(accs)
        WG.post([(running[r.rid][1], a) for r, a in accs])
        for k, (rv, rg) in list(running.items()):
            if not rv.prefilling() and len(rv.out_tokens) >= rv.gen_len:
                WG.expand([rg])                                     # (a finished request's ring is just dropped
                WV.release(rv)                                      #  in the plugin; expanding here only lets the
                WG.release(rg)                                      #  final comparison read committed states)
                free_rs.append(rv.rs)
                del running[k]
    WG.expand([rg for _, rg in running.values()])
    obs_diff = (len(WV.obs) != len(WG.obs)) + sum(bits(a[1], b[1]) + bits(a[2], b[2])
                                                  for a, b in zip(WV.obs, WG.obs))
    cache_keys = set(WV.cache) | set(WG.cache)
    cache_diff = sum((k not in WV.cache) + (k not in WG.cache) for k in cache_keys)
    for k in set(WV.cache) & set(WG.cache):
        bv, bg = WV.cache[k], WG.cache[k]
        cache_diff += bits(WV.st.S[bv], WG.st.S[bg]) + bits(WV.st.ring[bv, :CWS], WG.st.ring[bg, :CWS])
    fin = sum(bits(WV.committed(rv), WG.committed(rg)) for rv, rg in running.values())
    return out_diff, obs_diff, cache_diff, fin, WG.stats, len(WV.obs), len(cache_keys)


if __name__ == "__main__":
    total = {}
    for seed, kw in ((1, {}), (2, {"B": 6, "u": 3}), (3, {"B": 10, "u": 2, "max_running": 8}),
                     (4, {"B": 4, "u": 2}), (5, {"B": 8, "u": 4, "nreq": 30, "steps": 300})):
        o, ob, c, f, st, nobs, ncache = run(seed, **kw)
        for k, v in st.items():
            total[k] = total.get(k, 0) + v
        check(f"caching traffic, seed {seed} {kw}", o == 0 and ob == 0 and c == 0 and f == 0,
              f"out {o}, observed reads {ob} ({nobs} reads), cache entries {c} ({ncache} entries), committed {f}; "
              f"G3v steps {st['steps']}, flushes {st['flushes']}, expanded {st['expanded']}, ckpt {st['ckpt']}, "
              f"fill {st['fill']}, cow {st['cow']}, hits {st['hits']}, reused blocks {st['reused_blocks']}")
    for neg in ("no_h1_mixed", "cow_decode"):
        o, ob, c, f, st, nobs, ncache = run(1, neg=neg)
        check(f"negative control {neg}: the simulator detects it", o + ob + c + f > 0,
              f"out {o}, observed {ob}, cache {c}, committed {f}")
    check("coverage: every kind of caching traffic occurred",
          all(total[k] > 0 for k in ("steps", "flushes", "expanded", "ckpt", "fill", "cow", "hits", "reused_blocks",
                                     "precopy", "postcopy")), str(total))
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)
