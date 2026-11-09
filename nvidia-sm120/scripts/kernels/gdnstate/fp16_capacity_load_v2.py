#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Resident capacity on the isolated WP1.3 engine, under hz GPU lock only.

Synthetic unique prefixes, 4k/32k approximate total context including output.
Warm-prefix mode removes cold-prefill serialization from the residency probe.
A clean level requires simultaneous full-context decode for every request,
exact output counts, zero preemption, valid metrics and no other requests.
The largest clean level is a measured lower bound. An overlap/scheduling failure
is not evidence of a memory limit. Original fixed-token load method retained.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse
import urllib.request

WORDS = ("the of and to in that is was for on with as by at from this which or an be are have has not but they "
         "their it its were been more one all would there when who will out up about into than them can only other "
         "new some time could these two may first then do any like over such our after most also made did many "
         "before must through back years where much your way well down should because each just those people how "
         "too little state good very make world still own see men work long get here between both life being under "
         "never day same another know while last might us great old year off come since against go came right used "
         "take three").split()


def post(base, path, body, timeout=600):
    r = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


class Prompts:
    def __init__(self, base, model, corpus, seed):
        self.base, self.model = base, model
        self.rng = random.Random(seed)
        self.text = open(corpus, encoding="utf-8", errors="ignore").read() if corpus and os.path.exists(corpus) else ""
        if len(self.text) < 20000:
            self.text = " ".join(self.rng.choice(WORDS) for _ in range(60000))
        self.words = self.text.split()
        self.ratio = None

    def ntok(self, s):
        return len(post(self.base, "/tokenize", {"model": self.model, "prompt": s})["tokens"])

    def make(self, n_tokens, i):
        if self.ratio is None:
            sample = " ".join(self.words[:4000])
            self.ratio = self.ntok(sample) / 4000.0
        nonce = f"[session {i}-{self.rng.randrange(10**9)}]"
        need = max(16, int((n_tokens - 40) / self.ratio))
        start = self.rng.randrange(len(self.words))
        ws = [self.words[(start + k) % len(self.words)] for k in range(need)]
        s = nonce + "\n" + " ".join(ws) + "\n\nWrite a detailed, structured summary of the text above."
        # one exact correction step (chat template adds ~10-20 tokens; the target is approximate by design)
        got = self.ntok(s)
        if abs(got - n_tokens) > 32:
            need = max(16, need + int((n_tokens - got) / self.ratio))
            ws = [self.words[(start + k) % len(self.words)] for k in range(need)]
            s = nonce + "\n" + " ".join(ws) + "\n\nWrite a detailed, structured summary of the text above."
        return s


def metric(text, name):
    tot, seen = 0.0, False
    for line in text.splitlines():
        if line.startswith(name + "{") or line.startswith(name + " "):
            try:
                tot += float(line.rsplit(" ", 1)[1])
                seen = True
            except ValueError:
                pass
    return tot if seen else None


class Sampler(threading.Thread):
    NAMES = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting",
             "kv": "vllm:kv_cache_usage_perc", "preempt": "vllm:num_preemptions_total", "success": "vllm:request_success_total"}

    def __init__(self, base):
        super().__init__(daemon=True)
        self.base, self.stop, self.rows = base, False, []

    def snap(self):
        try:
            t = urllib.request.urlopen(self.base + "/metrics", timeout=5).read().decode()
        except Exception:  # noqa: BLE001
            return None
        return {k: metric(t, n) for k, n in self.NAMES.items()}

    def run(self):
        while not self.stop:
            s = self.snap()
            if s:
                self.rows.append(s)
            time.sleep(0.25)


def one(base, model, prompt, out_tokens, res, delay, temperature):
    time.sleep(delay)
    body = {"model": model, "stream": True, "max_tokens": out_tokens, "ignore_eos": True, "temperature": temperature,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    first = None
    usage = {}
    try:
        req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                ch = d.get("choices") or []
                if ch and first is None and (ch[0].get("delta", {}).get("content") or
                                             ch[0].get("delta", {}).get("reasoning_content")):
                    first = time.time()
                if d.get("usage"):
                    usage = d["usage"]
    except Exception as e:  # noqa: BLE001
        res.append({"error": f"{type(e).__name__}: {e}"[:200]})
        return
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    res.append({"ttft": (first or t1) - t0, "tokens": n, "prompt_tokens": usage.get("prompt_tokens"),
                "decode_tok_s": (n - 1) / max(1e-6, t1 - (first or t0)) if n > 1 else 0.0, "t_start": t0, "t_first": first, "t_end": t1})


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))] if v else None


def peak_decode_overlap(rows):
    events = []
    for r in rows:
        if r.get('t_first') is not None and r.get('t_end', 0) > r['t_first']:
            events.extend([(r['t_first'], 1), (r['t_end'], -1)])
    current = peak = 0
    for _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def level(base, model, prompts, ctx, c, out_tokens, stagger, temperature):
    ps = [prompts.make(ctx - out_tokens, i) for i in range(c)]
    # Populate the same synthetic prefixes before the resident decode test.
    # This separates capacity from cold-prefill serialization; retained entries
    # that do not fit are evicted by the engine and cannot create a false hit.
    if getattr(prompts, 'warm_prefixes', False):
        for p in ps:
            body = {"model": model, "max_tokens": 1, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "user", "content": p}]}
            post(base, "/v1/chat/completions", body)
    smp = Sampler(base)
    before = smp.snap() or {}
    if (before.get("running") or 0) + (before.get("waiting") or 0) > 0:
        raise RuntimeError("capacity measurement requires an idle engine")
    smp.start()
    res, th = [], []
    t0 = time.time()
    for i, p in enumerate(ps):
        d = stagger * i / max(1, c - 1) if stagger else 0.0
        t = threading.Thread(target=one, args=(base, model, p, out_tokens, res, d, temperature))
        t.start()
        th.append(t)
    for t in th:
        t.join()
    dt = time.time() - t0
    smp.stop = True
    smp.join()
    after = smp.snap() or {}
    errs = [r["error"] for r in res if "error" in r]
    ok = [r for r in res if "error" not in r]
    rows = smp.rows
    peak = lambda k: max((r[k] for r in rows if r.get(k) is not None), default=None)  # noqa: E731
    pre = (after.get("preempt") or 0) - (before.get("preempt") or 0) if after.get("preempt") is not None else None
    tt = [r["ttft"] for r in ok]
    ds = [r["decode_tok_s"] for r in ok]
    wrong = sum(1 for r in ok if r["tokens"] != out_tokens)
    row = {"context": ctx, "level": c, "out_tokens": out_tokens, "requests_ok": len(ok), "errors": len(errs),
           "wrong_token_counts": wrong, "wall_s": round(dt, 1),
           "total_tok_s": round(sum(r["tokens"] for r in ok) / dt, 1),
           "user_tok_s_p50": round(statistics.median(ds), 1) if ds else None,
           "user_tok_s_p5": round(pct(ds, 5), 1) if ds else None,
           "ttft_p50": round(statistics.median(tt), 2) if tt else None, "ttft_p95": round(pct(tt, 95), 2) if tt else None,
           "prompt_tokens_mean": round(statistics.mean([r["prompt_tokens"] or 0 for r in ok])) if ok else None,
           "peak_running": peak("running"), "peak_waiting": peak("waiting"),
           "peak_kv_usage": round(peak("kv"), 4) if peak("kv") is not None else None, "preemptions": pre,
           "first_errors": errs[:3], "peak_decode_overlap": peak_decode_overlap(ok),
           "mode": "warm_prefix" if getattr(prompts, 'warm_prefixes', False) else "cold",
           "finished_delta": (after["success"]-before["success"] if before.get("success") is not None and after.get("success") is not None else None)}
    # all c requests resident at once (waiting > 0 alone is normal at a simultaneous start: prefills queue for the
    # token budget), no preemption, no error, exact token counts
    row["all_resident"] = (row["peak_running"] or 0) >= c
    row["clean"] = (len(ok) == c and not errs and not wrong and row["all_resident"]
                    and row["peak_decode_overlap"] == c and pre == 0
                    and row["peak_kv_usage"] is not None and row["finished_delta"] == c
                    and (row["peak_running"] or 0) <= c)
    row["limit_reason"] = ("traffic_or_missing_success_counter" if row["finished_delta"] != c or (row["peak_running"] or 0) > c else
                           "request_error" if errs or wrong else
                           "preemption" if pre else
                           "missing_metrics" if pre is None or row["peak_kv_usage"] is None else
                           "incomplete_full_context_overlap" if row["peak_decode_overlap"] < c else
                           "resident_peak" if not row["all_resident"] else None)
    return row


def vllm_estimates(log):
    out = {}
    if log and os.path.exists(log):
        for line in open(log, errors="ignore"):
            m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", line)
            if m:
                out[m.group(1)] = float(m.group(2))
            m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
            if m:
                out["kv_cache_tokens"] = int(m.group(1).replace(",", ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("matrix", "probe"))
    ap.add_argument("tag")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3.8")
    ap.add_argument("--levels", default="1,8,32,50,64")
    ap.add_argument("--contexts", default="4096,32768")
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--out-tokens", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--start", type=int, default=8)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--max", type=int, default=128)
    ap.add_argument("--corpus", default=os.path.expanduser("$HK_WORK/corpus/prose.md"))
    ap.add_argument("--vllm-log")
    ap.add_argument("--warm-prefixes", action="store_true")
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--out", default=os.path.expanduser("~/flashnext_load.jsonl"))
    ap.add_argument("--allow-remote", action="store_true")
    a = ap.parse_args(argv)
    host = urllib.parse.urlparse(a.base).hostname
    if host not in ("127.0.0.1", "localhost", "::1") and not a.allow_remote:
        raise SystemExit(f"{a.base}: load tests only against the rented box's loopback vLLM")
    prompts = Prompts(a.base, a.model, a.corpus, a.seed)
    prompts.warm_prefixes = a.warm_prefixes
    outf = open(a.out, "a")

    def emit(d):
        d = {"tag": a.tag, "ts": time.strftime("%d-%m-%Y %H:%M:%S"), **d}
        print(json.dumps(d), flush=True)
        outf.write(json.dumps(d) + "\n")
        outf.flush()

    emit({"test": "vllm_estimates", **vllm_estimates(a.vllm_log)})
    if a.mode == "matrix":
        summary = []
        for ctx in [int(x) for x in a.contexts.split(",")]:
            for c in [int(x) for x in a.levels.split(",")]:
                level(a.base, a.model, prompts, min(ctx, 1024), min(c, 4), 16, 0, a.temperature)   # warm-up
                rows = []
                for rnd in range(a.rounds):
                    r = level(a.base, a.model, prompts, ctx, c, a.out_tokens, a.stagger, a.temperature)
                    emit({"test": "round", "round": rnd, **r})
                    rows.append(r)
                med = lambda k: statistics.median([r[k] for r in rows if r[k] is not None]) if any(r[k] is not None for r in rows) else None  # noqa: E731,E501
                s = {"context": ctx, "level": c, "total_tok_s": med("total_tok_s"), "user_tok_s_p50": med("user_tok_s_p50"),
                     "user_tok_s_p5": med("user_tok_s_p5"), "ttft_p50": med("ttft_p50"), "ttft_p95": med("ttft_p95"),
                     "peak_kv_usage": med("peak_kv_usage"), "peak_waiting": max((r["peak_waiting"] or 0) for r in rows),
                     "preemptions": sum((r["preemptions"] or 0) for r in rows), "clean": all(r["clean"] for r in rows)}
                emit({"test": "level", **s})
                summary.append(s)
        emit({"test": "complete", "levels": len(summary)})
    else:
        c, best, last = a.start, None, None
        while c <= a.max:
            r = level(a.base, a.model, prompts, a.context, c, a.out_tokens, a.stagger, a.temperature)
            emit({"test": "probe", **r})
            last = r
            if not r["clean"]:
                break
            best = c
            c += a.step
        emit({"test": "probe_result", "context": a.context, "max_clean_concurrency": best,
              "first_unclean": None if last is None or last["clean"] else last["level"],
              "limit_reason": last.get("limit_reason") if last else None,
              "mode": "warm_prefix" if a.warm_prefixes else "cold"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
