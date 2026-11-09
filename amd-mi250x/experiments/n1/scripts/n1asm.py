"""Load-grouping summary of a gfx90a disassembly (one .dis per code object). Finds loops from backward branches and reports
the hot K loop, the innermost loop that holds MFMA. For that loop it prints a token stream: L<n> = n global_load_dwordx4 in
a row, l = other global load, W<k> = s_waitcnt vmcnt(k), S<n> = LDS writes, R<n> = LDS reads, M<n> = MFMA, B = forward branch.
load_groups = how many dwordx4 loads leave between two vmcnt waits; waits_between_loads = vmcnt waits inside the load run.
Usage: n1asm.py a.dis [b.dis ...]   JSON per file."""
import hashlib, json, re, sys
LINE = re.compile(r"^\s+(\S+)(.*?)//\s*([0-9A-Fa-f]+):")

def parse(path):
    ins = []
    for l in open(path):
        m = LINE.match(l)
        if m:
            t = re.search(r"<main_kernel\+0x([0-9a-f]+)>", l)
            ins.append((int(m.group(3), 16), m.group(1), m.group(2).strip(), int(t.group(1), 16) if t else None))
    return ins

def summary(path):
    ins = parse(path)
    base = ins[0][0] if ins else 0
    addr = [i[0] - base for i in ins]
    # code bytes hashed without addresses, so two identical builds compare equal
    sha = hashlib.sha256("\n".join(f"{o} {a}" for _, o, a, _t in ins).encode()).hexdigest()[:12]
    loops = []
    for k, (ad, op, _, tgt) in enumerate(ins):
        if op.startswith("s_cbranch") or op == "s_branch":
            if tgt is not None and tgt - 0 <= ad - base:
                s = next((j for j, x in enumerate(addr) if x >= tgt), None)
                if s is not None:
                    loops.append((s, k))
    withm = [(s, e) for s, e in loops if any(ins[j][1].startswith("v_mfma") for j in range(s, e + 1))]
    res = dict(file=path.split("/")[-1], sha=sha, n_ins=len(ins), code_bytes=(ins[-1][0] - base + 8) if ins else 0,
               mfma=sum(i[1].startswith("v_mfma") for i in ins), load4=sum(i[1] == "global_load_dwordx4" for i in ins),
               load_all=sum(i[1].startswith("global_load") for i in ins), vmcnt=sum(i[1] == "s_waitcnt" and "vmcnt" in i[2] for i in ins))
    g, c = {}, 0          # dwordx4 loads between two vmcnt waits, whole function: a table of run length -> how many runs
    for i in ins:
        if i[1] == "global_load_dwordx4": c += 1
        elif i[1] == "s_waitcnt" and "vmcnt" in i[2] and c: g[c] = g.get(c, 0) + 1; c = 0
    res["load_runs"] = dict(sorted(g.items()))
    full = not withm     # K loop fully unrolled: no loop holds MFMA, so report the stream from the first load to the last MFMA
    if full:
        idx = [j for j, i in enumerate(ins) if i[1] == "global_load_dwordx4" or i[1].startswith("v_mfma")]
        if not idx:
            return res
        s, e = idx[0], idx[-1]
    else:
        s, e = min(withm, key=lambda x: x[1] - x[0])   # innermost
    toks, groups, cur, between, seen = [], [], 0, 0, False
    def push(t):
        if toks and toks[-1][0] == t: toks[-1][1] += 1
        else: toks.append([t, 1])
    for j in range(s, e + 1):
        _, op, ar, _t = ins[j]
        if op == "global_load_dwordx4":
            push("L"); cur += 1; seen = True
        elif op.startswith("global_load"): push("l")
        elif op == "s_waitcnt" and "vmcnt" in ar:
            m = re.search(r"vmcnt\((\d+)\)", ar)
            toks.append([f"W{m.group(1)}", 1])
            if cur: groups.append(cur); cur = 0
            if seen: between += 1
        elif op.startswith("ds_write"): push("S")
        elif op.startswith("ds_read"): push("R")
        elif op.startswith("v_mfma"): push("M")
        elif op.startswith("s_cbranch") and j != e: push("B")
    if cur: groups.append(cur)
    res.update(unrolled=full, loop_ins=e - s + 1, loop_mfma=sum(ins[j][1].startswith("v_mfma") for j in range(s, e + 1)), loop_load4=sum(ins[j][1] == "global_load_dwordx4" for j in range(s, e + 1)),
               load_groups=groups if not full else dict(sorted(((g, groups.count(g)) for g in set(groups)))), waits_between_loads=between,
               stream=" ".join(t if n == 1 and t[0] == "W" else f"{t}{n}" for t, n in toks)[:600])
    return res

for p in sys.argv[1:]:
    print(json.dumps(summary(p)), flush=True)
