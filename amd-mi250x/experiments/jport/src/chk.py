"""CPU. (1) d128_tile and mha_tile of the landed and candidate modules over L 1-20000, B 1-2, clang 18 and 22: equal except where the candidate returns form J. (2) _j_source(L) minus the RegVec block equals fork-main's generated source kept in D128/bytes. Usage: chk.py OLDDIR NEWDIR D128BYTES"""
import importlib.util, json, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
def load(name, d):
    sp = importlib.util.spec_from_file_location(name, os.path.join(d, "gqa_fa.py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
old, new, by = load("g_old", sys.argv[1]), load("g_new", sys.argv[2]), sys.argv[3]
res = {}
for c in (18, 22):
    old._CLANG[:] = [c]; new._CLANG[:] = [c]
    diff, j, same = [], [], 0
    for B in (1, 2):
        for L in range(1, 20001):
            a, b = old.d128_tile(L, B), new.d128_tile(L, B)
            if b.get("form") == "J":
                j.append((B, L, a))
            elif a == b:
                same += 1
            else:
                diff.append((B, L))
            for Hkv in (8, 4):
                if old.mha_tile(L, B, Hkv) != new.mha_tile(L, B, Hkv):
                    diff.append(("mha", B, L, Hkv))
    olds = sorted({json.dumps(x[2], sort_keys=True) for x in j})
    res[c] = dict(same=same, diff=diff[:10], ndiff=len(diff), nJ=len(j), J_B=sorted({x[0] for x in j}), J_L=[min((x[1] for x in j), default=None), max((x[1] for x in j), default=None)], J_mod64=all(x[1] % 64 == 0 for x in j), old_tiles_at_J=olds)
print(json.dumps(res))
start = new._J_SRC.index("\nnamespace tl {"); end = new._J_SRC.index('extern "C"')
for L in (2816, 3072, 3584, 4096, 8192, 16384):
    s = new._j_source(L); s = s[:start] + s[end:]
    print(json.dumps(dict(L=L, equal_fork_source=s == open(f"{by}/d128_{L}_r128_b64_t512_JDRF.cc").read())))
for L in (3000, 2817):
    try:
        new._j_source(L); print(json.dumps(dict(L=L, raised=False)))
    except ValueError:
        print(json.dumps(dict(L=L, raised=True)))
