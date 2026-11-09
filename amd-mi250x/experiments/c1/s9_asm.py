# Assembly-level test of the s_nop effect seen in s8: add or remove s_nop 0 between consecutive global loads, nothing else.
import re, sys
real = lambda l: l.strip() and not re.match(r'\s*(\.loc|;|\.file|\.Ltmp|\.cfi)', l)
isload = lambda l: re.match(r'\s*global_load', l)
isnop = lambda l: re.match(r'\s*s_nop 0\b', l)
def addnops(lines):
    out = []
    for i, l in enumerate(lines):
        out.append(l)
        if isload(l):
            nxt = [x for x in lines[i + 1:i + 12] if real(x)][:3]
            for x in nxt:
                if isload(x):
                    if not isnop(nxt[0]):
                        out.append('\ts_nop 0\n')
                    break
                if re.match(r'\s*(s_waitcnt|s_cbranch|s_branch|s_barrier|ds_)', x) or x.strip().endswith(':'):
                    break
    return out
def delnops(lines):
    out = []
    R = [i for i, l in enumerate(lines) if real(l)]
    pos = {i: k for k, i in enumerate(R)}
    for i, l in enumerate(lines):
        if isnop(l) and i in pos:
            k = pos[i]
            if k > 0 and isload(lines[R[k - 1]]) and k + 1 < len(R) and isload(lines[R[k + 1]]):
                continue
        out.append(l)
    return out
def between(lines):
    # indices of s_nop 0 lines whose neighbouring real instructions are both global loads
    R = [i for i, l in enumerate(lines) if real(l)]
    pos = {i: k for k, i in enumerate(R)}
    return [i for i, l in enumerate(lines) if isnop(l) and i in pos and 0 < pos[i] < len(R) - 1 and isload(lines[R[pos[i] - 1]]) and isload(lines[R[pos[i] + 1]])]
def repnops(lines, n):
    # same bytes, longer gap: an existing between-load s_nop 0 becomes s_nop n
    idx = set(between(lines))
    return [f'\ts_nop {n}\n' if i in idx else l for i, l in enumerate(lines)]
def fillnops(lines, n):
    # a gap of s_nop n after every global load that has another load close behind it, keeping existing nops' place
    out = addnops(lines)
    return repnops(out, n) if n else out
def ctrlnops(lines):
    # alignment control: same number of nop bytes, placed before the first load of each run, none between loads
    idx = set(between(lines))
    R = [i for i, l in enumerate(lines) if real(l)]
    pos = {i: k for k, i in enumerate(R)}
    first = {}
    for i in sorted(idx):
        k = pos[i]
        while k >= 2 and isload(lines[R[k - 1]]) and R[k - 2] in idx:
            k -= 2
        first[R[k - 1]] = first.get(R[k - 1], 0) + 1
    out = []
    for i, l in enumerate(lines):
        if i in idx:
            continue
        if i in first:
            out.extend(['\ts_nop 0\n'] * first[i])
        out.append(l)
    return out
f, mode, o = sys.argv[1:4]
L = open(f).readlines()
m = re.match(r'(rep|fill)(\d+)$', mode)
if m:
    L2 = (repnops if m.group(1) == 'rep' else fillnops)(L, int(m.group(2)))
elif mode == 'ctrl':
    L2 = ctrlnops(L)
else:
    L2 = addnops(L) if mode == 'add' else delnops(L) if mode == 'del' else L
open(o, 'w').writelines(L2)
anynop = lambda l: re.match(r'\s*s_nop \d+', l)
print(o, 'mode', mode, 'nops', sum(1 for l in L if anynop(l)), '->', sum(1 for l in L2 if anynop(l)), 'between', len(between(L)), '->', sum(1 for i in range(len(L2)) if anynop(L2[i]) and i and isload([x for x in L2[:i] if real(x)][-1]) and any(real(x) for x in L2[i+1:]) and isload([x for x in L2[i+1:] if real(x)][0])))
