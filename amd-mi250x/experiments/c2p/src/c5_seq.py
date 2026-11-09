"""Compress assembly to its memory, barrier, MFMA and branch sequence, with ALU runs as counts. c5_seq.py file.s"""
import re, sys
out = []; alu = 0; regs=0
for l in open(sys.argv[1]):
    l = l.strip()
    if l.startswith(".LBB") and ":" in l:
        if alu: out.append(str(alu)); alu = 0
        out.append("\n" + l.split(":")[0] + ("^" if "Loop Header" in l else "") + ":")
        continue
    if not l or l.startswith((".", ";")): continue
    m = l.split()[0]
    k = {"global_load_dwordx4": "GL4", "global_load_dwordx2": "GL2", "global_store_dwordx2": "GS2", "s_barrier": "BAR", "ds_write_b128": "DW16", "ds_read_b128": "DR16", "v_mfma_f32_16x16x16f16": "M", "ds_read_u16": "dru16", "ds_read_b64": "DR8", "ds_write_b64": "DW8", "s_nop": "nop", "v_exp_f32_e32": "exp", "ds_write2_b32": "DW2", "ds_read_u16_d16": "dru16", "ds_read_u16_d16_hi": "dru16h"}.get(m)
    if m.startswith("s_cbranch") or m == "s_branch": k = m.replace("s_cbranch_", "br_")
    if m == "s_waitcnt": k = "w"
    if k:
        if alu: out.append(str(alu)); alu = 0
        out.append(k)
    else:
        alu += 1
s = " ".join(out)
s = re.sub(r"\b(\w+)( \1\b)+", lambda m: f"{m.group(1)}x{len(m.group(0).split())}", s)
print(s)
