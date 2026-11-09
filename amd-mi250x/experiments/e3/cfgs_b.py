from cfgs_cases import CASES  # noqa: F401


def row(case, P, bN, th, edge=False, recip=False, mode="full", vt=False, v_pack=1, rev=False):
    n = f"{case}_P{P}_b{bN}_t{th}_{mode}"
    for f, s in ((edge, "E"), (rev, "R"), (recip, "D"), (vt, "T")):
        if f:
            n += s
    if v_pack != 1:
        n += f"vp{v_pack}"
    return dict(name=n, case=case, P=P, bN=bN, th=th, edge=edge, recip=recip, mode=mode, vt=vt, v_pack=v_pack, rev=rev)


CONFIGS = []
for case, P, bN in (("d256", 32, 16), ("d128", 32, 32)):
    for m in ("full", "comp_b", "c_nrsc", "c_nred", "c_nexp", "c_nmsk", "mfma_b", "mfma1", "mfma2", "empty", "load", "l_gem"):
        CONFIGS.append(row(case, P, bN, 768, mode=m))
    CONFIGS.append(row(case, P, bN, 768, vt=True))
    CONFIGS.append(row(case, P, bN, 768, mode="comp_b", vt=True))
    CONFIGS.append(row(case, P, bN, 768, mode="mfma2", vt=True))

# real-kernel tiles with transposed V; base for pairing is in the same pass.
for bN in (16, 32, 48):
    for kw in (dict(), dict(vt=True), dict(vt=True, recip=True), dict(vt=True, edge=True, recip=True), dict(vt=True, edge=True, rev=True, recip=True), dict(mode="comp_b", vt=True)):
        if not (bN == 16 and kw in (dict(), dict(vt=True), dict(mode="comp_b", vt=True))):
            CONFIGS.append(row("d256", 32, bN, 768, **kw))
for bN in (32, 64, 96, 112):
    for kw in (dict(), dict(vt=True), dict(vt=True, recip=True), dict(vt=True, edge=True, recip=True), dict(vt=True, edge=True, rev=True, recip=True), dict(mode="comp_b", vt=True), dict(edge=True, rev=True, recip=True)):
        if not (bN == 32 and kw in (dict(), dict(vt=True), dict(mode="comp_b", vt=True))):
            CONFIGS.append(row("d128", 32, bN, 768, **kw))
for P, bN, th in ((128, 32, 512), (128, 64, 512), (128, 96, 512), (192, 96, 768), (192, 112, 768), (128, 112, 512), (256, 32, 1024), (256, 64, 1024)):
    for kw in (dict(), dict(vt=True), dict(vt=True, recip=True), dict(mode="comp_b", vt=True)):
        CONFIGS.append(row("mha", P, bN, th, **kw))
