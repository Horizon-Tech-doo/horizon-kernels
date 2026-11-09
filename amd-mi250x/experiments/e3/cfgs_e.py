from cfgs_cases import CASES  # noqa: F401
from cfgs_b import row
V = [dict(), dict(edge=True), dict(rev=True), dict(edge=True, rev=True), dict(recip=True), dict(edge=True, rev=True, recip=True)]
CONFIGS = []
for bN in (16, 32, 48):
    for kw in V:
        CONFIGS.append(row("d256", 32, bN, 768, **kw))
for bN in (32, 64, 96):
    for kw in V:
        CONFIGS.append(row("d128", 32, bN, 768, **kw))
for P, bN, th in ((128, 32, 512), (128, 64, 512), (128, 96, 512), (192, 32, 768), (192, 64, 768), (192, 96, 768)):
    CONFIGS.append(row("mha", P, bN, th))
    CONFIGS.append(row("mha", P, bN, th, recip=True))
