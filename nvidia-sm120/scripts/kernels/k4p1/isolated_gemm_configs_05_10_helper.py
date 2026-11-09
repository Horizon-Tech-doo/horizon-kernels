# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Frozen isolation grid: compare launch geometry, not a deployment policy."""
def configs(rows,n):
    tiles=[(128,4,128),(128,4,256)]
    if n==34816:
        tiles.append((256,4,256))
        if rows==4:tiles.append((256,6,256))
    return [dict(width=w,stages=s,threads=t,grid=g) for w,s,t in tiles for g in (188,376)
            if not (g==376 and w==256 and (rows==32 or s==6))]
def label(c):return 'n{width}-s{stages}-t{threads}-g{grid}'.format(**c)
