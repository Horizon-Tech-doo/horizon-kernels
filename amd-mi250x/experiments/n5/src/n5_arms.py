from n5lib import arm
ARMS = []
# d256: tile grid at 8k; bN64 is 65536 B
for bN in (16, 32, 40, 48, 56, 64):
    for fl in ("", "E", "ER", "ERD", "D"):
        ARMS += arm(["d256_8192"], 32, bN, 768, fl)
# d128: tile grid at 4k
for bN in (32, 64, 80, 96, 112):
    for fl in ("", "E", "ER", "ERD", "D"):
        ARMS += arm(["d128_4096"], 32, bN, 768, fl)
# mha: tile grid at 4k
for P, bN, th in ((128, 32, 512), (128, 48, 512), (128, 64, 512), (192, 32, 768), (192, 64, 768), (192, 96, 768), (64, 32, 256)):
    ARMS += arm(["mha_4096"], P, bN, th)
# N6: transposed V (e3 form so comp_b/mfma_b stay available), V read packed 1 or 2
for vp in (1, 2):
    for fl in ("ERT", "ERDT", "T"):
        for bN in (16, 32, 48):
            ARMS += arm(["d256_8192"], 32, bN, 768, fl, form="e3", vp=vp)
        for bN in (32, 64, 96):
            ARMS += arm(["d128_4096"], 32, bN, 768, fl, form="e3", vp=vp)
    for P, bN, th in ((128, 32, 512), (128, 64, 512), (192, 64, 768)):
        ARMS += arm(["mha_4096"], P, bN, th, "T", form="e3", vp=vp)
# clock stamps
ARMS += arm(["d256_8192"], 32, 16, 768, form="e3", mode="mfma_b") + arm(["d256_8192"], 32, 16, 768, form="e3", mode="full") + arm(["d256_8192"], 32, 48, 768, form="e3", mode="mfma_b") + arm(["d256_8192"], 32, 48, 768, "ER", form="e3", mode="full")
ARMS += arm(["d128_4096"], 32, 96, 768, form="e3", mode="mfma_b") + arm(["d128_4096"], 32, 96, 768, "ERD", form="e3", mode="full")
SEL = dict(
    land3=r"d256_8192_P32_b48_t768_land_fullER$|d128_4096_P32_b96_t768_land_fullERD$|mha_4096_P128_b32_t512_land_full$",
    t1=r"d256_8192_P32_b(16|32|48)_t768_e3_fullERTvp|d256_8192_P32_b(16|32|48)_t768_e3_fullERT$|d128_4096_P32_b(32|64|96)_t768_e3_fullERT|mha_4096_P(128|192)_b(32|64)_t(512|768)_e3_fullT",
)
SEL.update(
    land3l=r"d256_8192_P32_b48_t768_land_fullER$|d128_4096_P32_b96_t768_land_fullERD$|mha_4096_P128_b32_t512_land_full$",
    t2=r"d256_8192_P32_b(16|32|48)_t768_e3_fullERT$|d128_4096_P32_b(32|64|96)_t768_e3_fullERT$|mha_4096_P128_b(32|64)_t512_e3_fullT$",
    v2=r"d256_8192_P32_b(16|32|48)_t768_land_fullER$|d128_4096_P32_b(32|64|96)_t768_land_fullER$|mha_4096_P128_b(32|64)_t512_land_full$",
)
SEL.update(grid=r"_land_full")
SEL.update(z1=r"d256_8192_P32_b48_t768_e3_fullERT$|d256_8192_P32_b(16|48)_t768_land_fullER$|d128_4096_P32_b96_t768_land_fullER$|d128_4096_P32_b96_t768_e3_fullERT$")
ARMS += arm(["mha_4096"], 192, 96, 768, "T", form="e3") + arm(["mha_4096"], 64, 32, 256, "T", form="e3")
SEL.update(z3=r"d256_8192_P32_b(16|32|48)_t768_(land_full|e3_full)ER(D|T|DT)?$|d128_4096_P32_b(32|64|96)_t768_(land_full|e3_full)ER(D|T|DT)?$|mha_4096_P(128_b(32|64)_t512|192_b96_t768|64_b32_t256)_(land_full|e3_fullT)$",
           st=r"d256_8192_P32_b(16|48)_t768_e3_mfma_b$|d256_8192_P32_b16_t768_e3_full$|d256_8192_P32_b48_t768_e3_fullER$|d128_4096_P32_b96_t768_e3_mfma_b$|d128_4096_P32_b96_t768_e3_fullERD$")
SEL.update(fail=r"d256_8192_P32_b32_t768_e3_fullERT$|d128_4096_P32_b64_t768_e3_fullERT$|mha_4096_P192_b64_t768_e3_fullT$")
SEL.update(z4=r"d256_8192_P32_b(32|48)_t768_(land_full|e3_full)ER(T)?$|d128_4096_P32_b(64|96)_t768_(land_full|e3_full)ERD?(T)?$|mha_4096_P(128_b32_t512|192_b96_t768|64_b32_t256)_(land_full|e3_fullT)$")
SEL.update(z5=r"d256_8192_P32_b(32|48)_t768_land_fullER$|d128_4096_P32_b(64|96)_t768_land_fullERD?$|mha_4096_P(128_b(32|64)_t512|192_b96_t768|64_b32_t256)_land_full$")
SEL.update(z6=r"d256_8192_P32_b32_t768_land_full(|E|D|ERD)$|d256_8192_P32_b16_t768_land_fullER$|d128_4096_P32_b(32|80)_t768_land_fullERD?$|d128_4096_P32_b96_t768_land_full(|E|D)$")
# more waves in one block: more rows share one K/V tile (fewer load bytes per flop); needs Occupancy 4 (V <= 128)
for P, th in ((40, 960),):
    for bN in (32, 48, 64, 80, 96):
        for fl in ("ER", "ERD"):
            ARMS += arm(["d128_4096"], P, bN, th, fl)
for P, bN, th in ((256, 32, 1024), (256, 64, 1024), (256, 96, 1024), (224, 64, 896), (240, 48, 960), (160, 32, 640), (160, 64, 640)):
    ARMS += arm(["mha_4096"], P, bN, th)
SEL.update(w=r"d128_4096_P40_|mha_4096_P(256|224|240|160)_")
SEL.update(st2=r"d256_8192_P32_b32_t768_land_fullER$|d128_4096_P32_b96_t768_land_fullER$|mha_4096_P128_b32_t512_land_full$")
SEL.update(t4=r"d256_8192_P32_b(32|48)_t768_e3_fullERT$|d128_4096_P32_b(32|64|96)_t768_e3_fullERT$|mha_4096_P(128_b32_t512|192_b96_t768)_e3_fullT$")
ARMS += arm(["d256_8192"], 32, 32, 768, "ERW") + arm(["d128_4096"], 32, 96, 768, "ERW") + arm(["d128_4096"], 32, 96, 768, "ERDW") + arm(["mha_4096"], 192, 96, 768, "W") + arm(["mha_4096"], 128, 32, 512, "W") + arm(["d256_8192"], 32, 32, 768, "EW")
SEL.update(sw=r"land_full(ER|ERD|E)?W$")
SEL.update(sw0=r"land_full(ER|ERD|E)?W$|d256_8192_P32_b32_t768_land_fullER$|d128_4096_P32_b96_t768_land_fullERD?$|mha_4096_P192_b96_t768_land_full$|mha_4096_P128_b32_t512_land_full$")
_d256 = ["d256_1024", "d256_4096", "d256_16384"]
_d128 = ["d128_1024", "d128_8192", "d128_16384"]
for bN, fl in ((16, "ER"), (32, "ER"), (32, "ERD"), (48, "ER"), (48, "ERD")):
    ARMS += arm(_d256, 32, bN, 768, fl)
for bN, fl in ((32, "ERD"), (64, "ERD"), (80, "ERD"), (96, "ER"), (96, "ERD")):
    ARMS += arm(_d128, 32, bN, 768, fl)
SEL.update(len=r"(d256_(1024|4096|8192|16384)_P32_b(16|32|48)_t768_land_fullERD?$)|(d128_(1024|4096|8192|16384)_P32_b(32|64|80|96)_t768_land_fullERD?$)")
for bN, fl in ((32, "ERW"), (48, "ERW"), (32, "ERDW")):
    ARMS += arm(_d256, 32, bN, 768, fl)
ARMS += arm(["d256_8192"], 32, 48, 768, "ERW") + arm(["d256_8192"], 32, 32, 768, "ERDW")
for bN, fl in ((32, "ERDW"), (64, "ERDW"), (96, "ERW"), (96, "ERDW")):
    ARMS += arm(_d128, 32, bN, 768, fl)
ARMS += arm(["d128_4096"], 32, 32, 768, "ERDW") + arm(["d128_4096"], 32, 64, 768, "ERDW")
SEL["len"] = r"(d256_(1024|4096|8192|16384)_P32_b(16|32|48)_t768_land_fullERD?W?$)|(d128_(1024|4096|8192|16384)_P32_b(32|64|80|96)_t768_land_fullERD?W?$)"
SEL.update(ce=r"d256_8192_P32_b32_t768_land_fullERD?W$|d128_4096_P32_b96_t768_land_fullERD?W$|mha_4096_P192_b96_t768_land_fullW?$|d128_4096_P32_b64_t768_land_fullERDW$|d256_8192_P32_b48_t768_land_fullERW$",
           fin=r"d256_8192_P32_b32_t768_land_fullERW$|d128_4096_P32_b96_t768_land_fullERDW$|mha_4096_P192_b96_t768_land_full$")
for P, th in ((96, 384), (112, 448), (128, 512), (144, 576), (160, 640), (176, 704), (208, 832)):
    for bN in (16, 32, 48, 64):
        if not any(a["P"] == P and a["bN"] == bN and a["case"] == "mha_4096" and a["form"] == "land" and a["mode"] == "full" and not a["rev"] and not a["edge"] and not a["swap"] for a in ARMS):
            ARMS += arm(["mha_4096"], P, bN, th)
for P, th in ((24, 576), (16, 384)):
    for bN in (16, 32, 48, 64):
        ARMS += arm(["d128_4096"], P, bN, th, "ERDW") + arm(["d256_8192"], P, bN, th, "ERW")
SEL.update(wv=r"mha_4096_P(96|112|128|144|160|176|208)_b(16|32|48|64)_t\d+_land_full$|d128_4096_P(24|16)_|d256_8192_P(24|16)_")
SEL["fin"] = r"d256_8192_P32_b32_t768_land_fullERW$|d128_4096_P32_b96_t768_land_fullERDW$|mha_4096_P192_b96_t768_land_full$|mha_4096_P160_b32_t640_land_full$"
SEL["bad"] = r"mha_4096_P128_b48_t512_land_full$"
SEL.update(wv2=r"mha_4096_P(96|112|128|144|160|176|208|192)_b(16|32|48|64|96)_t\d+_land_full$|d128_4096_P(24|16)_b(16)_|d128_4096_P32_b(80|112)_t768_land_fullERDW$")
ARMS += arm(["d128_4096"], 32, 80, 768, "ERDW") + arm(["d128_4096"], 32, 112, 768, "ERDW")
for rows, th in ((128, 512), (144, 576), (160, 640), (176, 704), (208, 832)):
    for bN in (32, 48, 64, 96):
        ARMS += arm(["d128_4096"], 0, bN, th, "ERW", rows=rows) + arm(["d128_4096"], 0, bN, th, "ER", rows=rows)
    for bN in (16, 32):
        ARMS += arm(["d256_8192"], 0, bN, th, "ERW", rows=rows)
ARMS += arm(["d128_4096"], 0, 96, 768, "ERW", rows=192)  # same rows as P32, through the rows form
SEL.update(r=r"_r\d+_b")
for rows, th in ((64, 256), (80, 320), (96, 384), (112, 448), (128, 512)):
    for bN in (16, 32, 48):
        for fl in ("ERW", "ERDW"):
            if not any(a["rows"] == rows and a["bN"] == bN and a["edge"] and a["rev"] and a["swap"] and a["recip"] == ("D" in fl) and a["case"] == "d128_4096" for a in ARMS):
                ARMS += arm(["d128_4096"], 0, bN, th, fl, rows=rows)
SEL.update(r2=r"d128_4096_r(64|80|96|112|128)_b(16|32|48)_t\d+_land_fullERD?W$")
ARMS += arm(["d256_1024", "d256_4096", "d256_16384"], 0, 32, 768, "ERDW", rows=192)[:0]
ARMS += arm(["d128_1024", "d128_8192", "d128_16384"], 0, 32, 512, "ERDW", rows=128)
SEL.update(final=r"(d256_(1024|4096|8192|16384)_P32_b32_t768_land_fullERDW$)|(d128_(1024|4096|8192|16384)_r128_b32_t512_land_fullERDW$)|(d128_(1024|4096|8192|16384)_P32_b96_t768_land_fullERDW$)|mha_4096_P160_b32_t640_land_full$|mha_4096_P192_b96_t768_land_full$",
           plain=r"(d256_(1024|4096|8192|16384)_P32_b48_t768_land_fullER$)|(d128_(1024|4096|8192|16384)_P32_b96_t768_land_fullERD$)|mha_4096_P128_b32_t512_land_full$",
           fin2=r"d256_8192_P32_b32_t768_land_fullERDW$|d128_4096_r128_b32_t512_land_fullERDW$",
           st3=r"d256_8192_P32_b32_t768_land_fullERDW$|d128_4096_r128_b32_t512_land_fullERDW$|d128_4096_P32_b96_t768_land_fullERDW$|mha_4096_P160_b32_t640_land_full$|mha_4096_P192_b96_t768_land_full$")
for c_, P_, b_, th_, fl_, r_ in (("d128_4096", 0, 32, 512, "ERDW", 128), ("d128_4096", 32, 96, 768, "ERDW", 0), ("d256_8192", 32, 32, 768, "ERDW", 0), ("d256_8192", 32, 48, 768, "ERDW", 0), ("mha_4096", 160, 32, 640, "", 0), ("mha_4096", 192, 96, 768, "", 0)):
    ARMS += arm([c_], P_, b_, th_, fl_ + "T", form="vt", rows=r_)
SEL.update(vt=r"_vt_full")
SEL.update(rec=r"d256_8192_P32_b32_t768_land_fullERDW$|d256_8192_P32_b48_t768_land_fullERW$|d128_4096_r128_b32_t512_land_fullERDW$|d128_4096_P32_b96_t768_land_fullERDW$|mha_4096_P160_b32_t640_land_full$|mha_4096_P192_b96_t768_land_full$")
# Round 2, lever 2: reciprocal on MHA P160 under OYBD; bN80 has 640 load slots = threads.
ARMS += arm(["mha_4096"], 160, 32, 640, "D") + arm(["mha_4096"], 160, 32, 640, "DW") + arm(["mha_4096"], 160, 80, 640)
SEL.update(r2m=r"mha_4096_P160_b(32|80)_t640_land_full(D|DW)?$")
