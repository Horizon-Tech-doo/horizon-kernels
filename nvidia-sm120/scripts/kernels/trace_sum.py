# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
import gzip,json,glob,collections,re
f=sorted(glob.glob("$HK_WORK/prof/rank0*.json.gz"))[-1]
ev=json.load(gzip.open(f))["traceEvents"]
k=[e for e in ev if e.get("cat")=="kernel"]
k.sort(key=lambda e:e["ts"])
t0=k[0]["ts"]; t1=k[-1]["ts"]+k[-1]["dur"]
busy=sum(e["dur"] for e in k)
# merge overlapping intervals for true busy
iv=[];
for e in k:
    s,en=e["ts"],e["ts"]+e["dur"]
    if iv and s<=iv[-1][1]: iv[-1][1]=max(iv[-1][1],en)
    else: iv.append([s,en])
ubusy=sum(b-a for a,b in iv)
print(f"window {(t1-t0)/1e3:.1f} ms, GPU busy {ubusy/1e3:.1f} ms ({100*ubusy/(t1-t0):.1f}%), kernels {len(k)}")
def cat(n):
    n2=n.lower()
    for key,pat in [("marlin_gemm","marlin"),("lm_head/bf16 gemm","gemm|gemv|cutlass|sm90|sm100|sm120|ampere|nvjet|cublas"),("gdn/fla (linear attn)","fla|gated_delta|chunk_|fused_recurrent|causal_conv|conv1d|l2norm"),("attention","flash|attn|attention|paged|reshape_and_cache|mla"),("norm","rms|norm"),("act/elementwise","silu|act_and_mul|elementwise|vectorized|unrolled|copy|fill|cat|index|gather|scatter"),("sampling/spec","argmax|sample|reject|topk|softmax|reduce")]:
        if re.search(pat,n2): return key
    return "other"
by=collections.Counter(); cnt=collections.Counter(); names=collections.Counter()
for e in k:
    c=cat(e["name"]); by[c]+=e["dur"]; cnt[c]+=1; names[e["name"][:110]]+=e["dur"]
for c,v in by.most_common(): print(f"{c:28s} {v/1e3:8.1f} ms {100*v/ubusy:5.1f}%  n={cnt[c]}")
print("--- top kernels")
for n,v in names.most_common(25): print(f"{v/1e3:8.1f} ms  {n}")
# gaps
gaps=[b[0]-a[1] for a,b in zip(iv,iv[1:])]
big=[g for g in gaps if g>50]
print(f"idle gaps: total {sum(gaps)/1e3:.1f} ms, gaps>50us: {len(big)} totalling {sum(big)/1e3:.1f} ms")
