# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
# Stage 2: retrain the MTP head (fc + 1 decoder layer + norms) against the NVFP4 model's own hidden states and predictions.
# Main model, embeddings and lm_head stay frozen. Loss = KL(main model next-token dist || MTP dist) for draft steps 1-3.
import json, os, sys, time, math, random, shutil, subprocess, traceback, torch, torch.nn as nn, torch.nn.functional as F
from safetensors.torch import load_file, save_file
MODEL="$HK_DATA/models/Qwen3.8-27B-NVFP4"; DATA="$HK_DATA/models/mtp-data/gen.jsonl"
MAXLEN=4096; LR=float(os.environ.get("LR","5e-5")); EPOCHS=int(os.environ.get("EPOCHS","1")); STEP_W=[1.0,0.7,0.5]; TOPK=64
def log(*a): print(time.strftime("%d-%m-%Y %H:%M:%S"),*a,flush=True)
ok=False
try:
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, CompressedTensorsConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5RMSNorm
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
            quantization_config=CompressedTensorsConfig(run_compressed=False), attn_implementation="sdpa")
    model.eval().requires_grad_(False)
    lm=model.model.language_model; tcfg=model.config.text_config; H=tcfg.hidden_size
    log("main model loaded", round(torch.cuda.memory_allocated()/1e9,1),"GB")
    full_idx=tcfg.layer_types.index("full_attention")
    class MTP(nn.Module):
        def __init__(s):
            super().__init__()
            s.fc=nn.Linear(2*H,H,bias=False)
            s.pre_fc_norm_embedding=Qwen3_5RMSNorm(H,eps=tcfg.rms_norm_eps); s.pre_fc_norm_hidden=Qwen3_5RMSNorm(H,eps=tcfg.rms_norm_eps)
            s.layers=nn.ModuleList([Qwen3_5DecoderLayer(tcfg, full_idx)]); s.norm=Qwen3_5RMSNorm(H,eps=tcfg.rms_norm_eps)
        def forward(s, emb, h, pe):
            x=s.fc(torch.cat([s.pre_fc_norm_embedding(emb), s.pre_fc_norm_hidden(h)],-1))
            x=s.layers[0](x, position_embeddings=pe, attention_mask=None)
            return s.norm(x)
    orig=load_file(f"{MODEL}/model_mtp.safetensors")
    mtp=MTP().to("cuda", torch.float32)
    missing,unexp=mtp.load_state_dict({k[4:]:v.float() for k,v in orig.items()}, strict=True), None
    mtp.train()
    lm_head=model.lm_head.weight; embed=lm.embed_tokens
    # ---- data
    rows=[json.loads(l) for l in open(DATA)]; random.Random(0).shuffle(rows)
    def text(r):
        th=f"<think>\n{r['reasoning'].strip()}\n</think>\n\n" if r["think"] else "<think>\n\n</think>\n\n"
        return f"<|im_start|>user\n{r['prompt']}<|im_end|>\n<|im_start|>assistant\n{th}{r['content'].strip()}<|im_end|>\n"
    seqs=[tok(text(r),add_special_tokens=False)["input_ids"][:MAXLEN] for r in rows]
    seqs=[s for s in seqs if len(s)>64]
    held=seqs[:120]; train=seqs[120:]
    # pack training sequences into ~MAXLEN chunks (document boundaries kept by packing whole docs only)
    packs=[]; cur=[]
    for s in train:
        if len(cur)+len(s)>MAXLEN and cur: packs.append(cur); cur=[]
        cur=cur+s
    if cur: packs.append(cur)
    log("docs",len(seqs),"train packs",len(packs),"train tokens",sum(map(len,packs)),"heldout docs",len(held))
    def main_pass(ids):
        x=torch.tensor([ids],device="cuda")
        with torch.no_grad():
            out=lm(input_ids=x, use_cache=False)
            h=out.last_hidden_state                    # post-norm, what vLLM hands to the MTP head
            pos=torch.arange(x.shape[1],device="cuda").view(1,1,-1).expand(3,1,-1)
            pe=lm.rotary_emb(h,pos)
            # teacher: top-k of main model next-token dist for every position (chunked)
            tv,ti=[],[]
            for c in range(0,h.shape[1],1024):
                lg=(h[0,c:c+1024].float()@lm_head.float().T); p=lg.softmax(-1); v,i=p.topk(TOPK,-1); tv.append(v); ti.append(i)
            tv=torch.cat(tv); ti=torch.cat(ti); tv=tv/tv.sum(-1,keepdim=True)
            e=embed(x)
        return x,h,pe,e,tv,ti
    def mtp_steps(x,h,pe,e,tv,ti,train=True):
        T=x.shape[1]; hid=h; losses=[]; accs=[]
        for k in range(3):
            n=T-2-k
            if n<8: break
            emb=e[:,k+1:k+1+n]; hin=hid[:,:n]
            pe_k=(pe[0][:,:,:n] if pe[0].dim()==4 else pe[0][:,:n], pe[1][:,:,:n] if pe[1].dim()==4 else pe[1][:,:n])
            out=mtp(emb.float(), hin.float(), (pe_k[0].float(),pe_k[1].float()))
            tgt_v=tv[k+1:k+1+n]; tgt_i=ti[k+1:k+1+n]
            L=0; A=0
            for c in range(0,n,1024):
                lg=out[0,c:c+1024].to(torch.bfloat16)@lm_head.T
                lsm=lg.float().log_softmax(-1)
                L=L - (tgt_v[c:c+1024]*lsm.gather(-1,tgt_i[c:c+1024])).sum()
                A=A + (lg.argmax(-1)==tgt_i[c:c+1024,0]).sum()
            losses.append(L/n); accs.append((A/n).item())
            hid=out   # next draft step consumes this step's (normed) output, as in vLLM
        return losses, accs
    def evaluate():
        mtp.eval(); tot=[0,0,0]; cnt=0
        with torch.no_grad():
            for s in held:
                _,a=mtp_steps(*main_pass(s),train=False)
                if len(a)==3: tot=[t+x for t,x in zip(tot,a)]; cnt+=1
        mtp.train(); acc=[round(t/cnt,4) for t in tot]
        # expected accepted tokens per step (greedy, chain): 1 + a1 + a1*a2 + a1*a2*a3 (approximation using per-step agreement)
        exp=1+acc[0]+acc[0]*acc[1]+acc[0]*acc[1]*acc[2]
        return acc, round(exp,3)
    base_acc,base_len=evaluate(); log("BASELINE (original Qwen MTP head on NVFP4) per-step agreement",base_acc,"est. mean acceptance length",base_len)
    opt=torch.optim.AdamW(mtp.parameters(), lr=LR, betas=(0.9,0.95), weight_decay=0.0)
    total=len(packs)*EPOCHS; warm=min(50,total//10); step=0; best=(base_len,None)
    sched=lambda s: min(1,(s+1)/max(1,warm)) * 0.5*(1+math.cos(math.pi*min(1,s/total)))
    t0=time.time()
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(packs)
        for pk in packs:
            for g in opt.param_groups: g["lr"]=LR*sched(step)
            losses,accs=mtp_steps(*main_pass(pk))
            loss=sum(w*l for w,l in zip(STEP_W,losses)); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mtp.parameters(),1.0); opt.step(); step+=1
            if step%25==0: log(f"step {step}/{total} loss {loss.item():.4f} acc {[round(a,3) for a in accs]} {round((time.time()-t0)/step,2)}s/step")
            if step%300==0 or step==total:
                acc,ml=evaluate(); log("EVAL step",step,"agreement",acc,"est. mean acceptance length",ml)
                if ml>best[0]: best=(ml,{k:v.detach().to("cpu",torch.bfloat16).clone() for k,v in mtp.state_dict().items()})
    if best[1] is not None:
        bk=f"{MODEL}/model_mtp.safetensors.orig-qwen"
        if not os.path.exists(bk): shutil.copy(f"{MODEL}/model_mtp.safetensors",bk)
        new={"mtp."+k:v.contiguous() for k,v in best[1].items()}
        assert set(new)==set(orig), (set(new)^set(orig))
        save_file(new,f"{MODEL}/model_mtp.safetensors",metadata={"format":"pt"})
        log("SAVED retrained MTP head: est. mean acceptance length",base_len,"->",best[0],"(original kept as model_mtp.safetensors.orig-qwen)")
    else: log("NO IMPROVEMENT - original MTP head kept")
    ok=True
except Exception:
    log("FAILED"); traceback.print_exc()
finally:
    try: del model
    except: pass
    torch.cuda.empty_cache()
    log("run finished")  # (site-specific: restart your vLLM service here)
