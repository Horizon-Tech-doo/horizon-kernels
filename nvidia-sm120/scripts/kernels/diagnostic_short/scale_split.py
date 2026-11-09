# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""One loaded reference, separate qkv/z interventions, repeat control, bounded 16k document."""
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(os.environ.get("HK_WORK", ".")) / 'trainctx'))
import torch
import trainctx as T
import torch.nn.attention as tna

F = Path(os.environ.get("HK_WORK", ".")) / 'fp8fix'
OUT = Path(os.environ.get("HK_WORK", ".")) / 'diagnostic-short'
pr = json.loads((Path(os.environ.get("HK_WORK", ".")) / 'trainctx/probe.json').read_text())
T.Impl.sdpa_backends = [getattr(tna.SDPBackend, pr['sdpa_choice'])]
T.Impl.conv = 'cuda' if pr.get('causal_conv1d', {}).get('ok') else 'torch'
T._TRITON['ok'] = bool(pr.get('triton_dequant', {}).get('ok'))
model, _ = T.build_model(T.MODEL, 'r2', 'cuda', shard_scales='exact')
model.eval()
docs = json.loads((F / 'invest_docs_q38.json').read_text())
ids = torch.tensor(docs['prose_chat_16384']['ids'], device='cuda')
original = {}
for i, layer in enumerate(model.layers):
    if layer.kind == 'linear_attention':
        la = layer.linear_attn
        original[i] = (la.in_proj_qkv.gscale.clone(), la.in_proj_z.gscale.clone())

report = {'document': 'prose_chat_16384', 'arms': {}, 'activation_samples': {},
          'note': 'Reference BF16 computation; no static FP8 activation quantization. Single document, no deployment.'}
current = ''
hooks = []
# Tiny, fixed sample of hidden channels/tokens per layer. Not full activation distributions.
for i, layer in enumerate(model.layers):
    def hook(mod, args, out, idx=i):
        x = out.detach()[0, ::256, ::64].float()
        report['activation_samples'][current][str(idx)] = {
            'rms': x.square().mean().sqrt().item(), 'absmax': x.abs().max().item(),
            'finite': bool(torch.isfinite(x).all().item())}
    hooks.append(layer.register_forward_hook(hook))

with torch.no_grad():
    for mode in ('exact', 'qkv_shared', 'z_shared', 'both_shared', 'exact_repeat'):
        current = mode
        report['activation_samples'][mode] = {}
        for i, (q, z) in original.items():
            la = model.layers[i].linear_attn
            mx = torch.maximum(q, z)
            la.in_proj_qkv.gscale = mx if mode in ('qkv_shared', 'both_shared') else q
            la.in_proj_z.gscale = mx if mode in ('z_shared', 'both_shared') else z
        t0 = time.time()
        h = model.hidden(ids)[0]
        scores = []
        for start in range(0, len(ids)-1, 512):
            end = min(start+512, len(ids)-1)
            lp = torch.log_softmax(model.logits(h[start:end]).float(), -1)
            scores.extend(lp.gather(1, ids[start+1:end+1, None]).squeeze(1).cpu().tolist())
            del lp
        del h
        assert len(scores) == 16383
        nll = lambda a: -sum(a)/len(a)
        arm = {'nll': nll(scores), 'first8k': nll(scores[:7999]), 'after8k': nll(scores[7999:]),
               'seconds': time.time()-t0, 'lp': scores}
        report['arms'][mode] = arm
        print(mode, {k:v for k,v in arm.items() if k != 'lp'}, flush=True)
        tmp = OUT / 'scale_split.json.tmp'
        tmp.write_text(json.dumps(report))
        os.replace(tmp, OUT / 'scale_split.json')
        torch.cuda.empty_cache()
a, b = report['arms']['exact']['lp'], report['arms']['exact_repeat']['lp']
report['repeat_max_abs_logprob'] = max(abs(x-y) for x,y in zip(a,b))
(OUT / 'scale_split.json').write_text(json.dumps(report))
print('REPEAT_MAX_ABS_LOGPROB', report['repeat_max_abs_logprob'], flush=True)
