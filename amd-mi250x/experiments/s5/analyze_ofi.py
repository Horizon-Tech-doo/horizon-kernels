"""Medians per arm per job for the OFI serving jobs. Run from evidence/S5: python3 analyze_ofi.py. Reads raw/round5-ofi/<job>/*.out."""
import glob, json, re, statistics as st
for f in sorted(glob.glob('raw/round5-ofi/*/vllm-mn-o-*.out')) + sorted(glob.glob('raw/round5-skeptic/vllm-mn-w-23525497.out')) + sorted(glob.glob('raw/logs/vllm-mn-2351[68]*.out')):
    job = re.search(r'(\d{8})', f).group(1); R = {}; arm = None; info = {}
    for l in open(f, errors='replace'):
        m = re.match(r'## (\w+) (server up after (\d+)|client rc=(\d+)|server NOT UP)', l)
        if m:
            arm = m.group(1); info.setdefault(arm, {})
            if m.group(3): info[arm]['up_s'] = int(m.group(3))
            if m.group(4) is not None: info[arm]['client_rc'] = int(m.group(4))
            if 'NOT UP' in l: info[arm]['not_up'] = True
        if l.startswith('THINK_ON') and arm: info[arm]['think'] = int(re.search(r"'completion_tokens': (\d+)", l).group(1))
        if l.startswith('LONG_PROMPT') and arm: info[arm]['long_s'] = float(l.rsplit('sec', 1)[1]); info[arm]['long_answer'] = re.search(r"\} '(\d+)' sec", l).group(1)
        if l.startswith('{"arm"'):
            o = json.loads(l); k = (o['arm'], 'bs%d' % o['bs'] if o['kind'] == 'decode' else 'p%d' % o['prompt_tokens'])
            R.setdefault(k, []).append(o['tok_per_s'])
            if o['kind'] == 'decode' and o['bs'] == 1: info.setdefault(o['arm'], {}).setdefault('b1', set()).add(o['text_md5'])
    print('==', job)
    for a in info:
        print(a, {k: v for k, v in info[a].items()})
        for key in ('bs1', 'bs8', 'bs32', 'p7211', 'p28812'):
            v = R.get((a, key))
            if v: print('  ', key, 'n', len(v), 'med', st.median(v), v)
