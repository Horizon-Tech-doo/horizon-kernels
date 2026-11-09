# From a JIT cache dir: pull each code object out of executable.so, disassemble it, count gaps between global loads; also recompile device_kernel.cu -S with the flag.
import glob, json, os, re, struct, subprocess, sys
BIN = '/opt/rocm-6.2.4/llvm/bin/'
d = sys.argv[1]; tmp = sys.argv[2]; os.makedirs(tmp, exist_ok=True)
for kd in sorted(glob.glob(d + '/0.1.15/*/kernels/*')):
    src = open(kd + '/device_kernel.cu').read()
    m = re.search(r"__launch_bounds__\([^)]*\)(?:\s+__attribute__\(\(\w+\(\d+\)\)\))?\s+(\w+)\(", src)
    name = m.group(1) if m else '?'
    so = open(kd + '/executable.so', 'rb').read()
    # the embedded amdgcn ELF: machine 0xE0
    elf = None; i = 0
    while True:
        i = so.find(b'\x7fELF', i + 1)
        if i < 0: break
        if struct.unpack('<H', so[i + 18:i + 20])[0] == 0xE0:
            shoff, = struct.unpack('<Q', so[i + 40:i + 48]); shes, shn = struct.unpack('<HH', so[i + 58:i + 62])
            elf = so[i:i + shoff + shes * shn]; break
    rec = dict(key=os.path.basename(kd)[:12], kernel=name, bounds=m.group(0)[:30] if m else None, resource=json.load(open(kd + '/resource_usage.json')), rr='a_r_0' in src)
    if elf is None:
        rec['elf'] = 'not found in executable.so'
    else:
        p = f'{tmp}/{name}_{rec["key"]}.elf'; open(p, 'wb').write(elf)
        dis = subprocess.run([BIN + 'llvm-objdump', '-d', '--mcpu=gfx90a', p], capture_output=True, text=True).stdout.splitlines()
        ins = [re.sub(r'\s*//.*', '', l).strip() for l in dis if '//' in l and not l.strip().startswith('//')]
        while ins and ins[-1].startswith('s_code_end'): ins.pop()
        last = max((i for i, l in enumerate(ins) if l.startswith('s_endpgm')), default=len(ins))
        ins = ins[:last + 1]
        rec['elf_bytes'] = len(elf)
        rec['global_loads'] = sum(l.startswith('global_load') for l in ins)
        rec['gaps_between_loads'] = sum(ins[i].startswith('s_nop') and ins[i - 1].startswith('global_load') and ins[i + 1].startswith('global_load') for i in range(1, len(ins) - 1))
        rec['s_nop_types'] = sorted({l for l in ins if l.startswith('s_nop')})
    print(json.dumps(rec, default=str), flush=True)
