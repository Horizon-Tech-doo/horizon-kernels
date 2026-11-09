#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Horizon Tech d.o.o.
# SPDX-License-Identifier: Apache-2.0
"""Static check: every vllm symbol the Horizon plugins use must exist in a vllm tree. No imports (no CUDA).
usage: symcheck.py VLLM_SITE_PACKAGES plugin.py [...]"""
import ast, os, sys

SP = sys.argv[1]
_cache = {}

def modfile(mod):
    p = os.path.join(SP, *mod.split("."))
    if os.path.isfile(p + ".py"): return p + ".py"
    if os.path.isfile(os.path.join(p, "__init__.py")): return os.path.join(p, "__init__.py")
    return None

def names_of(mod):
    """top-level names of module + {class: set(member names)}; None if module missing"""
    if mod in _cache: return _cache[mod]
    f = modfile(mod)
    if not f:
        _cache[mod] = None; return None
    t = ast.parse(open(f, encoding="utf8").read())
    top, classes = set(), {}
    def add_target(n):
        if isinstance(n, ast.Name): top.add(n.id)
        elif isinstance(n, (ast.Tuple, ast.List)): [add_target(e) for e in n.elts]
    def walk(body):
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)): top.add(n.name)
            elif isinstance(n, ast.ClassDef):
                top.add(n.name); mem = set()
                for m in n.body:
                    if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)): mem.add(m.name)
                    elif isinstance(m, ast.Assign): [add_t(mem, x) for x in m.targets]
                    elif isinstance(m, ast.AnnAssign) and isinstance(m.target, ast.Name): mem.add(m.target.id)
                for sub in ast.walk(n):  # self.x = ... assignments
                    if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == "self" and isinstance(sub.ctx, ast.Store):
                        mem.add(sub.attr)
                classes[n.name] = (mem, [ast.unparse(b) for b in n.bases])
            elif isinstance(n, ast.Assign): [add_target(x) for x in n.targets]
            elif isinstance(n, ast.AnnAssign): add_target(n.target)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names: top.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.If, ast.Try)):
                walk(n.body); walk(getattr(n, "orelse", [])); [walk(h.body) for h in getattr(n, "handlers", [])]
    def add_t(s, n):
        if isinstance(n, ast.Name): s.add(n.id)
    walk(t.body)
    _cache[mod] = (top, classes); return _cache[mod]

def is_pkg_member(mod, name):
    return modfile(mod + "." + name) is not None

problems = []
for path in sys.argv[2:]:
    src = open(path, encoding="utf8").read()
    tree = ast.parse(src)
    alias = {}  # local name -> ("mod", modname) | ("name", modname, symbol)
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and (n.module == "vllm" or n.module.startswith("vllm.")):
            for a in n.names:
                loc = a.asname or a.name
                if is_pkg_member(n.module, a.name):
                    alias[loc] = ("mod", n.module + "." + a.name)
                else:
                    info = names_of(n.module)
                    if info is None:
                        problems.append((path, n.lineno, f"module missing: {n.module}"))
                    elif a.name not in info[0]:
                        problems.append((path, n.lineno, f"{n.module} has no '{a.name}'"))
                    else:
                        alias[loc] = ("name", n.module, a.name)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name == "vllm" or a.name.startswith("vllm."):
                    if modfile(a.name) is None: problems.append((path, n.lineno, f"module missing: {a.name}"))
                    if a.asname: alias[a.asname] = ("mod", a.name)
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias:
            kind = alias[n.value.id]
            if kind[0] == "mod":
                if is_pkg_member(kind[1], n.attr): continue
                info = names_of(kind[1])
                if info is None: problems.append((path, n.lineno, f"module missing: {kind[1]}"))
                elif n.attr not in info[0] and isinstance(n.ctx, ast.Load):
                    problems.append((path, n.lineno, f"{kind[1]}.{n.attr} missing"))
            else:
                info = names_of(kind[1])
                cls = info[1].get(kind[2]) if info else None
                if cls:
                    mem, bases = cls
                    if n.attr not in mem and isinstance(n.ctx, ast.Load):
                        problems.append((path, n.lineno, f"class {kind[2]}.{n.attr} not defined in class body (may be inherited; bases={bases})"))
        # alias.Class.attr
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and isinstance(n.value.value, ast.Name) and n.value.value.id in alias and alias[n.value.value.id][0] == "mod":
            info = names_of(alias[n.value.value.id][1])
            if info and n.value.attr in info[1]:
                mem, bases = info[1][n.value.attr]
                if n.attr not in mem and isinstance(n.ctx, ast.Load):
                    problems.append((path, n.lineno, f"class {alias[n.value.value.id][1]}.{n.value.attr}.{n.attr} not defined in class body (bases={bases})"))
seen = set()
for p, l, m in sorted(problems):
    if (p, m) in seen: continue
    seen.add((p, m)); print(f"{os.path.basename(p)}:{l}: {m}")
print(f"-- {len(seen)} distinct problems")
