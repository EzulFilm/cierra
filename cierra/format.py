"""Canonical text for a checked program: one statement per line, parameters in table order, defaults left out,
numbers in shortest round-trip form. Two programs that mean the same thing format to the same text, so a text diff
of canonical forms is a meaningful diff, and the content hash identifies a version."""

import hashlib

from . import ast as A
from .ops import MASKS, OPS, OUTPUTS


def fnum(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    r = repr(float(v))
    if r.endswith(".0"):
        r = r[:-2]
    if r == "-0":
        r = "0"
    return r


def fval(v):
    if isinstance(v, A.Keys):
        return f"keys({v.interp}, " + ", ".join(f"{f}: {fval(x)}" for f, x in v.keys) + ")"
    if isinstance(v, A.Range):
        return f"{fnum(v.lo)}..{fnum(v.hi)}"
    if isinstance(v, A.Ident):
        return v.name
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fval(x) for x in v) + "]"
    return fnum(v)


def _same(a, b):
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def fcall(call, spec):
    args = []
    for key, p in spec.params.items():
        if key not in call.args:
            continue
        v = call.args[key]
        if v is None or (not p.required and _same(v, p.default)):
            continue
        args.append(f"{key}: {fval(v if p.kind != 'enum' else A.Ident(v))}")
    return f"{call.name}(" + ", ".join(args) + ")"


def fmask(m, top=True):
    if isinstance(m, A.MaskCall):
        return fcall(m.call, MASKS[m.call.name])
    if isinstance(m, A.MaskNot):
        return "not " + fmask(m.m, False)
    if isinstance(m, A.MaskRef):
        extra = "".join(f", {k}: {fval(v)}" for k, v, d in (("gain", m.gain, 1.0), ("offset", m.offset, 0.0))
                        if not (isinstance(v, (int, float)) and v == d))
        return f"key({m.name}{extra})"
    s = f"{fmask(m.a, False)} {m.op} {fmask(m.b, False)}"
    return s if top else f"({s})"


def fgraph(g, top=True):
    if isinstance(g, A.Ref):
        return g.name
    if isinstance(g, A.Pass):
        return "pass"
    if isinstance(g, A.Serial):
        s = " -> ".join(fgraph(x, False) for x in g.items)
        return s if top else f"({s})"
    if isinstance(g, A.Parallel):
        return "parallel(" + ", ".join(fgraph(x) for x in g.branches) + ")"
    if isinstance(g, A.Layer):
        extra = "" if g.mode == "normal" else f", mode: {g.mode}"
        extra += "" if g.opacity == 1.0 else f", opacity: {fval(g.opacity)}"
        return f"layer({fgraph(g.bottom)}, {fgraph(g.top)}{extra})"
    raise AssertionError(g)


def format_program(prog):
    """Checked Program -> canonical source text."""
    lines = [f"cierra {prog.version}", ""]
    if prog.name:
        lines.append(f"name {fval(prog.name)}")
    if prog.about:
        lines.append(f"about {fval(prog.about)}")
    lines.append(f"input {prog.input}")
    lines.append(f"output {fcall(prog.output, OUTPUTS[prog.output.name])}")
    if prog.reference_width != 1920:
        lines.append(f"reference_width {prog.reference_width}")
    if prog.fps != 24.0:
        lines.append(f"fps {fnum(prog.fps)}")
    if prog.seed:
        lines.append(f"seed {prog.seed}")
    if prog.requires:
        lines.append("requires " + ", ".join(prog.requires))
    lines.append("")
    for kid, k in prog.keys.items():
        lines.append(f"key {kid} = {fmask(k.mask)}")
    for nid, n in prog.nodes.items():
        lines.append(f"node {nid} = {fcall(n.call, OPS[n.call.name])}")
        if n.label:
            lines.append(f"  label {fval(n.label)}")
        if n.mask is not None:
            lines.append(f"  mask {fmask(n.mask)}")
        if not (isinstance(n.mix, float) and n.mix == 1.0):
            lines.append(f"  mix {fval(n.mix)}")
        if not n.enabled:
            lines.append("  off")
    lines.append("")
    if prog.photo_graph is not None:
        lines.append(f"photo_graph {fgraph(prog.photo_graph)}")
    if prog.graph is not None:  # omitted graph = identity
        lines.append(f"graph {fgraph(prog.graph)}")
    return "\n".join(lines) + "\n"


def program_hash(prog):
    return hashlib.sha256(format_program(prog).encode("utf-8")).hexdigest()[:12]
