"""The checker: rejects anything unknown, mistyped, out of range, unbounded or non-deterministic, and returns a
normalised program (defaults filled in, numbers as floats, vectors as tuples) that every engine and compiler can trust.

All problems are collected (not just the first), each with its line and node, so a revision can fix them in one go.
"""

import math

from . import ast as A
from .errors import CheckError
from .ops import BLEND_MODES, IDENTITY_DEFAULTS, INPUTS, INTERPS, LIMITS, MASKS, OPS, OUTPUTS
from .parser import parse
from .targets import blurred
from .walk import carried_keys, mask_calls, same

AUX = ("depth", "segments")
PHOTO_INPUTS = ("srgb_photo", "rec1886_photo", "linear_photo")  # display-referred inputs: photo_graph runs for these


class _Ctx:
    def __init__(self):
        self.problems, self.warnings = [], []

    def err(self, where, msg):
        self.problems.append(f"{where}: {msg}")


def _is_num(v):
    """A finite number. Integers of any size are fine here (the range check rejects them, without converting)."""
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and (not isinstance(v, float) or math.isfinite(v)))


def _g(v):
    """Number for a message: huge integers are shortened instead of overflowing a float conversion."""
    if isinstance(v, int) and abs(v) >= 10 ** 15:
        s = str(abs(v))
        return f"{'-' if v < 0 else ''}{s[:6]}... ({len(s)} digits)"
    return f"{v:g}"


def _num(ctx, where, v, lo, hi, keyable, integer=False):
    if isinstance(v, A.Keys):
        if not keyable:
            ctx.err(where, "keyframes are not allowed here")
            return None
        return _keys(ctx, where, v, lambda w, x: _num(ctx, w, x, lo, hi, False, integer))
    if not _is_num(v) or (integer and not isinstance(v, int)):
        ctx.err(where, f"expected {'an integer' if integer else 'a number'}, got {_show(v)}")
        return None
    if not lo <= v <= hi:
        ctx.err(where, f"{_g(v)} is outside [{lo:g}, {hi:g}]")
        return None
    return int(v) if integer else float(v)


def _keys(ctx, where, k, check_value):
    if k.interp not in INTERPS:
        ctx.err(where, f"keyframe interpolation must be one of {', '.join(INTERPS)}, got {k.interp!r}")
        return None
    if not 1 <= len(k.keys) <= LIMITS["keys"]:
        ctx.err(where, f"keys() needs 1..{LIMITS['keys']} keyframes, got {len(k.keys)}")
        return None
    out, prev = [], -1
    for f, v in k.keys:
        if not isinstance(f, int) or not 0 <= f <= LIMITS["max_frame"]:
            ctx.err(where, f"keyframe frame must be an integer in [0, {LIMITS['max_frame']}], got {_show(f)}")
            return None
        if f <= prev:
            ctx.err(where, f"keyframe frames must increase strictly ({f} after {prev})")
            return None
        prev = f
        cv = check_value(f"{where} @ frame {f}", v)
        if cv is None:
            return None
        out.append((f, cv))
    return A.Keys(k.interp, tuple(out))


def _show(v):
    if isinstance(v, A.Ident):
        return v.name
    if isinstance(v, A.Range):
        return f"{v.lo}..{v.hi}"
    if isinstance(v, A.Keys):
        return "keys(...)"
    return repr(v)


def _param(ctx, where, p, v):
    k = p.kind
    if k == "num":
        return _num(ctx, where, v, p.lo, p.hi, True)
    if k == "int":
        return _num(ctx, where, v, p.lo, p.hi, False, integer=True)
    if k == "bool":
        if not isinstance(v, bool):
            ctx.err(where, f"expected true or false, got {_show(v)}")
            return None
        return v
    if k == "enum":
        name = v.name if isinstance(v, A.Ident) else None
        if name not in p.choices:
            ctx.err(where, f"expected one of {', '.join(p.choices)}, got {_show(v)}")
            return None
        return name
    if k in ("rgb", "pair"):
        n = 3 if k == "rgb" else 2
        if not isinstance(v, list) or len(v) != n:
            ctx.err(where, f"expected [{', '.join('rgb'[:n] if k == 'rgb' else 'ab')}] ({n} numbers), got {_show(v)}")
            return None
        out = tuple(_num(ctx, f"{where}[{i}]", x, p.lo, p.hi, True) for i, x in enumerate(v))
        return None if any(x is None for x in out) else out
    if k == "range":
        if not isinstance(v, A.Range):
            ctx.err(where, f"expected a range lo..hi, got {_show(v)}")
            return None
        lo = _num(ctx, where + " (lo)", v.lo, p.lo, p.hi, False)
        hi = _num(ctx, where + " (hi)", v.hi, p.lo, p.hi, False)
        if lo is None or hi is None:
            return None
        if lo > hi and p.unit != "deg":
            ctx.err(where, f"range {lo:g}..{hi:g} is reversed (only hue ranges may wrap)")
            return None
        if lo == hi and p.unit == "deg":
            ctx.err(where, f"hue range {lo:g}..{hi:g} is empty (use 0..360 for every hue)")
            return None
        return A.Range(lo, hi)
    if k == "points":
        if not isinstance(v, list) or not p.min_len <= len(v) <= p.max_len:
            ctx.err(where, f"expected a list of {p.min_len}..{p.max_len} points, got {_show(v) if not isinstance(v, list) else str(len(v)) + ' points'}")
            return None
        rows = []
        for i, row in enumerate(v):
            ncol = len(p.cols)
            if not isinstance(row, list) or not ncol <= len(row) <= ncol + len(p.opt_cols):
                ctx.err(f"{where}[{i}]", f"a point is [{', '.join(c[2] for c in p.cols + p.opt_cols)}]"
                        + (" (last optional)" if p.opt_cols else ""))
                return None
            vals = []
            for j, x in enumerate(row):
                lo, hi, label = (p.cols + p.opt_cols)[j]
                cv = _num(ctx, f"{where}[{i}] {label}", x, lo, hi, False)
                if cv is None:
                    return None
                vals.append(cv)
            rows.append(tuple(vals))
        if p.sorted:
            for a, b in zip(rows, rows[1:]):
                if not b[0] > a[0]:
                    ctx.err(where, f"point x values must increase strictly ({b[0]:g} after {a[0]:g})")
                    return None
                if b[0] - a[0] < LIMITS["point_gap"] * (1 - 1e-9):  # 0.5001 - 0.5 is 9.99...e-5 in binary
                    ctx.err(where, f"points {a[0]:g} and {b[0]:g} are closer than {LIMITS['point_gap']:g} in x")
                    return None
        return tuple(rows)
    if k == "ints":
        if not isinstance(v, list) or not p.min_len <= len(v) <= p.max_len:
            ctx.err(where, f"expected a list of {p.min_len}..{p.max_len} integers")
            return None
        out = tuple(_num(ctx, f"{where}[{i}]", x, p.lo, p.hi, False, integer=True) for i, x in enumerate(v))
        return None if any(x is None for x in out) else out
    raise AssertionError(k)


def _call(ctx, where, call, spec):
    """Check a call's arguments against an OpSpec; return the full argument dict with defaults."""
    out = {}
    for key in call.args:
        if key not in spec.params:
            known = ", ".join(spec.params) or "no parameters"
            ctx.err(where, f"{spec.name}() has no parameter {key!r} (it takes: {known})")
    for key, p in spec.params.items():
        if key in call.args:
            out[key] = _param(ctx, f"{where} {key}", p, call.args[key])
        elif p.required:
            ctx.err(where, f"{spec.name}() needs {key!r} ({p.doc})")
        else:
            out[key] = p.default
    return A.Call(call.name, out, call.line)


def _mask(ctx, where, m, requires):
    if isinstance(m, A.MaskNot):
        inner = _mask(ctx, where, m.m, requires)
        return A.MaskNot(inner)
    if isinstance(m, A.MaskBin):
        return A.MaskBin(m.op, _mask(ctx, where, m.a, requires), _mask(ctx, where, m.b, requires))
    if isinstance(m, A.MaskRef):  # the name is resolved by _resolve_keys once every mask is checked
        m.gain = _num(ctx, f"{where} key({m.name}) gain", m.gain, 0.0, 4.0, True)
        m.offset = _num(ctx, f"{where} key({m.name}) offset", m.offset, -1.0, 1.0, True)
        return m
    spec = MASKS.get(m.call.name)
    if spec is None:
        ctx.err(where, f"unknown mask {m.call.name!r} (masks: {', '.join(MASKS)})")
        return m
    for a in spec.aux:
        if a not in requires:
            ctx.err(where, f"mask {spec.name}() needs the {a!r} plane: add `requires {a}`")
    c = _call(ctx, f"{where} mask {spec.name}()", m.call, spec)
    if spec.name == "qualifier" and all(c.args.get(k) is None for k in ("hue", "sat", "lum")):
        ctx.warnings.append(f"{where}: qualifier() with no hue, sat or lum selects everything")
    return A.MaskCall(c)


def _direct_refs(m):
    """key(...) calls written in one mask expression (not those inside the masks they name)."""
    return [x for x in _flat(m) if isinstance(x, A.MaskRef)]


def _resolve_keys(ctx, prog):
    """Point every key(name) at the checked mask of that node or key statement; a reference cycle, an unknown name or
    a node without a mask is an error. A node's key is carried from where that node ran; a key statement's mask is
    made again wherever it is used, so only those expand (their size and depth are bounded here). Every walk here is
    iterative and the chain depth is checked before each step, so no program can exhaust the stack."""
    owners = {**prog.nodes, **prog.keys}
    edges = {}
    for name, owner in owners.items():
        edges[name] = []
        for r in (_direct_refs(owner.mask) if owner.mask is not None else []):
            where = f"line {r.line or owner.line} {'key' if name in prog.keys else 'node'} {name}"
            src = owners.get(r.name)
            if src is None:
                ctx.err(where, f"key({r.name}): no node or key named {r.name!r}")
            elif src.mask is None:
                ctx.err(where, f"key({r.name}): node {r.name} has no mask")
            else:
                r.target, r.carried = src.mask, r.name in prog.nodes
                edges[name].append((r.name, where))
                if r.carried and not src.enabled:
                    ctx.warnings.append(f"{where}: key({r.name}) is the key of a node that is off (its key is "
                                        "still made)")
    state = {}  # 1 on the current path, 2 finished
    for start in owners:
        if start in state:
            continue
        path, todo = [start], [iter(edges[start])]
        state[start] = 1
        while todo:
            step = next(todo[-1], None)
            if step is None:
                state[path.pop()] = 2
                todo.pop()
                continue
            nxt, where = step
            if state.get(nxt) == 1:
                ctx.err(where, f"key({nxt}) makes a cycle: {' -> '.join(path[path.index(nxt):] + [nxt])}")
            elif nxt not in state:
                state[nxt] = 1
                path.append(nxt)
                todo.append(iter(edges[nxt]))
    used = {n for refs in edges.values() for n, _ in refs}
    for kid, k in prog.keys.items():
        if kid not in used:
            ctx.warnings.append(f"line {k.line} key {kid} is defined but no node uses it")
    if ctx.problems:
        return
    # Sizes are counted per key statement (memoised, in dependency order), never by walking the expansion: keys that
    # each use the previous one twice would make it exponential in their number. A key chain deeper than the
    # expanded-depth limit, or a node mask over a limit, stops the check here.
    size, deep = {}, LIMITS["mask_depth_expanded"]

    def measure(m):
        """(calls, depth) of one written mask expression (parser-bounded depth), its key statements already sized."""
        if isinstance(m, A.MaskCall):
            return 1, 1
        if isinstance(m, A.MaskNot):
            c, d = measure(m.m)
            return c, d + 1
        if isinstance(m, A.MaskBin):
            (ca, da), (cb, db) = measure(m.a), measure(m.b)
            return ca + cb, max(da, db) + 1
        if m.carried:
            return 0, 1
        return size[m.name][0], size[m.name][1] + 1

    templates = {k: [n for n, _ in edges[k] if n in prog.keys] for k in prog.keys}
    for start in prog.keys:
        if start in size:
            continue
        path, todo = [start], [iter(templates[start])]
        while todo:
            nxt = next(todo[-1], None)
            if nxt is None:
                k = path.pop()
                todo.pop()
                size[k] = measure(prog.keys[k].mask)
            elif nxt not in size:
                if len(path) >= deep:
                    ctx.err(f"line {prog.keys[start].line} key {start}",
                            f"key statements nest more than {deep} deep ({' -> '.join(path[:3])} -> ...)")
                    raise CheckError(ctx.problems)
                path.append(nxt)
                todo.append(iter(templates[nxt]))
    for nid, node in prog.nodes.items():
        if node.mask is None:
            continue
        calls, d = measure(node.mask)
        if calls > LIMITS["mask_calls_expanded"] or d > deep:
            ctx.err(f"line {node.line} node {nid}", f"its mask makes {calls} mask calls, nested {d} deep, once the "
                                                    "key statements it uses are expanded; at most "
                                                    f"{LIMITS['mask_calls_expanded']} calls, {deep} deep")
            raise CheckError(ctx.problems)


def _check_carried(ctx, prog):
    """A node's key is carried from where it ran: that node runs exactly once, in the same graph as every node that
    reads its key, and before them (depth first: serial in order, parallel branches in order, layer bottom first)."""
    order = {}
    for gname in ("graph", "photo_graph"):
        g = getattr(prog, gname)
        for i, r in enumerate(_walk_refs(g, []) if g is not None else []):
            order.setdefault(r.name, []).append((gname, i))
    for nid, node in prog.nodes.items():
        for src in carried_keys(node.mask):
            where = f"line {node.line} node {nid}"
            runs = order.get(src, [])
            if len(runs) != 1:
                ctx.err(where, f"key({src}) carries node {src}'s key, so {src} must run exactly once in graph / "
                               f"photo_graph (it runs {len(runs)} times); a `key` statement can be used anywhere")
                continue
            (sg, si), = runs
            for ug, ui in order.get(nid, []):
                if ug != sg or ui < si:
                    ctx.err(where, f"key({src}) is read before node {src} runs (or in the other graph): {src} must "
                                   f"come before {nid} in {sg}")
                    break


def _flat(m):
    """The nodes of one mask expression, not entering the masks key(...) names."""
    out, todo = [], [m]
    while todo:
        x = todo.pop()
        out.append(x)
        if isinstance(x, A.MaskNot):
            todo.append(x.m)
        elif isinstance(x, A.MaskBin):
            todo.extend((x.b, x.a))
    return out


def _walk_refs(g, out):
    if isinstance(g, A.Ref):
        out.append(g)
    elif isinstance(g, A.Serial):
        for x in g.items:
            _walk_refs(x, out)
    elif isinstance(g, A.Parallel):
        for x in g.branches:
            _walk_refs(x, out)
    elif isinstance(g, A.Layer):
        _walk_refs(g.bottom, out)
        _walk_refs(g.top, out)
    return out


def _graph(ctx, g, name):
    if isinstance(g, A.Layer):
        if g.mode not in BLEND_MODES:
            ctx.err(name, f"layer blend mode must be one of {', '.join(BLEND_MODES)}, got {g.mode!r}")
        g.opacity = _num(ctx, f"{name} layer opacity", g.opacity, 0.0, 1.0, True)
        _graph(ctx, g.bottom, name)
        _graph(ctx, g.top, name)
    elif isinstance(g, A.Serial):
        for x in g.items:
            _graph(ctx, x, name)
    elif isinstance(g, A.Parallel):
        if len(g.branches) > LIMITS["parallel_branches"]:
            ctx.err(name, f"parallel() takes at most {LIMITS['parallel_branches']} branches")
        for x in g.branches:
            _graph(ctx, x, name)


def check(prog):
    """ast.Program (from parse) -> (normalised Program, warnings). Raises CheckError listing every problem."""
    ctx = _Ctx()
    if prog.version != "0.1":
        ctx.err("header", f"this checker reads Cierra 0.1, the program says {prog.version}")
    if prog.input not in INPUTS:
        ctx.err("input", f"unknown input {prog.input!r} (inputs: {', '.join(INPUTS)})")
    out_call = prog.output or A.Call("aces2", {})
    if out_call.name not in OUTPUTS:
        ctx.err("output", f"unknown output {out_call.name!r} (outputs: {', '.join(OUTPUTS)})")
    else:
        prog.output = _call(ctx, "output", out_call, OUTPUTS[out_call.name])
    prog.reference_width = _num(ctx, "reference_width", prog.reference_width, 64, 16384, False, integer=True)
    prog.fps = _num(ctx, "fps", prog.fps, 1.0, 240.0, False)
    prog.seed = _num(ctx, "seed", prog.seed, 0, 2 ** 31 - 1, False, integer=True)
    for r in prog.requires:
        if r not in AUX:
            ctx.err("requires", f"unknown plane {r!r} (planes: {', '.join(AUX)})")
    if len(prog.keys) > LIMITS["key_statements"]:
        ctx.err("program", f"{len(prog.keys)} key statements; at most {LIMITS['key_statements']}")
    if len(prog.nodes) > LIMITS["nodes"]:
        ctx.err("program", f"{len(prog.nodes)} nodes; at most {LIMITS['nodes']}")

    for nid, node in prog.nodes.items():
        where = f"line {node.line} node {nid}"
        spec = OPS.get(node.call.name)
        if spec is None:
            ctx.err(where, f"unknown operation {node.call.name!r} (operations: {', '.join(OPS)})")
            continue
        node.call = _call(ctx, where, node.call, spec)
        node.mix = _num(ctx, f"{where} mix", node.mix, 0.0, 1.0, True)
        if node.mask is not None:
            node.mask = _mask(ctx, where, node.mask, prog.requires)
        if not node.enabled:
            ctx.warnings.append(f"{where}: node is off (it passes its input through)")
        elif spec.name in IDENTITY_DEFAULTS and all(same(node.call.args.get(k), p.default)
                                                    for k, p in spec.params.items()):
            ctx.warnings.append(f"{where}: every parameter is at its default, so {spec.name}() does nothing")
        if spec.name == "photo_to_scene":
            _check_knee_below_max(ctx, where, node.call.args.get("knee_stops"), node.call.args.get("max_stops"))

    for kid, k in prog.keys.items():
        k.mask = _mask(ctx, f"line {k.line} key {kid}", k.mask, prog.requires)
    if not ctx.problems:  # names and cycles only once every mask is well formed
        _resolve_keys(ctx, prog)

    # An omitted `graph` is the identity (an output-only program). Limits count both graphs together: node uses, and
    # spatial passes (every use of a spatial node plus every blurred mask call in the node's mask).
    used, spatial_uses, total_refs = set(), 0, 0
    for gname in ("graph", "photo_graph"):
        g = getattr(prog, gname)
        if g is None:
            continue
        refs = _walk_refs(g, [])
        total_refs += len(refs)
        for i, r in enumerate(refs):
            node = prog.nodes.get(r.name)
            if node is None:
                ctx.err(f"line {r.line} {gname}", f"no node named {r.name!r}")
                continue
            used.add(r.name)
            spec = OPS.get(node.call.name)
            if spec is None:
                continue
            spatial_uses += spec.spatial + sum(blurred(mc) for mc in mask_calls(node.mask, carried=False))
            if spec.photo_only and gname != "photo_graph":
                ctx.err(f"line {r.line} {gname}", f"{node.call.name}() may only run in photo_graph")
            if spec.photo_only and gname == "photo_graph" and not (i == 0 and _leads_serially(g, r)):
                ctx.err(f"line {r.line} photo_graph", f"{node.call.name}() must come first in photo_graph, once, "
                                                      "outside parallel / layer")
        _graph(ctx, g, gname)
    if total_refs > LIMITS["graph_refs"]:
        ctx.err("program", f"{total_refs} node uses in graph and photo_graph together; at most {LIMITS['graph_refs']}")
    if spatial_uses > LIMITS["spatial_passes"]:
        ctx.err("program", f"{spatial_uses} spatial passes (spatial node uses + blurred mask calls); "
                           f"at most {LIMITS['spatial_passes']}")
    if not ctx.problems:
        _check_carried(ctx, prog)
    for nid in prog.nodes:
        if nid not in used:
            ctx.warnings.append(f"node {nid} is defined but not in any graph")
    if ctx.problems:
        raise CheckError(ctx.problems)
    return prog, ctx.warnings


def _check_knee_below_max(ctx, where, kn, mx):
    """knee_stops < max_stops at every (integer) frame, exactly. Between consecutive keys of either value both are
    polynomials of degree <= 3 in the frame (step: constant, linear, smooth: cubic Hermite), so their difference d is
    a cubic there; over the integers of such an interval d is largest at an end or next to a stationary point of d
    (it is monotone between them). Those candidate frames are evaluated with the engine's own key_value."""
    import numpy as np
    from .engine import key_poly, key_value
    if kn is None or mx is None:
        return
    if not isinstance(kn, A.Keys) and not isinstance(mx, A.Keys):
        if not kn < mx:
            ctx.err(where, f"knee_stops ({kn:g}) must be below max_stops ({mx:g})")
        return
    val = lambda v, f: key_value(v, f) if isinstance(v, A.Keys) else v  # noqa: E731
    poly = lambda v, a, b: key_poly(v, a, b) if isinstance(v, A.Keys) else np.poly1d([v])  # noqa: E731
    frames = sorted({f for v in (kn, mx) if isinstance(v, A.Keys) for f, _ in v.keys})
    candidates = set(frames)  # before the first / after the last key both values are constant
    for a, b in zip(frames, frames[1:]):
        if b - a < 2:
            continue
        candidates.update((a + 1, b - 1))
        for r in (poly(kn, a, b) - poly(mx, a, b)).deriv().roots:
            if abs(r.imag) < 1e-9 and a < r.real < b:
                candidates.update(f for f in (int(np.floor(r.real)), int(np.ceil(r.real))) if a < f < b)
    for f in sorted(candidates):
        k, m = val(kn, f), val(mx, f)
        if not k < m:
            ctx.err(where, f"knee_stops must stay below max_stops at every frame (frame {f}: knee {k:g}, "
                           f"max {m:g})")
            return


def _leads_serially(g, ref):
    """True when `ref` is the graph itself or the first item of a top-level serial chain."""
    return g is ref or (isinstance(g, A.Serial) and g.items[0] is ref)


def lint(prog, host=None):
    """Warnings that depend on what the host provides. host: optional dict with
    frames (clip length), segment_ids (list of available ids), planes (provided aux planes), targets (e.g. ["dctl",
    "lut"]) and input (the actual input kind). Returns a list of messages (empty = nothing to say)."""
    from .targets import dctl_reasons, lut_reasons
    from .walk import graph_keys, node_keys
    host = host or {}
    out = []
    frames = host.get("frames")
    if frames is not None:
        for node in prog.nodes.values():
            last = max((f for k in node_keys(node) for f, _ in k.keys), default=None)
            if last is not None and last >= frames:
                out.append(f"node {node.id}: keyframe at frame {last} is past the clip's last frame ({frames - 1})")
        for gname in ("photo_graph", "graph"):
            last = max((f for k in graph_keys(getattr(prog, gname)) for f, _ in k.keys), default=None)
            if last is not None and last >= frames:
                out.append(f"{gname}: layer opacity keyframe at frame {last} is past the clip's last frame "
                           f"({frames - 1})")
    for p in prog.requires:
        if "planes" in host and p not in host["planes"]:
            out.append(f"the program needs the {p!r} plane, which the host does not provide")
    ids = host.get("segment_ids")
    if ids is not None:
        for node in prog.nodes.values():
            for mc in mask_calls(node.mask):
                if mc.call.name == "segment":
                    missing = [i for i in mc.call.args["ids"] if i not in ids]
                    if missing:
                        out.append(f"node {node.id}: segment ids {missing} do not exist in this shot")
    kind = host.get("input")
    if prog.photo_graph is not None and kind is not None and kind not in PHOTO_INPUTS:
        out.append(f"photo_graph will not run: the input is {kind}, not a photo ({' / '.join(PHOTO_INPUTS)})")
    targets = host.get("targets") or []
    if "dctl" in targets and prog.graph is not None:
        why = dctl_reasons(prog)
        if why:
            out.append("not expressible in DCTL v0: " + "; ".join(why))
    if "lut" in targets and prog.graph is not None:
        why = lut_reasons(prog)
        if why:
            out.append("not expressible as a LUT: " + "; ".join(why))
    return out


def load(src):
    """Source text -> checked Program (raises ParseError / CheckError)."""
    prog, _ = check(parse(src))
    return prog


def load_file(path):
    from pathlib import Path
    return load(Path(path).read_text(encoding="utf-8"))
