"""Tree walks shared by the checker, the engine, the compilers, the linter and the diff (one place, so a new kind of
keyframe or mask is seen by every consumer)."""

from . import ast as A


def graph_refs(g, out=None):
    out = [] if out is None else out
    if isinstance(g, A.Ref):
        out.append(g.name)
    elif isinstance(g, A.Serial):
        for x in g.items:
            graph_refs(x, out)
    elif isinstance(g, A.Parallel):
        for x in g.branches:
            graph_refs(x, out)
    elif isinstance(g, A.Layer):
        graph_refs(g.bottom, out)
        graph_refs(g.top, out)
    return out


def graph_nodes(prog, g):
    """Distinct nodes used by a graph, in first-use order."""
    seen, out = set(), []
    for name in graph_refs(g):
        if name not in seen:
            seen.add(name)
            out.append(prog.nodes[name])
    return out


def graph_layers(g, out=None):
    """Every layer() in a graph."""
    out = [] if out is None else out
    if isinstance(g, A.Serial):
        for x in g.items:
            graph_layers(x, out)
    elif isinstance(g, A.Parallel):
        for x in g.branches:
            graph_layers(x, out)
    elif isinstance(g, A.Layer):
        out.append(g)
        graph_layers(g.bottom, out)
        graph_layers(g.top, out)
    return out


def mask_calls(m, carried=True):
    """Every mask call in a mask expression (iterative: deep expressions cannot exhaust the stack), including those of
    the masks key(...) names. carried=False leaves out a carried node key's calls: they run where that node runs."""
    out, todo = [], [m]
    while todo:
        x = todo.pop()
        if x is None:
            continue
        if isinstance(x, A.MaskCall):
            out.append(x)
        elif isinstance(x, A.MaskNot):
            todo.append(x.m)
        elif isinstance(x, A.MaskRef):
            if carried or not x.carried:
                todo.append(x.target)
        else:
            todo.extend((x.b, x.a))
    return out


def same(a, b):
    """Value equality where lists and tuples compare by content."""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def keys_in(v):
    """Every Keys inside a value (scalars, vectors)."""
    if isinstance(v, A.Keys):
        return [v]
    if isinstance(v, (tuple, list)):
        return [k for x in v for k in keys_in(x)]
    return []


def node_keys(node):
    """Every Keys of a node: op parameters, mix and mask numbers."""
    out = keys_in(node.mix)
    for v in node.call.args.values():
        out += keys_in(v)
    for mc in mask_calls(node.mask):
        for v in mc.call.args.values():
            out += keys_in(v)
    for r in mask_refs(node.mask):
        out += keys_in(r.gain) + keys_in(r.offset)
    return out


def carried_keys(m):
    """Names of the nodes whose carried key a mask reads (through `key` statements, not through other carried keys),
    sorted."""
    out, todo = set(), [m]
    while todo:
        x = todo.pop()
        if isinstance(x, A.MaskRef):
            if x.carried:
                out.add(x.name)
            else:
                todo.append(x.target)
        elif isinstance(x, A.MaskNot):
            todo.append(x.m)
        elif isinstance(x, A.MaskBin):
            todo.extend((x.b, x.a))
    return sorted(out)


def mask_refs(m):
    """Every key(...) in a mask expression, including those inside the masks it refers to."""
    out, todo = [], [m]
    while todo:
        x = todo.pop()
        if isinstance(x, A.MaskRef):
            out.append(x)
            todo.append(x.target)
        elif isinstance(x, A.MaskNot):
            todo.append(x.m)
        elif isinstance(x, A.MaskBin):
            todo.extend((x.b, x.a))
    return out


def has_keys(node):
    return bool(node_keys(node))


def graph_keys(g):
    """Every Keys on the graph itself (layer opacity)."""
    return [k for layer in graph_layers(g) for k in keys_in(layer.opacity)]


def program_keys(prog):
    """Every Keys in a program: nodes used by either graph, and layer opacities."""
    out = []
    for g in (prog.photo_graph, prog.graph):
        for n in graph_nodes(prog, g):
            out += node_keys(n)
        out += graph_keys(g)
    return out
