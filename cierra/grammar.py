"""A GBNF grammar for the grading loop's reply, generated from the op table, so a llama.cpp server can only sample
replies of the right shape: a ```cierra block (header, optional name / input, output, nodes with modifier lines, one
graph line) and then `message: ...`. Op, mask and parameter names and each parameter's value shape (number, [r, g, b],
lo..hi, points, enum choices) are fixed by the grammar; ranges, uniqueness, graph references and every other rule stay
with the checker. The grammar is narrower than the language (single spaces, no keyframes, no comments, no
photo_graph), so every reply it allows is ordinary Cierra text.

`matches(gbnf, text)` is a small reference recogniser for the GBNF subset written here (tests, no server needed).
"""

import re
from functools import lru_cache

from .ops import BLEND_MODES, INPUTS, MASKS, OPS, OUTPUTS

MAX_NODES = 12

_COMMON = r'''
num ::= "-"? [0-9]{1,6} ("." [0-9]{1,7})?
int ::= "-"? [0-9]{1,6}
ident ::= [a-zA-Z_] [a-zA-Z0-9_]{0,39}
text ::= [ !#-[\]-~]{0,400}
message ::= [^\n]{1,400}
gexpr ::= gterm (" -> " gterm)*
gterm ::= ident | "parallel(" gexpr (", " gexpr)+ ")" | "layer(" gexpr ", " gexpr (", " layer-arg)* ")"
layer-arg ::= "mode: " ("''' + '" | "'.join(BLEND_MODES) + r'''") | "opacity: " num
mask ::= mand (" or " mand)*
mand ::= munary (" and " munary)*
modline ::= "  " ("mask " mask | "mix " num | "label \"" text "\"" | "off") "\n"
'''


def _rule(name):
    return name.replace("_", "-")


def _lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _value(p):
    if p.kind == "num":
        return "num"
    if p.kind == "int":
        return "int"
    if p.kind == "bool":
        return '("true" | "false")'
    if p.kind == "enum":
        return "(" + " | ".join(_lit(c) for c in p.choices) + ")"
    if p.kind == "rgb":
        return '"[" num ", " num ", " num "]"'
    if p.kind == "pair":
        return '"[" num ", " num "]"'
    if p.kind == "range":
        return 'num ".." num'
    if p.kind == "points":
        cols = ' ", " '.join(["num"] * len(p.cols))
        opt = "".join(' (", " num' for _ in p.opt_cols) + ")?" * len(p.opt_cols)
        point = f'"[" {cols}{opt} "]"'
        return f'"[" {point} (", " {point})* "]"'
    if p.kind == "ints":
        return '"[" int (", " int)* "]"'
    raise ValueError(f"no grammar for parameter kind {p.kind!r}")


def _call(prefix, name, spec):
    """`<prefix>-<name> ::= "name(" [arg {", " arg}] ")"` and its arg rule."""
    r = f"{prefix}-{_rule(name)}"
    if not spec.params:
        return [f'{r} ::= "{name}()"']
    args = " | ".join(f'"{k}: " {_value(p)}' for k, p in spec.params.items())
    return [f'{r} ::= "{name}(" ({r}-arg (", " {r}-arg)*)? ")"', f"{r}-arg ::= {args}"]


def reply_gbnf(ops=None, masks=None):
    """GBNF for a loop reply using the named ops / masks (default: all of them)."""
    ops = sorted(ops or OPS)
    masks = sorted(masks or MASKS)
    unknown = [o for o in ops if o not in OPS] + [m for m in masks if m not in MASKS]
    if unknown:
        raise ValueError(f"not in the op table: {unknown}")
    ops = [o for o in ops if not OPS[o].photo_only]
    lines = [
        'root ::= "```cierra\\n" program "```\\nmessage: " message "\\n"?',
        # blank lines as in the canonical form the model is shown (format_program)
        'program ::= "cierra 0.1\\n" blank (meta-line blank)* output-line blank (node blank){0,' + str(MAX_NODES)
        + '} graph-line? blank',
        'blank ::= "\\n"{0,2}',
        'meta-line ::= name-line | about-line | input-line',
        'name-line ::= "name \\"" text "\\"\\n"',
        'about-line ::= "about \\"" text "\\"\\n"',
        'input-line ::= "input " (' + " | ".join(_lit(i) for i in INPUTS if not i.endswith("_photo")) + ') "\\n"',
        'output-line ::= "output " (' + " | ".join(f"out-{_rule(o)}" for o in OUTPUTS) + ') "\\n"',
        'node ::= "node " ident " = " (' + " | ".join(f"op-{_rule(o)}" for o in ops) + ') "\\n" modline*',
        'graph-line ::= "graph " gexpr "\\n"',
        'munary ::= "not " munary | "(" mask ")" | ' + " | ".join(f"mask-{_rule(m)}" for m in masks),
    ]
    for o, spec in OUTPUTS.items():
        lines += _call("out", o, spec)
    for o in ops:
        lines += _call("op", o, OPS[o])
    for m in masks:
        lines += _call("mask", m, MASKS[m])
    return "\n".join(lines) + "\n" + _COMMON.strip() + "\n"


def _mask_names(m):
    if m is None:
        return set()
    if hasattr(m, "call"):
        return {m.call.name}
    if hasattr(m, "target"):  # key(...): loop_grammar returns None before it gets here
        return set()
    if hasattr(m, "m"):
        return _mask_names(m.m)
    return _mask_names(m.a) | _mask_names(m.b)


def wrap(src, message="-"):
    """A program as a loop reply (the shape the grammar's root describes)."""
    return "```cierra\n" + src + "```\nmessage: " + message + "\n"


@lru_cache(maxsize=32)
def _cached(ops, masks):
    return reply_gbnf(ops, masks)


def loop_grammar(current, profile="short"):
    """The reply grammar for one loop round: the ops and masks the prompt documents (the short reference's set, or
    all of them) plus any the current program uses, since the model must be able to send it back unchanged. None
    when the current program uses what the grammar leaves out (keyframes, photo_graph, requires, comments, key
    statements and key(...) references)."""
    from .format import format_program
    from .llm import SHORT_MASKS, SHORT_OPS
    from .walk import mask_refs
    if current.keys or any(mask_refs(n.mask) for n in current.nodes.values()):
        return None
    ops, masks = (set(SHORT_OPS), set(SHORT_MASKS)) if profile == "short" else (set(OPS), set(MASKS))
    for node in current.nodes.values():
        ops.add(node.call.name)
        masks |= _mask_names(node.mask)
    g = _cached(tuple(sorted(ops)), tuple(sorted(masks)))
    return g if matches(g, wrap(format_program(current))) else None


# ---------------------------------------------------------------- reference recogniser (tests)

_TOKEN = re.compile(r'\s*(?:(?P<lit>"(?:[^"\\]|\\.)*")|(?P<cls>\[(?:[^\]\\]|\\.)*\])|(?P<name>[a-z0-9-]+)'
                    r'|(?P<rep>\{\d+(?:,\d*)?\})|(?P<op>[()|?*+]))')


def _unescape(s):
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), s)


def _parse_class(src):
    neg, body = src[1] == "^", src[2 if src[1] == "^" else 1:-1]
    items, i = [], 0
    while i < len(body):
        c = body[i + 1] if body[i] == "\\" else body[i]
        c = {"n": "\n", "t": "\t"}.get(c, c) if body[i] == "\\" else c
        i += 2 if body[i] == "\\" else 1
        if i + 1 < len(body) and body[i] == "-":
            hi = body[i + 2] if body[i + 1] == "\\" else body[i + 1]
            items.append((c, hi))
            i += 3 if body[i + 1] == "\\" else 2
        else:
            items.append((c, c))
    return neg, items


def _parse(gbnf):
    rules = {}
    for line in gbnf.splitlines():
        if "::=" not in line:
            continue
        name, body = line.split("::=", 1)
        toks, pos, body = [], 0, body.strip()
        while pos < len(body):
            m = _TOKEN.match(body, pos)
            if not m:
                raise ValueError(f"cannot read {body[pos:pos + 20]!r} in rule {name.strip()}")
            toks.append((m.lastgroup, m.group(m.lastgroup)))
            pos = m.end()
        rules[name.strip()] = _alt(toks, [0])
    return rules


def _alt(toks, i):
    alts = [_seq(toks, i)]
    while i[0] < len(toks) and toks[i[0]] == ("op", "|"):
        i[0] += 1
        alts.append(_seq(toks, i))
    return ("alt", alts)


def _seq(toks, i):
    items = []
    while i[0] < len(toks) and toks[i[0]] not in (("op", "|"), ("op", ")")):
        kind, v = toks[i[0]]
        i[0] += 1
        if kind == "lit":
            item = ("lit", _unescape(v[1:-1]))
        elif kind == "cls":
            item = ("cls", *_parse_class(v))
        elif kind == "name":
            item = ("ref", v)
        elif v == "(":
            item = _alt(toks, i)
            i[0] += 1  # ")"
        else:
            raise ValueError(f"unexpected {v!r}")
        while i[0] < len(toks) and (toks[i[0]][0] == "rep" or toks[i[0]][1] in "?*+"):
            r = toks[i[0]][1]
            i[0] += 1
            lo, hi = {"?": (0, 1), "*": (0, None), "+": (1, None)}.get(r, (None, None))
            if lo is None:
                a, _, b = r[1:-1].partition(",")
                lo, hi = int(a), (int(a) if not _ else (int(b) if b else None))
            item = ("rep", item, lo, hi)
        items.append(item)
    return ("seq", items)


def matches(gbnf, text, start="root"):
    """True when `text` is a complete sentence of the grammar (memoised set-of-ends recogniser)."""
    rules = _parse(gbnf)

    @lru_cache(maxsize=None)
    def ends_ref(name, pos):
        return ends(rules[name], pos)

    def ends(node, pos):
        kind = node[0]
        if kind == "lit":
            return {pos + len(node[1])} if text.startswith(node[1], pos) else set()
        if kind == "cls":
            if pos >= len(text):
                return set()
            hit = any(lo <= text[pos] <= hi for lo, hi in node[2])
            return {pos + 1} if hit != node[1] else set()
        if kind == "ref":
            return ends_ref(node[1], pos)
        if kind == "alt":
            return set().union(*(ends(a, pos) for a in node[1]))
        if kind == "seq":
            cur = {pos}
            for item in node[1]:
                cur = set().union(*(ends(item, p) for p in cur)) if cur else set()
            return cur
        _, item, lo, hi = node

        def step(ps):
            return set().union(*(ends(item, p) for p in ps)) if ps else set()

        cur = {pos}
        for _ in range(lo):
            cur = step(cur)
        out = set(cur)
        if hi is None:  # closure: keep stepping from positions not seen yet
            while cur:
                cur = step(cur) - out
                out |= cur
        else:
            for _ in range(hi - lo):
                cur = step(cur)
                if not cur:
                    break
                out |= cur
        return out

    return len(text) in ends_ref(start, 0)
