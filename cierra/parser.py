"""Tokenizer and recursive-descent parser for Cierra source text (grammar: docs/spec.md section 2).

The parser builds a syntax tree and never evaluates anything; it knows the shape of the language, not the ops. The
checker (checker.py) decides what is allowed.
"""

import math
import re

from . import ast as A
from .errors import ParseError
from .ops import LIMITS

_TOKEN = re.compile(r"""
    (?P<ws>[ \t\r]+)
  | (?P<comment>\#[^\n]*)
  | (?P<nl>\n)
  | (?P<number>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
  | (?P<string>"(?:[^"\\\n]|\\["\\])*")
  | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<op>->|\.\.|[()\[\],:=+-])
""", re.VERBOSE)

MODIFIERS = ("mask", "mix", "label", "off")
RESERVED = ("pass", "parallel", "layer", "keys", "true", "false", "not", "and", "or")
# C0 / C1 controls and Unicode line / paragraph separators: a string is one line of printable text, so a name or label
# can never start a new line in generated DCTL, WGSL or LUT files.
# also bidi embeddings, overrides and isolates (they reorder what a reader sees in a generated comment: "Trojan
# Source") and Unicode noncharacters (invalid in the CLF / CDL XML)
CONTROL = re.compile("[\x00-\x1f\x7f-\x9f\u2028-\u2029\u202a-\u202e\u2066-\u2069\ufdd0-\ufdef"
                     + "".join(chr(p * 0x10000 + 0xFFFE) + chr(p * 0x10000 + 0xFFFF) for p in range(17)) + "]")


class Tok:
    __slots__ = ("kind", "value", "line", "col")

    def __init__(self, kind, value, line, col):
        self.kind, self.value, self.line, self.col = kind, value, line, col

    def __repr__(self):
        return f"{self.kind}:{self.value!r}@{self.line}"


def tokenize(src):
    if len(src.encode("utf-8")) > LIMITS["source_bytes"]:
        raise ParseError(f"program is larger than {LIMITS['source_bytes']} bytes")
    toks, pos, line, line_start, depth = [], 0, 1, 0, 0
    while pos < len(src):
        m = _TOKEN.match(src, pos)
        if not m:
            ch = src[pos]
            raise ParseError(f"unexpected character {ch!r}", line, pos - line_start + 1)
        kind, text, col = m.lastgroup, m.group(), pos - line_start + 1
        pos = m.end()
        if kind == "nl":
            if depth == 0 and toks and toks[-1].kind != "nl":
                toks.append(Tok("nl", "\n", line, col))
            line, line_start = line + 1, pos
            continue
        if kind in ("ws", "comment"):
            continue
        if kind == "op" and text in "([":
            depth += 1
        elif kind == "op" and text in ")]":
            depth = max(0, depth - 1)
        if kind == "number":
            digits = sum(c.isdigit() for c in text.split("e")[0].split("E")[0])
            if digits > LIMITS["number_digits"]:
                raise ParseError(f"number with {digits} digits; at most {LIMITS['number_digits']}", line, col)
            value = float(text) if any(c in text for c in ".eE") else int(text)
            if isinstance(value, float) and not math.isfinite(value):
                raise ParseError(f"number {text} is too large", line, col)
        elif kind == "string":
            value = text[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            bad = CONTROL.search(value)
            if bad:
                raise ParseError(f"control character U+{ord(bad.group()):04X} in a string: strings are one line of "
                                 "printable text", line, col)
            if len(value) > LIMITS["string"]:
                raise ParseError(f"string longer than {LIMITS['string']} characters", line, col)
        else:
            value = text
        toks.append(Tok(kind, value, line, col))
    if not toks or toks[-1].kind != "nl":
        toks.append(Tok("nl", "\n", line, 1))
    toks.append(Tok("eof", None, line, 1))
    return toks


class Parser:
    def __init__(self, src):
        self.toks = tokenize(src)
        self.i = 0
        self.modifiers_seen = {}

    # -------------------------------------------------------------- token helpers
    @property
    def tok(self):
        return self.toks[self.i]

    def at(self, kind, value=None):
        t = self.tok
        return t.kind == kind and (value is None or t.value == value)

    def take(self, kind, value=None, what=None):
        t = self.tok
        if not self.at(kind, value):
            want = what or (repr(value) if value is not None else kind)
            got = "end of line" if t.kind == "nl" else ("end of file" if t.kind == "eof" else repr(t.value))
            raise ParseError(f"expected {want}, found {got}", t.line, t.col)
        self.i += 1
        return t

    def accept(self, kind, value=None):
        if self.at(kind, value):
            self.i += 1
            return True
        return False

    def ident(self, what="a name"):
        t = self.take("ident", what=what)
        if len(t.value) > LIMITS["ident"]:
            raise ParseError(f"name longer than {LIMITS['ident']} characters", t.line, t.col)
        return t.value

    # -------------------------------------------------------------- values
    def number(self):
        t = self.take("number", what="a number")
        return t.value

    def value(self, depth=0):
        t = self.tok
        if depth > LIMITS["value_depth"]:
            raise ParseError(f"values nested deeper than {LIMITS['value_depth']} levels of [ ]", t.line, t.col)
        if t.kind == "number":
            self.i += 1
            if self.accept("op", ".."):
                return A.Range(t.value, self.number())
            if self.at("op", "+") or self.at("op", "-"):
                raise ParseError("a value is one number: no arithmetic (+, -)", self.tok.line, self.tok.col)
            return t.value
        if t.kind == "string":
            self.i += 1
            return t.value
        if t.kind == "ident":
            self.i += 1
            if t.value in ("true", "false"):
                return t.value == "true"
            if t.value == "keys" and self.at("op", "("):
                return self.keys(depth + 1)
            return A.Ident(t.value)
        if self.accept("op", "["):
            items = []
            while not self.at("op", "]"):
                items.append(self.value(depth + 1))
                if not self.accept("op", ","):
                    break
            self.take("op", "]")
            return items
        raise ParseError(f"expected a value, found {t.value!r}", t.line, t.col)

    def keys(self, depth=0):
        self.take("op", "(")
        interp = self.ident("an interpolation (step, linear, smooth)")
        pairs = []
        while self.accept("op", ","):
            if self.at("op", ")"):
                break
            f = self.take("number", what="a frame number")
            self.take("op", ":")
            pairs.append((f.value, self.value(depth + 1)))
        self.take("op", ")")
        return A.Keys(interp, tuple(pairs))

    def call(self):
        t = self.tok
        name = self.ident("an operation name")
        args = {}
        if self.accept("op", "("):
            while not self.at("op", ")"):
                at = self.tok
                key = self.ident("a parameter name")
                self.take("op", ":")
                if key in args:
                    raise ParseError(f"parameter {key!r} given twice", at.line, at.col)
                args[key] = self.value()
                if not self.accept("op", ","):
                    break
            self.take("op", ")")
        return A.Call(name, args, t.line)

    # -------------------------------------------------------------- masks
    def mask(self):
        """A whole mask expression, with its depth and call count bounded (LIMITS mask_depth / mask_calls), so
        every recursive walk of it stays shallow."""
        self.mask_n = 0
        return self.mask_or(0)

    def _mask_depth(self, depth):
        if depth > LIMITS["mask_depth"]:
            t = self.tok
            raise ParseError(f"mask expression nested deeper than {LIMITS['mask_depth']}", t.line, t.col)

    def mask_or(self, depth):
        self._mask_depth(depth)
        m, first = self.mask_and(depth + 1), None
        while True:  # or, + and - (the key mixer's add / subtract): one kind per level, left to right; mixing needs ( )
            t = self.tok
            if self.accept("ident", "or"):
                op = "or"
            elif self.at("op", "+") or self.at("op", "-"):
                op = t.value
                self.i += 1
            else:
                return m
            if first is not None and op != first:
                raise ParseError(f"`{first}` and `{op}` in one mask need brackets to say which comes first",
                                 t.line, t.col)
            first = op
            m = A.MaskBin(op, m, self.mask_and(depth + 1))

    def mask_and(self, depth):
        self._mask_depth(depth)
        m = self.mask_unary(depth + 1)
        while self.accept("ident", "and"):
            m = A.MaskBin("and", m, self.mask_unary(depth + 1))
        return m

    def mask_unary(self, depth):
        self._mask_depth(depth)
        if self.accept("ident", "not"):
            return A.MaskNot(self.mask_unary(depth + 1))
        if self.accept("op", "("):
            m = self.mask_or(depth + 1)
            self.take("op", ")")
            return m
        t = self.tok
        self.mask_n += 1
        if self.mask_n > LIMITS["mask_calls"]:
            raise ParseError(f"more than {LIMITS['mask_calls']} mask calls in one mask", t.line, t.col)
        nxt = self.toks[self.i + 1] if self.i + 1 < len(self.toks) else None
        if t.kind == "ident" and t.value == "key" and nxt is not None and nxt.kind == "op" and nxt.value == "(":
            return self.key_ref()
        return A.MaskCall(self.call())

    def key_ref(self):
        """key(NAME [, gain: v] [, offset: v])"""
        t = self.tok
        self.i += 2
        ref = A.MaskRef(self.ident("a node or key name"), line=t.line)
        given = set()
        while self.accept("op", ","):
            k = self.tok
            arg = self.ident("gain or offset")
            if arg not in ("gain", "offset") or arg in given:
                raise ParseError(f"key() takes a name, then gain and offset at most once each, not {arg!r}",
                                 k.line, k.col)
            given.add(arg)
            self.take("op", ":")
            setattr(ref, arg, self.value())
        self.take("op", ")")
        return ref

    # -------------------------------------------------------------- graph
    def gexpr(self, depth=0):
        if depth > LIMITS["graph_depth"]:
            raise ParseError(f"graph nested deeper than {LIMITS['graph_depth']}", self.tok.line)
        items = [self.gterm(depth)]
        while self.accept("op", "->"):
            items.append(self.gterm(depth))
        return items[0] if len(items) == 1 else A.Serial(items)

    def gterm(self, depth):
        t = self.tok
        if self.accept("op", "("):
            g = self.gexpr(depth + 1)
            self.take("op", ")")
            return g
        name = self.ident("a node name, pass, parallel(...) or layer(...)")
        if name == "pass":
            return A.Pass()
        if name == "parallel" and self.accept("op", "("):
            branches = [self.gexpr(depth + 1)]
            while self.accept("op", ","):
                branches.append(self.gexpr(depth + 1))
            self.take("op", ")")
            return A.Parallel(branches)
        if name == "layer" and self.accept("op", "("):
            bottom = self.gexpr(depth + 1)
            self.take("op", ",")
            top = self.gexpr(depth + 1)
            mode, opacity, given = "normal", 1.0, set()
            while self.accept("op", ","):
                k = self.tok
                key = self.ident("mode or opacity")
                if key in given:
                    raise ParseError(f"layer() {key} given twice", k.line, k.col)
                given.add(key)
                self.take("op", ":")
                if key == "mode":
                    mode = self.ident("a blend mode")
                elif key == "opacity":
                    opacity = self.value()
                else:
                    raise ParseError(f"layer() takes mode and opacity, not {key!r}", k.line, k.col)
            self.take("op", ")")
            return A.Layer(bottom, top, mode, opacity)
        return A.Ref(name, t.line)

    # -------------------------------------------------------------- statements
    def program(self):
        while self.accept("nl"):
            pass
        t = self.tok
        if not self.accept("ident", "cierra"):
            raise ParseError("a program starts with the line `cierra 0.1`", t.line, t.col)
        v = self.take("number", what="the language version (0.1)")
        prog = A.Program(version=str(v.value))
        self.take("nl", what="end of line")
        last_node, seen = None, set()
        while not self.at("eof"):
            if self.accept("nl"):
                continue
            t = self.tok
            kw = self.ident("a statement")
            if kw in MODIFIERS:
                if last_node is None:
                    raise ParseError(f"`{kw}` must follow a node statement", t.line, t.col)
                self.modifier(kw, last_node, t)
                self.end_modifiers(last_node)
                continue
            last_node = None
            single = {"name", "about", "input", "output", "reference_width", "fps", "seed", "requires", "graph",
                      "photo_graph"}
            if kw in single:
                if kw in seen:
                    raise ParseError(f"`{kw}` given twice", t.line, t.col)
                seen.add(kw)
            if kw == "name":
                prog.name = self.take("string", what="a quoted name").value
            elif kw == "about":
                prog.about = self.take("string", what="a quoted description").value
            elif kw == "input":
                prog.input = self.ident("an input colour space")
            elif kw == "output":
                prog.output = self.call()
            elif kw == "reference_width":
                prog.reference_width = self.number()
            elif kw == "fps":
                prog.fps = self.number()
            elif kw == "seed":
                prog.seed = self.number()
            elif kw == "requires":
                req = [self.ident("an auxiliary plane (depth, segments)")]
                while self.accept("op", ","):
                    req.append(self.ident("an auxiliary plane"))
                prog.requires = tuple(req)
            elif kw in ("graph", "photo_graph"):
                self.accept("op", "=")
                g = self.gexpr()
                setattr(prog, kw, g)
            elif kw == "key":
                kt = self.tok
                kid = self.ident("a key name")
                if kid in prog.nodes or kid in prog.keys:
                    raise ParseError(f"{kid!r} defined twice", kt.line, kt.col)
                if kid in RESERVED:
                    raise ParseError(f"{kid!r} is reserved", kt.line, kt.col)
                self.take("op", "=")
                prog.keys[kid] = A.Key(kid, self.mask(), kt.line)
            elif kw == "node":
                nid_tok = self.tok
                nid = self.ident("a node name")
                if nid in prog.nodes or nid in prog.keys:
                    raise ParseError(f"{nid!r} defined twice", nid_tok.line, nid_tok.col)
                if nid in RESERVED:
                    raise ParseError(f"{nid!r} is reserved", nid_tok.line, nid_tok.col)
                self.take("op", "=")
                node = A.Node(nid, self.call(), line=nid_tok.line)
                prog.nodes[nid] = node
                last_node = node
                self.end_modifiers(node)
                continue
            else:
                raise ParseError(f"unknown statement {kw!r}", t.line, t.col)
            self.take("nl", what="end of line")
        return prog

    def modifier(self, kw, node, t):
        seen = self.modifiers_seen.setdefault(node.id, set())
        if kw in seen:
            if kw == "mask":
                raise ParseError(f"node {node.id!r} has two masks; combine them with and / or", t.line, t.col)
            raise ParseError(f"node {node.id!r} has `{kw}` twice", t.line, t.col)
        seen.add(kw)
        if kw == "mask":
            node.mask = self.mask()
        elif kw == "mix":
            node.mix = self.value()
        elif kw == "label":
            node.label = self.take("string", what="a quoted label").value
        elif kw == "off":
            node.enabled = False

    def end_modifiers(self, node):
        """Same-line modifiers, then end of line."""
        while self.at("ident") and self.tok.value in MODIFIERS:
            t = self.tok
            self.i += 1
            self.modifier(t.value, node, t)
        self.take("nl", what="end of line")


def parse(src):
    """Source text -> ast.Program (syntax only; run checker.check for meaning)."""
    return Parser(src).program()
