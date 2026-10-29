"""Parser and checker: valid programs load and format canonically; every unsafe or malformed program is rejected
with a message that names the problem."""

import pytest

from cierra import CheckError, ParseError, format_program, load
from cierra.checker import check
from cierra.parser import parse
from conftest import LOOKS

OK = """cierra 0.1
name "test"
input linear_rec709
output aces2(display: srgb)

node a = exposure(stops: 0.5)
node b = saturation(amount: 1.1, protect_skin: 1)
  mask skin(width: 30) and not luma(range: 0.6..1.3)
  mix 0.8
node c = curve(points: [[0, 0], [0.4136, 0.43], [1, 1]], channel: rgb)
node d = hue_vs_hue(points: [[206, -10, 40]])
node e = exposure(stops: keys(smooth, 0: -1, 24: 0, 48: 0.25))
node f = grain(amount: 0.1, seed: 7)
node g = wheels(lift: [0.01, 0, -0.01]) label "warm shadows" off

graph a -> parallel(b, c) -> layer(d, e, mode: multiply, opacity: 0.5) -> f -> g
"""


def test_ok_program_loads_and_round_trips():
    prog = load(OK)
    text = format_program(prog)
    again = format_program(load(text))
    assert text == again
    assert prog.nodes["b"].mix == 0.8
    assert prog.nodes["g"].enabled is False
    assert prog.nodes["g"].label == "warm shadows"


def test_shipped_looks_check_clean():
    paths = sorted(LOOKS.glob("*.cierra")) + sorted((LOOKS.parent / "examples").glob("*.cierra"))
    assert len(paths) >= 9
    for p in paths:
        prog, warnings = check(parse(p.read_text(encoding="utf-8")))
        assert not warnings, (p.name, warnings)
        assert format_program(load(format_program(prog))) == format_program(prog)


HEAD = "cierra 0.1\n"
BAD = {
    # unknown / unsafe constructs: there is no way to name a file, a URL, code or a loop
    "unknown op (file I/O)": (HEAD + 'node a = read_file(path: "/etc/passwd")\ngraph a\n', "unknown operation"),
    "lut op is not in v0": (HEAD + 'node a = lut(path: "x.cube")\ngraph a\n', "unknown operation"),
    "python expression": (HEAD + "node a = exposure(stops: __import__)\ngraph a\n", "expected a number"),
    "arithmetic is not a value": (HEAD + "node a = exposure(stops: 1 + 2)\ngraph a\n", "no arithmetic"),
    "semicolon": (HEAD + "node a = exposure(stops: 1); graph a\n", "unexpected character"),
    "loop keyword": (HEAD + "for i in 1..10\n", "unknown statement"),
    "missing header": ("node a = exposure()\ngraph a\n", "starts with the line"),
    "wrong version": ("cierra 9.9\nnode a = exposure()\ngraph a\n", "reads Cierra 0.1"),
    # out of range / mistyped
    "exposure out of range": (HEAD + "node a = exposure(stops: 40)\ngraph a\n", "outside"),
    "saturation negative": (HEAD + "node a = saturation(amount: -1)\ngraph a\n", "outside"),
    "huge number": (HEAD + "node a = exposure(stops: 1e308)\ngraph a\n", "outside"),
    "rgb wrong length": (HEAD + "node a = wheels(lift: [0.1, 0.1])\ngraph a\n", "3 numbers"),
    "enum typo": (HEAD + "node a = curve(points: [[0,0],[1,1]], channel: luminance)\ngraph a\n", "expected one of"),
    "unsorted curve": (HEAD + "node a = curve(points: [[0.5, 0.5], [0.2, 0.2]])\ngraph a\n", "increase strictly"),
    "too many points": (HEAD + "node a = curve(points: [" + ", ".join(f"[{i / 20}, {i / 20}]" for i in range(20))
                        + "])\ngraph a\n", "points"),
    "unknown parameter": (HEAD + "node a = exposure(stop: 1)\ngraph a\n", "no parameter 'stop'"),
    "duplicate parameter": (HEAD + "node a = exposure(stops: 1, stops: 2)\ngraph a\n", "given twice"),
    "bool for number": (HEAD + "node a = exposure(stops: true)\ngraph a\n", "expected a number"),
    "mix above 1": (HEAD + "node a = exposure()\n  mix 2\ngraph a\n", "outside"),
    "reversed range": (HEAD + "node a = exposure()\n  mask luma(range: 0.8..0.2)\ngraph a\n", "reversed"),
    # non-determinism: grain needs a fixed seed
    "grain without seed": (HEAD + "node a = grain(amount: 0.2)\ngraph a\n", "needs 'seed'"),
    "grain float seed": (HEAD + "node a = grain(amount: 0.2, seed: 1.5)\ngraph a\n", "integer"),
    # unbounded / structural
    "undefined node": (HEAD + "node a = exposure()\ngraph a -> b\n", "no node named 'b'"),
    "duplicate node": (HEAD + "node a = exposure()\nnode a = exposure()\ngraph a\n", "defined twice"),
    "reserved node name": (HEAD + "node parallel = exposure()\ngraph parallel\n", "reserved"),
    "keys not increasing": (HEAD + "node a = exposure(stops: keys(linear, 10: 0, 5: 1))\ngraph a\n", "increase"),
    "keys bad interp": (HEAD + "node a = exposure(stops: keys(cubic, 0: 0))\ngraph a\n", "interpolation"),
    "keys on points": (HEAD + "node a = curve(points: keys(linear, 0: 1))\ngraph a\n", "list of"),
    "keys negative frame": (HEAD + "node a = exposure(stops: keys(linear, -5: 0))\ngraph a\n", "frame"),
    "depth without plane": (HEAD + "node a = exposure()\n  mask depth(range: 1..5)\ngraph a\n", "requires depth"),
    "unknown plane": (HEAD + "requires audio\nnode a = exposure()\ngraph a\n", "unknown plane"),
    "photo op in main graph": (HEAD + "node a = photo_to_scene()\ngraph a\n", "only run in photo_graph"),
    "unknown blend mode": (HEAD + "node a = exposure()\nnode b = exposure()\ngraph layer(a, b, mode: dodge)\n",
                           "blend mode"),
    "too deep": (HEAD + "node a = exposure()\ngraph " + "(" * 12 + "a" + ")" * 12 + "\n", "deeper"),
    "too many nodes": (HEAD + "".join(f"node n{i} = exposure()\n" for i in range(70)) + "graph n0\n", "at most 64"),
    "too many spatial": (HEAD + "".join(f"node n{i} = blur(radius_px: 1)\n" for i in range(20))
                         + "graph " + " -> ".join(f"n{i}" for i in range(20)) + "\n", "spatial passes"),
    "spatial radius unbounded": (HEAD + "node a = glow(radius_px: 100000)\ngraph a\n", "outside"),
    "oversized source": (HEAD + "# " + "x" * 70000 + "\n", "larger than"),
    "long string": (HEAD + 'name "' + "x" * 500 + '"\nnode a = exposure()\ngraph a\n', "longer than"),
    "bad output": (HEAD + "output hdr10()\nnode a = exposure()\ngraph a\n", "unknown output"),
    "bad input": (HEAD + "input rawfile\nnode a = exposure()\ngraph a\n", "unknown input"),
    # every §3 limit has a rejection case (docs/conformance.md)
    "too many keyframes": (HEAD + "node a = exposure(stops: keys(linear, " + ", ".join(f"{i}: 0.1" for i in range(70))
                           + "))\ngraph a\n", "1..64 keyframes"),
    "frame past the limit": (HEAD + "node a = exposure(stops: keys(linear, 0: 0, 2000000: 1))\ngraph a\n",
                             "[0, 1000000]"),
    "long name": (HEAD + "node " + "a" * 45 + " = exposure()\ngraph " + "a" * 45 + "\n", "longer than 40"),
    "too many node uses": (HEAD + "node a = exposure()\ngraph " + " -> ".join(["a"] * 130) + "\n", "at most 128"),
    "too many branches": (HEAD + "node a = exposure()\ngraph parallel(" + ", ".join(["a"] * 9) + ")\n",
                          "at most 8 branches"),
}


@pytest.mark.parametrize("label", sorted(BAD))
def test_rejects(label):
    src, needle = BAD[label]
    with pytest.raises((ParseError, CheckError)) as e:
        load(src)
    assert needle in str(e.value), str(e.value)


def test_all_problems_reported_at_once():
    src = HEAD + "node a = exposure(stops: 40)\nnode b = saturation(amount: -1)\ngraph a -> b -> c\n"
    with pytest.raises(CheckError) as e:
        load(src)
    assert len(e.value.problems) == 3
