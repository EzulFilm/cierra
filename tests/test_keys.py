"""Key routing (G10): `key` statements, key(node) carrying a node's key, the outside of a key, and the key mixer
(+, -, gain, offset)."""

import numpy as np
import pytest

from cierra import ast as A
from cierra import colour as C
from cierra.checker import check, load
from cierra.compile.cdl import CDLError, fold
from cierra.compile.lut import LUTError, bake
from cierra.engine import grade_working, make_ctx, mask_value, run_graph
from cierra.errors import CheckError, ParseError
from cierra.format import format_program
from cierra.parser import parse
from conftest import frame

HEAD = "cierra 0.1\noutput acescct()\n"
WIN = "window(centre: [0.4, 0.5], radius: [0.25, 0.3])"
GRAD = "graduated(centre: [0.5, 0.4], angle: 90, falloff: [0.1, 0.2])"
SKIN = "qualifier(hue: 100..180, sat: 0.01..0.2, soft: 0.1)"


def code():
    return C.to_working(frame("crowd"))[:, ::8, ::8]


def masks(src):
    """Each node's mask as the graph runs (carried keys included), over the crowd frame, keyed by node name."""
    prog = load(HEAD + src)
    c = code()
    ctx = make_ctx(prog, c.shape[-1])
    seen = {}
    import cierra.engine as E
    real = E.run_node

    def spy(x, node, ctx):
        if node.mask is not None:
            seen[node.id] = mask_value(x, node.mask, ctx) if node.id not in ctx["keys"] else None
        out = real(x, node, ctx)
        if seen.get(node.id) is None and node.id in ctx["keys"]:
            seen[node.id] = ctx["keys"][node.id]
        return out
    E.run_node = spy
    try:
        run_graph(c, prog.graph, prog, ctx)
    finally:
        E.run_node = real
    return seen


def test_outside_node_is_one_minus_the_inside():
    m = masks(f"node face = exposure(stops: 0.5)\n  mask {WIN}\nnode rest = saturation(amount: 0.5)\n"
              "  mask not key(face)\ngraph face -> rest\n")
    assert np.array_equal(m["rest"], 1 - m["face"]) and 0 < m["face"].mean() < 1


def test_node_key_is_carried_from_where_the_node_ran():
    """A skin node moves the skin hue; the outside node still takes the key made on the skin node's input (as the
    grading app routes a key), while a key statement is made again on the outside node's own input."""
    src = (f"key skin = {SKIN}\nnode warm = hue_vs_hue(points: [[60, 0], [140, 40], [220, 0]])\n  mask {SKIN}\n"
           "node outside = saturation(amount: 0.5)\n  mask not key(warm)\n"
           "node remade = saturation(amount: 0.5)\n  mask not key(skin)\ngraph warm -> outside -> remade\n")
    m = masks(src)
    assert np.allclose(m["outside"], 1 - m["warm"])
    assert np.abs(m["remade"] - m["outside"]).max() > 0.05


def test_off_node_still_gives_its_key():
    prog, warnings = check(parse(HEAD + f"node a = exposure(stops: 1)\n  mask {WIN}\n  off\n"
                                       "node b = exposure(stops: 1)\n  mask key(a)\ngraph a -> b\n"))
    assert any("off" in w and "key(a)" in w for w in warnings)
    c = code()
    assert not np.array_equal(grade_working(c, prog), c)


def test_named_key_is_the_same_mask_as_writing_it_out():
    m = masks(f"key sky = {GRAD}\nnode a = exposure(stops: 1)\n  mask key(sky)\n"
              f"node b = exposure(stops: 1)\n  mask {GRAD}\ngraph a -> b\n")
    assert np.array_equal(m["a"], m["b"])


def test_key_mixer():
    m = masks(f"key w = {WIN}\nkey g = {GRAD}\n"
              "node add = exposure(stops: 0)\n  mask key(w) + key(g)\n"
              "node sub = exposure(stops: 0)\n  mask key(w) - key(g)\n"
              "node mul = exposure(stops: 0)\n  mask key(w) and key(g)\n"
              "node gain = exposure(stops: 0)\n  mask key(w, gain: 2, offset: -0.5)\n"
              "node inv = exposure(stops: 0)\n  mask not key(w, gain: 0.5)\n"
              f"node w_ = exposure(stops: 0)\n  mask {WIN}\nnode g_ = exposure(stops: 0)\n  mask {GRAD}\n"
              "graph add -> sub -> mul -> gain -> inv -> w_ -> g_\n")
    w, g = m["w_"], m["g_"]
    assert np.allclose(m["add"], np.minimum(w + g, 1)) and np.allclose(m["sub"], np.maximum(w - g, 0))
    assert np.allclose(m["mul"], w * g) and np.allclose(m["gain"], np.clip(2 * w - 0.5, 0, 1))
    assert np.allclose(m["inv"], 1 - 0.5 * w)  # gain / offset apply before `not`
    assert m["add"].max() <= 1 and m["sub"].min() >= 0


def test_mixing_or_plus_minus_needs_brackets():
    with pytest.raises(ParseError, match="brackets"):
        load(HEAD + f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask key(a) - {GRAD} or luma(range: 0.2..0.5)\n")
    prog = load(HEAD + f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask (key(a) - {GRAD}) or luma(range: 0.2..0.5)\n"
                       "graph x\n")
    m = prog.nodes["x"].mask
    assert m.op == "or" and m.a.op == "-"
    prog = load(HEAD + f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask key(a) - {GRAD} - {WIN}\ngraph x\n")
    assert prog.nodes["x"].mask.a.op == "-"  # one kind repeats left to right


def test_round_trip():
    src = (HEAD + f"key sky = {GRAD}\nnode face = exposure(stops: 0.5)\n  mask {WIN}\n"
           "node rest = saturation(amount: 0.6)\n  mask not key(face) - key(sky, gain: 0.5, offset: 0.1)\n"
           "graph face -> rest\n")
    text = format_program(load(src))
    assert "key sky = graduated(" in text and "mask not key(face) - key(sky, gain: 0.5, offset: 0.1)" in text
    assert format_program(load(text)) == text


@pytest.mark.parametrize("src,msg", [
    ("key a = key(b)\nkey b = key(a)\n", "cycle"),
    ("node x = exposure(stops: 1)\n  mask key(x)\ngraph x\n", "cycle"),
    ("node x = exposure(stops: 1)\n  mask key(nope)\ngraph x\n", "no node or key named"),
    ("node y = exposure(stops: 1)\nnode x = exposure(stops: 1)\n  mask key(y)\ngraph y -> x\n", "has no mask"),
    (f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask key(a, gain: 9)\ngraph x\n", "gain"),
    (f"node a = exposure(stops: 1)\n  mask {WIN}\nnode b = exposure(stops: 1)\n  mask key(a)\ngraph b -> a\n",
     "before node a runs"),
    (f"node a = exposure(stops: 1)\n  mask {WIN}\nnode b = exposure(stops: 1)\n  mask key(a)\ngraph a -> b -> a\n",
     "exactly once"),
    (f"node a = exposure(stops: 1)\n  mask {WIN}\nnode b = exposure(stops: 1)\n  mask key(a)\ngraph b\n",
     "exactly once"),
])
def test_bad_references(src, msg):
    with pytest.raises(CheckError, match=msg):
        load(HEAD + src)


def test_carried_key_through_a_key_statement_and_parallel_branch():
    prog = load(HEAD + f"node a = exposure(stops: 1)\n  mask {WIN}\nkey outside_a = not key(a)\n"
                       "node b = saturation(amount: 0.5)\n  mask key(outside_a)\ngraph parallel(a, b)\n")
    c = code()
    assert np.isfinite(grade_working(c, prog)).all()


@pytest.mark.parametrize("src", [f"key a = {WIN}\nkey a = {WIN}\n", f"key x = {WIN}\nnode x = exposure(stops: 1)\n",
                                 f"key not = {WIN}\n", "node x = exposure(stops: 1)\n  mask key(3)\n",
                                 "node x = exposure(stops: 1)\n  mask key(a, soft: 1)\n",
                                 "node x = exposure(stops: 1)\n  mask key(a, gain: 1, gain: 2)\n"])
def test_bad_syntax(src):
    with pytest.raises(ParseError):
        load(HEAD + src)


def test_expansion_is_bounded_without_walking_it():
    """40 keys that each use the previous one twice expand to 2^39 calls: refused at once, never expanded."""
    src = f"key k0 = {WIN}\n" + "".join(f"key k{i} = key(k{i - 1}) or key(k{i - 1})\n" for i in range(1, 40))
    with pytest.raises(CheckError, match="mask calls"):
        load(HEAD + src + "node x = exposure(stops: 1)\n  mask key(k39)\ngraph x\n")


def test_expansion_depth_is_bounded():
    src = f"key k0 = {WIN}\n" + "".join(f"key k{i} = not key(k{i - 1})\n" for i in range(1, 40))
    with pytest.raises(CheckError, match="deep"):
        load(HEAD + src + "node x = exposure(stops: 1)\n  mask key(k39)\ngraph x\n")


def test_key_statement_blur_counts_where_used_but_a_carried_one_once():
    uses = "".join(f"node n{i} = exposure(stops: 0.1)\n  mask key(soft)\n" for i in range(17))
    chain = "graph " + " -> ".join(f"n{i}" for i in range(17)) + "\n"
    with pytest.raises(CheckError, match="spatial passes"):
        load(HEAD + "key soft = window(blur_px: 10)\n" + uses + chain)
    load(HEAD + "node soft = exposure(stops: 0.1)\n  mask window(blur_px: 10)\n" + uses
         + "graph soft -> " + " -> ".join(f"n{i}" for i in range(17)) + "\n")


def test_unused_key_warns():
    _, warnings = check(parse(HEAD + f"key spare = {WIN}\nnode x = exposure(stops: 1)\ngraph x\n"))
    assert any("key spare" in w for w in warnings)


def test_keyframed_gain():
    prog = load(HEAD + f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask key(a, gain: keys(linear, 0: 0, 10: 1))\n"
                       "graph x\n")
    c = code()
    assert np.array_equal(grade_working(c, prog, frame=0), c)
    assert not np.array_equal(grade_working(c, prog, frame=10), c)


def test_lut_takes_a_pointwise_key_and_cdl_refuses_masks():
    src = HEAD + "key hi = luma(range: 0.5..1)\nnode x = saturation(amount: 0.5)\n  mask not key(hi)\ngraph x\n"
    bake(load(src), size=9)
    with pytest.raises(CDLError):
        fold(load(src))
    with pytest.raises(LUTError, match="not pointwise"):
        bake(load(HEAD + f"key w = {WIN}\nnode x = saturation(amount: 0.5)\n  mask key(w)\ngraph x\n"))


def test_ast_keeps_the_reference_for_formatting():
    prog = load(HEAD + f"key a = {WIN}\nnode b = exposure(stops: 1)\n  mask {GRAD}\nnode x = exposure(stops: 1)\n"
                       "  mask key(a) and key(b)\ngraph b -> x\n")
    m = prog.nodes["x"].mask
    assert isinstance(m.a, A.MaskRef) and m.a.target is prog.keys["a"].mask and not m.a.carried and m.b.carried


def test_a_long_key_chain_is_a_checker_error_not_a_crash():
    """1,199 chained key statements (25 KB of source): refused by the checker, never a RecursionError."""
    src = f"key k0 = {WIN}\n" + "".join(f"key k{i} = key(k{i - 1})\n" for i in range(1, 1200))
    with pytest.raises(CheckError, match="key statements"):
        load(HEAD + src + "node x = exposure(stops: 1)\n  mask key(k1199)\ngraph x\n")


def test_a_long_key_cycle_is_a_checker_error_not_a_crash():
    for n, msg in ((60, "cycle"), (1000, "key statements")):  # past 64 statements the count stops it first
        src = "".join(f"key k{i} = key(k{(i + 1) % n})\n" for i in range(n))
        with pytest.raises(CheckError, match=msg):
            load(HEAD + src)


def test_a_key_chain_deeper_than_the_limit_is_refused_before_it_is_walked():
    src = f"key k0 = {WIN}\n" + "".join(f"key k{i} = key(k{i - 1})\n" for i in range(1, 64))
    with pytest.raises(CheckError, match="deep"):
        load(HEAD + src + "node x = exposure(stops: 1)\n  mask key(k63)\ngraph x\n")


def test_key_programs_get_no_reply_grammar():
    from cierra.grammar import loop_grammar
    for src in (f"key a = {WIN}\nnode x = exposure(stops: 1)\n  mask key(a)\ngraph x\n",
                f"node a = exposure(stops: 1)\n  mask {WIN}\nnode b = exposure(stops: 1)\n  mask not key(a)\n"
                "graph a -> b\n"):
        assert loop_grammar(load("cierra 0.1\noutput aces2()\n" + src), "short") is None


def test_history_diff_shows_key_statements():
    from cierra.history import diff
    node = "node x = exposure(stops: 1)\n  mask key(sky)\ngraph x\n"
    a = load(HEAD + f"key sky = {GRAD}\n" + node)
    b = load(HEAD + f"key sky = {WIN}\n" + node)
    c = load(HEAD + f"key sky = {WIN}\nkey spare = {GRAD}\n" + node)
    assert diff(a, b) == [f"~ key sky: {format_program(a).split('key sky = ')[1].splitlines()[0]} -> "
                          f"{format_program(b).split('key sky = ')[1].splitlines()[0]}"]
    assert [x for x in diff(b, c) if "spare" in x][0].startswith("+ key spare added: graduated(")
    assert [x for x in diff(c, b) if "spare" in x][0].startswith("- key spare removed (graduated(")
    assert diff(None, c) == ["new program: 1 nodes, 2 key statements"]
