"""Reference engine: identities, semantics of the graph, keyframes, determinism."""

import numpy as np
import pytest

from cierra import load
from cierra import colour as C
from cierra.engine import grade_working, key_value, render
from cierra.ops import OPS
from conftest import small

HEAD = "cierra 0.1\noutput acescct()\n"


def run(src, img, **kw):
    prog = load(src)
    code = C.to_working(img)
    return code, grade_working(code, prog, px_scale=1.0, **kw)


IDENTITY = [
    "exposure()", "white_balance()", "printer_lights()", "wheels()", "cdl()", "contrast()", "saturation()",
    "curve(points: [[0, 0], [1, 1]])", "hue_vs_hue(points: [[100, 0]])", "hue_vs_sat(points: [[100, 1]])",
    "hue_vs_lum(points: [[100, 0]])", "lum_vs_sat(points: [[0, 1], [1, 1]])", "vignette(amount: 0)",
    "grain(amount: 0, seed: 1)", "glow(strength: 0)", "halation(strength: 0)", "diffusion(strength: 0)",
    "blur(radius_px: 0)", "sharpen(amount: 0)", "local_contrast(amount: 0)", "highlights_shadows()",
            "guided_detail(amount: 0)", "rgb_mixer()", "density()", "hue_rotate()", "vibrance()", "split_tone()",
            "sat_vs_lum(points: [[0, 0], [1, 0]])"]


@pytest.mark.parametrize("node", IDENTITY)
def test_neutral_parameters_are_identity(chart, node):
    code, out = run(HEAD + f"node a = {node}\ngraph a\n", small(chart))
    assert np.abs(out - code).max() < 1e-9


def test_every_op_has_an_identity_case():
    covered = {c.split("(")[0] for c in IDENTITY} | {"sat_vs_sat", "photo_to_scene", "gamut_compress"}  # own tests
    assert covered == set(OPS)


def test_sat_vs_sat_identity(chart):
    code, out = run(HEAD + "node a = sat_vs_sat(points: [[0, 0], [1, 1]])\ngraph a\n", small(chart))
    assert np.abs(out - code).max() < 1e-6


def test_exposure_is_a_gain_above_the_toe():
    img = np.full((3, 2, 2), 0.18)
    _, out = run(HEAD + "node a = exposure(stops: 1)\ngraph a\n", img)
    assert np.allclose(C.acescct_decode(out), C.acescct_decode(C.to_working(img)) * 2, rtol=1e-9)


def test_mask_and_mix(chart):
    img = small(chart)
    code, full = run(HEAD + "node a = exposure(stops: 1)\ngraph a\n", img)
    _, half = run(HEAD + "node a = exposure(stops: 1)\n  mix 0.5\ngraph a\n", img)
    assert np.allclose(half - code, (full - code) / 2)
    _, none = run(HEAD + "node a = exposure(stops: 1)\n  mask luma(range: 1.4..1.5, soft: 0)\ngraph a\n", img)
    assert np.allclose(none, code)
    _, both = run(HEAD + "node a = exposure(stops: 1)\n  mask luma(range: -0.1..1.5) and not luma(range: 1.4..1.5)\n"
                  "graph a\n", img)
    assert np.allclose(both, full, atol=1e-9)


def test_parallel_is_the_mean_and_layer_blends(chart):
    img = small(chart)
    src = HEAD + "node a = exposure(stops: 1)\nnode b = exposure(stops: -1)\n"
    code, par = run(src + "graph parallel(a, b)\n", img)
    _, a = run(src + "graph a\n", img)
    _, b = run(src + "graph b\n", img)
    assert np.allclose(par, (a + b) / 2)
    _, lay = run(src + "graph layer(a, b, opacity: 0.25)\n", img)
    assert np.allclose(lay, a + 0.25 * (b - a))
    _, dark = run(src + "graph layer(a, b, mode: darken)\n", img)
    assert np.allclose(dark, np.minimum(a, b))



def test_pass_is_the_unchanged_input(chart):
    img = small(chart)
    src = HEAD + "node a = exposure(stops: 1)\nnode b = saturation(amount: 1.5)\n"
    code, a = run(src + "graph a\n", img)
    _, chain = run(src + "graph pass -> a -> pass\n", img)
    assert np.array_equal(chain, a)
    _, half = run(src + "graph parallel(pass, a)\n", img)
    assert np.allclose(half, (code + a) / 2)
    _, scr = run(src + "graph layer(pass, a, mode: screen, opacity: 0.5)\n", img)
    assert np.allclose(scr, code + 0.5 * ((1 - (1 - code) * (1 - a)) - code))
    _, inner = run(src + "graph b -> layer(pass, a, mode: darken)\n", img)
    _, b = run(src + "graph b\n", img)
    _, ab = run(src + "graph b -> a\n", img)
    assert np.allclose(inner, np.minimum(b, ab))  # pass is the layer's own input, not the program's


def test_pass_is_reserved_and_round_trips():
    from cierra.errors import CierraError
    from cierra.format import format_program
    with pytest.raises(CierraError, match="reserved"):
        load(HEAD + "node pass = exposure(stops: 1)\ngraph pass\n")
    prog = load(HEAD + "node a = exposure(stops: 1)\ngraph layer(pass, a, mode: screen)\n")
    assert "graph layer(pass, a, mode: screen)" in format_program(prog)
    assert format_program(load(format_program(prog))) == format_program(prog)


def test_keyframes():
    prog = load(HEAD + "node a = exposure(stops: keys(linear, 10: 0, 20: 1))\ngraph a\n")
    k = prog.nodes["a"].call.args["stops"]
    assert [key_value(k, f) for f in (0, 10, 15, 20, 99)] == [0, 0, 0.5, 1, 1]
    s = load(HEAD + "node a = exposure(stops: keys(step, 0: 0, 5: 1))\ngraph a\n").nodes["a"].call.args["stops"]
    assert [key_value(s, f) for f in (4, 5, 6)] == [0, 1, 1]
    m = load(HEAD + "node a = exposure(stops: keys(smooth, 0: 0, 10: 1, 20: 1))\ngraph a\n").nodes["a"].call.args["stops"]
    vals = [key_value(m, f) for f in range(21)]
    assert all(b >= a for a, b in zip(vals, vals[1:])) and max(vals) <= 1.0  # monotone, no overshoot


def test_determinism_bit_identical(frames):
    src = (HEAD + "node g = grain(amount: 0.3, seed: 42)\nnode b = glow(radius_px: 10)\n"
           "node s = saturation(amount: 1.2)\n  mask skin()\ngraph g -> b -> s\n")
    prog = load(src)
    img = frames["window_0"]
    a = render(prog, img, frame=3)
    b = render(load(src), img.copy(), frame=3)
    assert a.tobytes() == b.tobytes()
    c = render(prog, img, frame=4)
    assert a.tobytes() != c.tobytes()  # grain changes per frame...
    other = load(src.replace("seed: 42", "seed: 43"))
    assert render(other, img, frame=3).tobytes() != a.tobytes()  # ...and per seed


def test_grain_statistics(chart):
    img = np.full((3, 128, 128), 0.18)
    code, out = run(HEAD + "node g = grain(amount: 1, size_px: 1, chroma: 0, seed: 5)\ngraph g\n", img)
    d = (out - code)[0]
    assert abs(d.mean()) < 0.002 and 0.01 < d.std() < 0.03
    assert np.allclose(out[0], out[1])  # chroma 0: same grain in every channel


def test_render_outputs(chart):
    for out in ("aces2(display: srgb)", "aces2(display: rec1886)", "display_encode()", "display_encode(eotf: gamma24)"):
        d = render(load(f"cierra 0.1\noutput {out}\nnode a = exposure()\ngraph a\n"), small(chart))
        assert d.min() >= 0 and d.max() <= 1


def test_gamut_compress_leaves_the_gamut_and_pulls_outliers_in():
    """Reference gamut compression: distance below threshold untouched, the limit distance lands on the boundary."""
    from cierra import colour as C
    from cierra.engine import gamut_compress_linear
    thr, lim = (0.815, 0.803, 0.880), (1.147, 1.264, 1.312)
    inside = np.array([[0.5, 0.4, 0.3], [0.2, 0.25, 0.3], [0.18, 0.18, 0.18]]).T[:, :, None]
    assert np.allclose(gamut_compress_linear(inside, thr, lim, 1.2), inside)
    # a colour whose cyan distance is exactly the limit: red channel = ach * (1 - limit) -> compressed to distance 1
    ach = 0.5
    outside = np.array([ach * (1 - lim[0]), ach, ach])[:, None, None]
    out = gamut_compress_linear(outside, thr, lim, 1.2)
    assert abs(out[0, 0, 0] - 0.0) < 1e-9 and np.allclose(out[1:], outside[1:])
    # through a program, on ACEScct values: finite and in gamut (no channel below zero)
    prog = load("cierra 0.1\noutput acescct()\nnode g = gamut_compress()\ngraph g\n")
    code = C.acescct_encode(np.array([-0.02, 0.4, 0.5])[:, None, None])
    lin = C.acescct_decode(render(prog, code, input="acescct"))
    assert np.isfinite(lin).all() and lin.min() > -1e-9
