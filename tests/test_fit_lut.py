"""LUT fitting (cierra/fit_lut.py): LUTs made from known programs are fitted back to short, readable programs, and
the residual between each LUT and its fitted program is measured (ΔE00 after the ACES 2.0 output).

How the limits are set. A ΔE00 of about 1 is a just visible difference. The LUTs here are 33-point (as the grading
application exports them; 65 is finer and fits at least as well), so each LUT already differs from the program it was
made from (`baseline`: that program measured against its own LUT the same way). The fit can do no better than the
LUT's own precision where the source program is one the fitted op set can write exactly (a CDL, printer lights),
and there the limits are a few hundredths. A look with narrow hue features or a keyed node is approximated by the
fixed op set (12 hue points 30 degrees apart, one global saturation by luma): there the limits sit just above what
the fit measures today on the six test frames, with a margin of about a quarter, and well below one ΔE00 on average.
"""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.compile.lut import bake, export_luts, write_cube
from cierra.fit_lut import LUTFitError, apply_lut, fit_lut, read_cube, residual
from cierra.format import format_program
from conftest import LOOKS, small

QUALIFIER = load("cierra 0.1\nnode grade = contrast(amount: 1.1, pivot: 0.4)\nnode skin = saturation(amount: 1.5)\n"
                 "  mask qualifier(hue: 120..165, sat: 0.02..0.25, lum: 0.2..0.75)\ngraph grade -> skin\n")
SKIN_HINT = {"nodes": [{"index": 1, "label": "balance", "tools": ["Primaries"]},
                       {"index": 2, "label": "skin", "tools": ["Qualifier", "Primaries"]}]}


@pytest.fixture(scope="module")
def test_frames(frames):
    return [np.clip(C.to_working(small(f, 4)), 0, 1) for f in frames.values()]


def _clean(prog, max_nodes):
    """Checks clean (canonical text loads back to the same text) and stays a handful of nodes."""
    text = format_program(prog)
    assert format_program(load(text)) == text
    assert 1 <= len(prog.nodes) <= max_nodes, text
    return text


def _within(rep, lattice, frames):
    for where, lim in (("lattice", lattice), ("frames", frames)):
        got = rep[where]
        assert all(got[k] <= v for k, v in zip(("mean", "p95", "max"), lim)), (where, got, lim)


@pytest.mark.parametrize("look, lat_lim, frame_lim", [
    # measured (33-point): lattice 0.15 / 0.80 / 2.7, frames 0.06 / 0.20 / 0.70 (the LUT itself: frames 0.06 / 0.23)
    ("cinematic_modern", (0.2, 1.0, 3.5), (0.08, 0.3, 1.0)),
    # measured: lattice 0.16 / 0.68 / 2.5, frames 0.36 / 1.7 / 2.5: the skin qualifier and protect_skin are not in
    # the op set without a keyed node (the hint below tries one; it helps too little here to be kept)
    ("beauty_modern", (0.2, 0.9, 3.5), (0.45, 2.1, 3.2)),
])
def test_fit_the_looks_from_their_cube(tmp_path, test_frames, look, lat_lim, frame_lim):
    prog = load_file(LOOKS / f"{look}.cierra")
    export_luts(prog, tmp_path, look, size=33)
    fitted, rep = fit_lut(tmp_path / f"{look}_ACEScct.cube", frames=test_frames,
                          hints=SKIN_HINT if look == "beauty_modern" else None)
    _clean(fitted, 5)
    _within(rep, lat_lim, frame_lim)
    assert rep["size"] == 33 and rep["domain"] == "acescct"
    # the fit is at the LUT's own precision on pictures where the look is in the op set
    if look == "cinematic_modern":
        base = residual(prog, read_cube(tmp_path / f"{look}_ACEScct.cube")[0], frames=test_frames)
        assert rep["frames"]["mean"] <= base["frames"]["mean"] + 0.02, (rep, base)


def test_a_cdl_comes_back_as_one_cdl(test_frames):
    src = load("cierra 0.1\nnode a = cdl(slope: [1.1, 1.0, 0.92], offset: [0.01, 0, -0.02], power: [1.05, 1, 0.95], "
               "sat: 0.9)\ngraph a\n")
    prog, rep = fit_lut(bake(src, 33), frames=test_frames)
    text = _clean(prog, 1)
    assert "cdl(" in text and rep["primary"] == "parametric"
    a = prog.nodes["cdl"].call.args
    assert np.allclose(a["slope"], (1.1, 1.0, 0.92), atol=2e-3) and abs(a["sat"] - 0.9) < 2e-3
    # measured: lattice 0.0007 / 0.0017 / 0.015, frames 0.0018 / 0.0037 / 0.0072 (the LUT's own rounding)
    _within(rep, (0.01, 0.02, 0.05), (0.01, 0.02, 0.05))


def test_offsets_come_back_as_printer_lights_exactly():
    src = load("cierra 0.1\nnode a = printer_lights(red: 6, green: 6, blue: 6)\n"
               "node b = printer_lights(red: 2, green: -1, blue: -3)\ngraph a -> b\n")
    prog, rep = fit_lut(bake(src, 17))
    assert "printer_lights(" in _clean(prog, 1)
    assert rep["lattice"]["max"] <= 0.01


def test_gains_come_back_as_exposure_and_white_balance():
    src = load("cierra 0.1\nnode a = exposure(stops: 0.5)\nnode b = white_balance(temperature: 0.2, tint: 0.1)\n"
               "graph a -> b\n")
    prog, rep = fit_lut(bake(src, 17))
    _clean(prog, 2)
    args = {n.call.name: n.call.args for n in prog.nodes.values()}
    assert abs(args["exposure"]["stops"] - 0.5) < 0.01
    assert abs(args["white_balance"]["temperature"] - 0.2) < 0.01 and abs(args["white_balance"]["tint"] - 0.1) < 0.01
    assert rep["lattice"]["max"] <= 0.5  # measured 0.31 dE00: the baked lattice rounds the gain's kink at the toe


def test_luma_curve_wheels_and_hue_shift(test_frames):
    src = load("cierra 0.1\nnode a = curve(points: [[0.0729, 0.06], [0.3, 0.28], [0.4136, 0.43], [0.6, 0.64], [1, 1]])"
               "\nnode b = wheels(lift: [-0.01, 0, 0.012], gain: [0.01, 0.003, -0.008])"
               "\nnode c = saturation(amount: 1.15)\nnode d = hue_vs_hue(points: [[150, 8, 40]])\ngraph a -> b -> c -> d\n")
    lut = bake(src, 33)
    prog, rep = fit_lut(lut, frames=test_frames)
    text = _clean(prog, 6)
    assert rep["primary"] == "luma" and "wheels(" in text and "hue_vs_hue(" in text
    # measured: lattice 0.09 / 0.44 / 2.8, frames 0.28 / 1.1 / 1.8, which is the LUT's own error on these frames
    # (the source program against the LUT: 0.28 / 1.1 / 1.9; a hue rotation is a cone at zero chroma that 33
    # points interpolate coarsely)
    _within(rep, (0.15, 0.6, 3.5), (0.35, 1.4, 2.4))
    base = residual(src, lut, frames=test_frames)
    assert rep["frames"]["mean"] <= base["frames"]["mean"] + 0.03


def test_a_qualifier_hint_adds_a_keyed_node(test_frames):
    prog, rep = fit_lut(bake(QUALIFIER, 33), hints=SKIN_HINT, frames=test_frames)
    text = _clean(prog, 5)
    assert rep["keyed"] == ["skin"] and "mask qualifier(" in text
    keyed = [n for n in prog.nodes.values() if n.mask is not None]
    assert len(keyed) == 1 and keyed[0].call.name == "saturation" and keyed[0].call.args["amount"] > 1.2
    # measured with the keyed node: lattice 0.21 / 0.93 / 19 (a hard qualifier edge on saturated colours);
    # frames 0.51 / 2.2 / 8.2 on the CC0 frames (2026-10-28; 0.54 / 2.3 / 4.7 on the earlier frames): the worst pixel
    # is a saturated red on the pot, at the qualifier's edge
    trials = rep["keyed_trials"]
    assert min(v for k, v in trials.items() if k != "none") < 0.6 * trials["none"]
    _within(rep, (0.3, 1.2, 25.0), (0.7, 3.0, 9.0))


def test_a_linear_domain_lut(tmp_path):
    """A LUT over ACEScg (linear AP1, 0..1) back to ACEScg: the domain is a parameter."""
    from cierra.compile.lut import lattice
    from cierra.engine import grade_working
    src = load("cierra 0.1\nnode a = exposure(stops: -0.5)\ngraph a\n")
    code = C.acescct_encode(lattice(17))
    out = C.acescct_decode(grade_working(code.reshape(3, -1, 1), src, px_scale=1.0).reshape(code.shape))
    write_cube(tmp_path / "lin.cube", out, "linear")
    prog, rep = fit_lut(tmp_path / "lin.cube", input="acescg")
    _clean(prog, 1)
    [node] = prog.nodes.values()
    assert node.call.name == "exposure" and abs(node.call.args["stops"] + 0.5) < 0.01 and prog.input == "acescg"
    assert rep["lattice"]["max"] <= 0.05 and rep["domain"] == "acescg"


def test_cube_reader_and_refusals(tmp_path):
    lut = bake(load("cierra 0.1\nnode a = saturation(amount: 1.2)\ngraph a\n"), 5, fit_iters=0)
    write_cube(tmp_path / "a.cube", lut, "a")
    back, dmin, dmax = read_cube(tmp_path / "a.cube")
    assert np.allclose(back, lut, atol=1e-6) and np.allclose(dmin, 0) and np.allclose(dmax, 1)
    pts = np.random.default_rng(1).uniform(0, 1, (3, 50))
    from cierra.compile.lut import apply_file_lut
    assert np.allclose(apply_lut(back, pts), apply_file_lut(tmp_path / "a.cube", pts), atol=1e-5)  # tetrahedral
    (tmp_path / "1d.cube").write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    (tmp_path / "short.cube").write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")
    (tmp_path / "nan.cube").write_text("LUT_3D_SIZE 2\n" + "nan 0 0\n" * 8)
    for name, why in (("1d", "1D LUT"), ("short", "needs 8"), ("nan", "non-finite")):
        with pytest.raises(LUTFitError, match=why):
            read_cube(tmp_path / f"{name}.cube")
    with pytest.raises(LUTFitError, match="unknown LUT domain"):
        fit_lut(lut, input="rec2020_gamma")
    with pytest.raises(LUTFitError, match=r"\(3, N, N, N\)"):
        fit_lut(np.zeros((3, 4, 4)))


def test_import_lut_command(tmp_path, capsys):
    from cierra.__main__ import main
    src = load("cierra 0.1\nnode a = contrast(amount: 1.2, pivot: 0.4)\nnode b = saturation(amount: 0.8)\n"
               "graph a -> b\n")
    write_cube(tmp_path / "grade.cube", bake(src, 17), "grade")
    assert not main(["import-lut", str(tmp_path / "grade.cube"), str(tmp_path / "out.cierra")])
    prog = load_file(tmp_path / "out.cierra")
    assert [n.call.name for n in prog.nodes.values()][0] == "contrast"
    assert '"lattice"' in capsys.readouterr().out
    (tmp_path / "bad.cube").write_text("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
    assert main(["import-lut", str(tmp_path / "bad.cube"), str(tmp_path / "x.cierra")]) == 1
    assert "1D LUT" in capsys.readouterr().err
