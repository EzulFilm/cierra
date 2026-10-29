"""Compiled targets against the reference engine: DCTL (run on the CPU through gcc) and CLF / .cube (applied by
OpenColorIO). Requirement: ΔE00 <= 0.5 after the program's output transform, on every test frame."""

import shutil

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.compile.dctl import DCTLError, compile_dctl
from cierra.compile.lut import LUTError, apply_file_lut, export_luts
from cierra.engine import grade_working
from conftest import LOOKS, small

KITCHEN_SINK = """cierra 0.1
name "every DCTL op"
output aces2(display: srgb)
node exp = exposure(stops: 0.2)
node wb = white_balance(temperature: 0.1, tint: -0.05)
node pl = printer_lights(red: 1, blue: -2)
node wh = wheels(lift: [0.01, 0, -0.01], gamma: [0, 0.005, 0], gain: [-0.01, 0, 0.01], master_gamma: 0.01)
node cd = cdl(slope: [1.05, 1, 0.95], offset: [0.01, 0, -0.01], power: [1, 1.05, 1], sat: 0.95)
node co = contrast(amount: 1.1, pivot: 0.35)
node sa = saturation(amount: 1.1, protect_skin: 0.8)
node cl = curve(points: [[0.0729, 0.0729], [0.25, 0.24], [0.4136, 0.42], [0.7, 0.71], [1, 1]])
node cr = curve(points: [[0, 0], [0.5, 0.52], [1, 1]], channel: red)
node cg = curve(points: [[0, 0.01], [0.5, 0.5], [1, 0.98]], channel: rgb)
node cdisp = curve(points: [[0, 0], [0.5, 0.48], [1, 1]], space: display)
node hh = hue_vs_hue(points: [[206, -12, 40], [350, 8]])
node hs = hue_vs_sat(points: [[141, 1.1, 30], [300, 0.9]])
node hl = hue_vs_lum(points: [[350, -0.02, 50]])
node ls = lum_vs_sat(points: [[0.2, 0.9], [0.45, 1.05], [0.65, 0.9]])
node ss = sat_vs_sat(points: [[0, 0], [0.05, 0.05], [0.12, 0.1], [0.3, 0.2]])
node q = saturation(amount: 1.2)
  mask qualifier(hue: 330..30, sat: 0.01..0.5, lum: 0.2..0.8, soft: 0.05) or skin(width: 30)
node lm = exposure(stops: -0.3)
  mask not luma(range: 0.1..0.5, soft: 0.05)
  mix 0.7
node win = exposure(stops: 0.3)
  mask window(centre: [0.4, 0.6], radius: [0.3, 0.2], soft: 0.4)
node vg = vignette(amount: 0.2, roundness: 0.5)
node gr = grain(amount: 0.2, size_px: 2, chroma: 0.5, seed: 11)
node pa = exposure(stops: 0.5)
node pb = saturation(amount: 0.5)
node la = exposure(stops: 1)
node lb = contrast(amount: 1.3)
node key = exposure(stops: keys(linear, 0: 0, 10: 0.4))
graph exp -> wb -> pl -> wh -> cd -> co -> sa -> cl -> cr -> cg -> cdisp -> hh -> hs -> hl -> ls -> ss -> q -> lm -> win -> vg -> gr -> parallel(pa, pb) -> layer(la, lb, mode: multiply, opacity: 0.3) -> layer(la, lb, mode: add, opacity: 0.2) -> layer(pa, pb, mode: lighten, opacity: 0.5) -> key
"""

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc needed for the DCTL CPU harness")


def _de(prog, a_code, b_code):
    """ΔE00 of two ACEScct results after the program's output, decoded with that output's EOTF."""
    return C.delta_e00_display(C.output_from_working(a_code, prog.output), C.output_from_working(b_code, prog.output),
                               C.output_eotf(prog.output) or "srgb")


def _inputs(frames):
    return {k: C.to_working(small(v, 2)) for k, v in frames.items()}


@needs_gcc
def test_dctl_kitchen_sink_matches_reference(frames):
    from cierra.compile.dctl_cpu import run_dctl
    prog = load(KITCHEN_SINK)
    [part] = compile_dctl(prog, frame=5)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, frame=5, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, run_dctl(part["dctl"], code), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


@needs_gcc
@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern", "teal_night"])
def test_dctl_looks(frames, look):
    """Whole looks; a split look runs its DCTL parts with the refused nodes done by the reference engine between
    them (what a colourist does with the host's own tool for that node)."""
    from cierra.compile.dctl_cpu import run_dctl
    from cierra.engine import make_ctx, run_graph
    prog = load_file(LOOKS / f"{look}.cierra")
    parts = compile_dctl(prog, split=True)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, px_scale=code.shape[-1] / prog.reference_width)
        ctx = make_ctx(prog, code.shape[-1])
        out = code
        for p in parts:
            out = run_dctl(p["dctl"], out) if p["dctl"] else run_graph(out, p["graph"], prog, ctx)
        de = _de(prog, out, ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


def test_dctl_refuses_spatial():
    prog = load("cierra 0.1\nnode a = glow()\ngraph a\n")
    with pytest.raises(DCTLError, match="spatial"):
        compile_dctl(prog)
    parts = compile_dctl(load("cierra 0.1\nnode a = exposure()\nnode b = glow()\nnode c = contrast()\n"
                              "graph a -> b -> c\n"), split=True)
    assert [p["dctl"] is None for p in parts] == [False, True, False]


@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern"])
def test_lut_exports_match_reference(frames, look, tmp_path):
    prog = load_file(LOOKS / f"{look}.cierra")
    files = export_luts(prog, tmp_path, look, size=65)
    assert len(files) == 3
    for name, code in _inputs(frames).items():
        code = np.clip(code, 0, 1)  # a LUT's domain is 0..1 ACEScct (linear ~0 .. 222)
        ref = grade_working(code, prog, px_scale=1.0)
        for ext in ("cube", "clf"):
            out = apply_file_lut(tmp_path / f"{look}_ACEScct.{ext}", code)
            de = _de(prog, out, ref)
            # every pixel: the lattice is fitted for tetrahedral interpolation (worst 0.48 on beauty_modern, 0.63 when
            # only sampled at the nodes)
            assert de.mean() <= 0.1 and de.max() <= 0.5, (name, ext, de.mean(), de.max())
            cv = np.abs(out - ref)  # the LMT's own output is ACEScct: a numerical tolerance as well
            assert np.quantile(cv, 0.999) <= 0.002 and cv.max() <= 0.005, (name, ext, np.quantile(cv, 0.999), cv.max())
        disp_cube = [f for f in files if "_to_" in f][0]
        out = apply_file_lut(tmp_path / disp_cube, code)
        # graph + output LUT: compare final outputs directly (the output transform is inside the LUT already)
        de = C.delta_e00_display(np.clip(out, 0, 1), C.output_from_working(ref, prog.output),
                                 C.output_eotf(prog.output))
        assert de.mean() <= 0.3, (name, "display cube", de.mean(), de.max())


def test_lut_refuses_spatial_and_says_why(tmp_path):
    prog = load_file(LOOKS / "teal_night.cierra")
    with pytest.raises(LUTError, match="spatial"):
        export_luts(prog, tmp_path, "teal_night")


def test_lut_fit_beats_node_sampling():
    """The fitted lattice interpolates the look better than the lattice sampled at its nodes (beauty_modern's narrow
    skin qualifier is the hard case), measured at points between the nodes."""
    from cierra.compile.lut import _tetra, bake
    prog = load_file(LOOKS / "beauty_modern.cierra")
    size = 17
    rng = np.random.default_rng(3)
    pts = rng.uniform(0.3, 0.7, size=(3, 20000))
    ref = grade_working(pts.reshape(3, 100, 200), prog, px_scale=1.0).reshape(3, -1)
    idx, w = _tetra(size, pts)
    err = {}
    for iters in (0, 20):
        lat = bake(prog, size, fit_iters=iters).reshape(3, -1)
        err[iters] = np.abs(sum(w[k] * lat[:, idx[k]] for k in range(4)) - ref)
    assert np.sqrt((err[20] ** 2).mean()) < 0.8 * np.sqrt((err[0] ** 2).mean())
    assert err[20].max() < err[0].max()


@needs_gcc
@pytest.mark.parametrize("mask", ["window(centre: [0.4, 0.5], radius: [0.35, 0.15], angle: 30, soft: 0.4)",
                                  "graduated(centre: [0.5, 0.4], angle: 100, falloff: [0.1, 0.3])",
                                  "graduated(angle: -45, falloff: [0, 0.2]) and not window(angle: 120)"])
def test_dctl_rotated_and_graduated_masks(chart, mask):
    """Position masks in a rotated frame (spec §6): the DCTL computes the same matte as the engine, on a non-square
    frame (the rotation works in frame widths, so the aspect matters)."""
    from cierra.compile.dctl_cpu import run_dctl
    prog = load(f"cierra 0.1\noutput acescct()\nnode a = exposure(stops: 1)\n  mask {mask}\ngraph a\n")
    [part] = compile_dctl(prog)
    code = C.to_working(small(chart, 2))
    ref = grade_working(code, prog, px_scale=1.0)
    out = run_dctl(part["dctl"], code)
    assert np.abs(out - ref).max() < 1e-4, np.abs(out - ref).max()


def test_graduated_mask_direction():
    from cierra.engine import mask_value
    prog = load("cierra 0.1\nnode a = exposure()\n  mask graduated(falloff: [0.05, 0.05])\ngraph a\n")
    m = mask_value(np.zeros((3, 40, 60)), prog.nodes["a"].mask, {"frame": 0, "px_scale": 1.0, "aux": {}})
    assert m[0].mean() > 0.99 and m[-1].mean() < 0.01  # angle 90: 1 at the top (a sky grad), 0 at the bottom
    rot = load("cierra 0.1\nnode a = exposure()\n  mask window(radius: [0.4, 0.1], angle: 90)\ngraph a\n")
    m = mask_value(np.zeros((3, 61, 61)), rot.nodes["a"].mask, {"frame": 0, "px_scale": 1.0, "aux": {}})
    assert m[14, 30] > 0.9 and m[30, 8] < 0.1  # a wide ellipse turned 90 deg stands upright


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1)])
def test_rotated_masks_on_one_pixel_frames_are_finite(shape):
    """A one-pixel row or column has no aspect (k = 1, spec §6): no 0 / 0 in a rotated window, in the engine and in
    the DCTL (the WGSL shares the generator)."""
    prog = load("cierra 0.1\noutput acescct()\nnode s = saturation(amount: 0)\n  mask window(angle: 25)\n"
                "node g = exposure(stops: 1)\n  mask graduated(angle: 30)\ngraph s -> g\n")
    code = np.full((3, *shape), 0.45) + np.array([0.05, 0.0, -0.05])[:, None, None]
    ref = grade_working(code, prog)
    assert np.isfinite(ref).all()
    if shutil.which("gcc"):
        from cierra.compile.dctl_cpu import run_dctl
        out = run_dctl(compile_dctl(prog)[0]["dctl"], code)
        assert np.isfinite(out).all() and np.abs(out - ref).max() < 1e-4


@pytest.mark.parametrize("mode", ["screen", "overlay", "subtract", "difference"])
def test_dctl_blend_modes(chart, mode):
    if not shutil.which("gcc"):
        pytest.skip("gcc needed for the DCTL CPU harness")
    from cierra.compile.dctl_cpu import run_dctl
    prog = load("cierra 0.1\noutput acescct()\nnode b = exposure(stops: 0.5)\nnode t = saturation(amount: 1.4)\n"
                f"graph layer(b, t, mode: {mode}, opacity: 0.7)\n")
    code = C.to_working(small(chart))
    ref = grade_working(code, prog)
    out = run_dctl(compile_dctl(prog)[0]["dctl"], code)
    assert np.abs(ref - code).max() > 0.01 and np.abs(out - ref).max() < 1e-4


PASS_PROGRAM = ("cierra 0.1\noutput acescct()\nnode lift = exposure(stops: 0.3)\nnode glow = saturation(amount: 1.6)\n"
                "graph lift -> layer(pass, glow, mode: screen, opacity: 0.6) -> parallel(pass, lift) -> pass\n")

# Key routing (G10): a key statement, a node's carried key (made on that node's input before it grades), the outside
# of it, a key read inside a parallel branch, the key mixer's + / - and gain / offset.
KEYS_PROGRAM = ("cierra 0.1\noutput acescct()\nkey sky = graduated(centre: [0.5, 0.4], angle: 90, falloff: [0.1, 0.2])\n"
                "node face = saturation(amount: 1.8)\n  mask qualifier(hue: 100..180, sat: 0.01..0.2, soft: 0.1)\n"
                "node rest = saturation(amount: 0.6)\n  mask not key(face) - key(sky, gain: 0.5)\n"
                "node cool = white_balance(temperature: -0.3)\n"
                "  mask key(sky) + key(face, gain: 2, offset: -0.5)\n"
                "graph face -> parallel(rest, cool)\n")


@pytest.mark.parametrize("src", [PASS_PROGRAM, KEYS_PROGRAM], ids=["pass", "keys"])
def test_dctl_pass_layer_bottom(chart, src):
    if not shutil.which("gcc"):
        pytest.skip("gcc needed for the DCTL CPU harness")
    from cierra.compile.dctl_cpu import run_dctl
    prog = load(src)
    code = C.to_working(small(chart))
    ref = grade_working(code, prog)
    out = run_dctl(compile_dctl(prog)[0]["dctl"], code)
    assert np.abs(ref - code).max() > 0.01 and np.abs(out - ref).max() < 1e-4


@pytest.mark.parametrize("mask", ["box(centre: [0.45, 0.55], size: [0.3, 0.2], angle: 20, roundness: 0.3, soft: 0.15)",
                                  "polygon(points: [[0.1, 0.2], [0.8, 0.15], [0.5, 0.5], [0.85, 0.85], [0.15, 0.8]], "
                                  "soft: 0.03)"])
def test_dctl_box_and_polygon(chart, mask):
    if not shutil.which("gcc"):
        pytest.skip("gcc needed for the DCTL CPU harness")
    from cierra.compile.dctl_cpu import run_dctl
    prog = load(f"cierra 0.1\noutput acescct()\nnode a = exposure(stops: 1)\n  mask {mask}\ngraph a\n")
    code = C.to_working(small(chart))
    ref = grade_working(code, prog)
    out = run_dctl(compile_dctl(prog)[0]["dctl"], code)
    assert np.abs(ref - code).max() > 0.01 and np.abs(out - ref).max() < 1e-4
