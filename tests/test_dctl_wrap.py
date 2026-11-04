"""DCTL wrapper (G5): input decode and output LUT for a Rec.709 (YRGB) project, checked on the gcc harness against
the engine (inputs through OpenColorIO for camera logs, outputs through the exact output transform)."""

import shutil

import numpy as np
import pytest

from cierra import colour as C
from cierra.checker import load, load_file
from cierra.compile.dctl import compile_dctl
from cierra.compile.dctl_wrap import MATRICES, WrapError, output_call
from cierra.engine import grade_working
from conftest import ROOT, frame, small

gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc needed for the DCTL CPU harness")
EXAMPLES = ROOT / "examples"
IDENTITY = "cierra 0.1\noutput acescct()\n"


def run(part, img, interp="tetrahedral"):
    from cierra.compile.dctl_cpu import run_dctl
    return run_dctl(part["dctl"], img, luts=part.get("luts"), interp=interp)


@gcc
@pytest.mark.parametrize("kind", sorted(MATRICES))
def test_camera_log_decode_matches_opencolorio(kind):
    """Each camera curve + fitted matrix against the studio config: grey exact, colours within a few 1e-4 cv."""
    rng = np.random.default_rng(3)
    g = np.linspace(0.0, 1.0, 64)
    img = np.concatenate([np.stack([g, g, g])[:, None, :], rng.uniform(0.0, 1.0, (3, 1, 64))], axis=1)
    part, = compile_dctl(load(IDENTITY), input=kind)
    out, ref = run(part, img), C.input_to_working(img, kind)[0]
    assert np.abs(out[:, 0] - ref[:, 0]).max() < 1e-4
    assert np.abs(out[:, 1] - ref[:, 1]).max() < 4e-3 and np.abs(out[:, 1] - ref[:, 1]).mean() < 5e-4


@gcc
@pytest.mark.parametrize("kind", ["rec1886_photo", "srgb_photo"])
def test_display_video_decode_matches_engine(kind):
    img = np.random.default_rng(4).uniform(-0.05, 1.05, (3, 4, 64))
    part, = compile_dctl(load(IDENTITY), input=kind)
    assert np.abs(run(part, img) - C.input_to_working(img, kind)[0]).max() < 2e-5


@gcc
@pytest.mark.parametrize("interp, mean, p99", [("tetrahedral", 0.05, 0.6), ("trilinear", 0.1, 1.0)])
def test_wrapped_look_matches_engine_on_rec709_video(interp, mean, p99):
    """A look on Rec.709 video, display out through the program's ACES 2.0 output: the whole wrapped DCTL (decode,
    graph, LUT) against the engine's decode, graph and exact output transform."""
    prog = load_file(EXAMPLES / "blockbuster_teal_orange.cierra")
    disp = C.aces2_display(small(frame("crowd"), 8), "rec1886")  # stands in for Rec.709 video frames
    part, = compile_dctl(prog, input="rec1886_photo", output="rec1886", stem="teal")
    assert list(part["luts"]) == ["teal_out_aces2_rec1886.cube"] and 'DEFINE_LUT(cr_out, "teal_out_' in part["dctl"]
    out = run(part, disp, interp)
    ref = C.aces2_display(grade_working(C.input_to_working(disp, "rec1886_photo")[0], prog), "rec1886")
    e = C.delta_e00_display(np.clip(out, 0, 1), ref, "gamma24")
    assert e.mean() < mean and np.percentile(e, 99) < p99


@gcc
@pytest.mark.parametrize("eotf, clip", [("gamma24", "soft"), ("srgb", "hard")])
def test_display_encode_program_and_split_out_part(eotf, clip):
    """A display_encode program, and a split graph whose last node is not in DCTL: the encode becomes its own part,
    written as code (no LUT) and equal to the engine's display_encode within float32 (4e-4 code value)."""
    prog = load("cierra 0.1\nnode a = exposure(stops: 0.3)\nnode b = glow(radius_px: 40)\ngraph a -> b\n"
                f"output display_encode(eotf: {eotf}, clip: {clip})\n")
    parts = compile_dctl(prog, split=True, input="sony_slog3_sgamut3", output="program", stem="p")
    assert [p["name"] for p in parts] == ["part1", "part2", "out"] and parts[1]["dctl"] is None
    assert "cr_in(" in parts[0]["dctl"] and "cr_out(" not in parts[0]["dctl"]
    assert "cr_in(" not in parts[2]["dctl"] and "cr_out(" in parts[2]["dctl"] and "luts" not in parts[2]
    assert "DEFINE_LUT" not in parts[2]["dctl"]
    code = frame("window_0")[:, ::8, ::8]  # bright, saturated: the soft clip works here
    assert np.abs(run(parts[2], code) - C.display_encode(code, eotf, clip)).max() < 1e-3  # float32 under the 1/2.4 power


def test_wrapper_refusals_and_defaults():
    prog = load(IDENTITY)
    with pytest.raises(WrapError, match="no DCTL decode"):
        compile_dctl(prog, input="canon_log3")
    with pytest.raises(WrapError, match="not an SDR Rec.709"):
        compile_dctl(load("cierra 0.1\noutput aces2(display: p3d65)\n"), output="program")
    assert output_call(prog, "program").name == "aces2"  # an LMT program gets the standard output
    assert output_call(load("cierra 0.1\noutput aces2(display: srgb)\n"), "rec1886").args["display"] == "rec1886"
    plain, = compile_dctl(prog)
    assert "cr_in" not in plain["dctl"] and "DEFINE_LUT" not in plain["dctl"] and "luts" not in plain
