"""Regression tests for the bugs found in the code review of 2026-11-04 (one test per fix)."""

import numpy as np
import pytest

from cierra import backend
from cierra.__main__ import ImageInputError, read_image, write_png


def test_upper_case_npy_loads(tmp_path):
    a = np.random.default_rng(1).random((3, 2, 2))
    np.save(tmp_path / "f.npy", a)
    (tmp_path / "f.npy").rename(tmp_path / "F.NPY")
    assert np.array_equal(read_image(tmp_path / "F.NPY"), a)


def test_tiff_of_unsupported_shape_is_one_line_not_a_traceback(tmp_path, monkeypatch):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "x.tif", np.zeros((2, 2), np.uint8))
    monkeypatch.setattr(tifffile, "imread", lambda *a, **k: np.zeros((2, 2, 2, 2), np.uint8))  # e.g. a volume
    with pytest.raises(ImageInputError):
        read_image(tmp_path / "x.tif")


def test_png_of_nan_and_inf_is_black_and_white(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    d = np.array([[[np.nan, np.inf, -np.inf, 0.5]]] * 3)
    with np.errstate(invalid="raise"):
        write_png(tmp_path / "o.png", d)
    assert np.asarray(Image.open(tmp_path / "o.png"))[0, :, 0].tolist() == [0, 255, 0, 128]


def test_numpy_input_does_not_load_cupy(monkeypatch):
    monkeypatch.setattr(backend, "_CUPY", {})
    assert backend.xp(np.zeros(2)) is np and backend.to_host(np.ones(1))[0] == 1
    assert backend._CUPY == {}


@pytest.mark.parametrize("node, worst", [("saturation(amount: 0)", 0.05),  # black and white: was 27.7 dE00 mean
                                         ("lum_vs_sat(points: [[0, 1], [0.41, 0], [1, 1]])", 1.0)])  # was 11.2, NaN
def test_lut_that_removes_colour_fits(node, worst):
    from cierra.checker import load
    from cierra.compile.lut import bake
    from cierra.fit_lut import fit_lut
    with np.errstate(invalid="raise", divide="raise"):
        _, rep = fit_lut(bake(load(f"cierra 0.1\nnode a = {node}\ngraph a\n"), 17))
    assert rep["lattice"]["mean"] <= worst


def test_cube_with_a_nan_domain_is_refused():
    from cierra.fit_lut import LUTFitError, _parse_cube
    rows = "\n".join(f"{r} {g} {b}" for b in (0, 1) for g in (0, 1) for r in (0, 1))
    with pytest.raises(LUTFitError):
        _parse_cube(f"LUT_3D_SIZE 2\nDOMAIN_MIN nan nan nan\nDOMAIN_MAX 1 1 1\n{rows}\n")


def test_flat_solve_keeps_the_start_value():
    from cierra.fit import _solve
    assert _solve(lambda x: 0.0, 0.0, 0.6, 1.4, x0=1.0) == 1.0


def test_fit_sequence_refuses_shots_of_the_wrong_length():
    from cierra.fit import fit_sequence
    im = np.full((3, 4, 4), 0.18)
    with pytest.raises(ValueError):
        fit_sequence([(0, im), (1, im)], im, shots=[0])


def test_history_diff_sees_requires_and_ignores_written_defaults():
    from cierra.checker import load
    from cierra.history import diff
    a = load("cierra 0.1\nnode s = saturation()\ngraph s\n")
    b = load("cierra 0.1\nnode s = saturation(amount: 1)\ngraph s\n")
    assert diff(a, b) == ["no change"]
    c = load("cierra 0.1\nrequires frames\nnode s = saturation()\ngraph s\n")
    assert any(line.startswith("requires") for line in diff(a, c))


def test_dated_model_id_is_priced_as_its_model():
    from types import SimpleNamespace
    from cierra.llm import usage_record
    u = SimpleNamespace(input_tokens=0, output_tokens=1_000_000)
    assert usage_record("claude-opus-5-5-20261015", u, 0)["cost_usd"] == usage_record("claude-opus-5-5", u, 0)["cost_usd"]


def test_side_effects_without_regions_skip_regional_axes():
    from cierra.notes import probes_for, side_axes, side_effects
    d = np.full((3, 8, 8), 0.4)
    rows, _ = side_effects(side_axes(probes_for("warmer faces")), d, d + 0.01)
    assert rows and all(r["region"] == "all" for r in rows)


def test_model_server_redirect_to_plain_http_elsewhere_is_refused():
    import http.server
    import threading
    from cierra.llm import OpenWeightProvider

    class Redirect(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), Redirect)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    try:
        with pytest.raises(ValueError, match="CIERRA_LLM_URL"):
            OpenWeightProvider(url=f"http://127.0.0.1:{srv.server_port}/v1", timeout=5).complete("s", "u")
    finally:
        srv.server_close()


def test_preview_of_a_one_row_image_is_not_empty():
    from cierra.llm import preview
    assert preview(np.zeros((3, 1, 1000))).shape[1:] == (1, 1000)


def test_error_inside_a_provider_is_not_retried():
    from cierra.llm import _ask
    calls = []

    class P:
        def complete(self, system, user, temperature=None):
            calls.append(temperature)
            raise TypeError("a bug inside the call")

    class Old:
        def complete(self, system, user):
            return "ok"

    with pytest.raises(TypeError):
        _ask(P(), "s", "u", temperature=0.7)
    assert calls == [0.7] and _ask(Old(), "s", "u", temperature=0.7) == "ok"


def test_unpaired_surrogate_is_a_parse_error():
    from cierra.errors import ParseError
    from cierra.parser import parse
    with pytest.raises(ParseError):
        parse('cierra 0.1\nname "' + chr(0xD800) + '"\n')


def test_damage_measure_survives_a_nan_pixel():
    from cierra.qc import measure
    d = np.full((3, 32, 32), 0.5)
    d[:, 3, 3] = np.nan
    assert np.isfinite(measure(d)["milky"])


def test_track_step_zero_is_refused_in_one_line():
    from cierra.checker import load
    from cierra.track import TrackError, track_program
    prog = load("cierra 0.1\nnode a = exposure(stops: 1)\n  mask window(centre: [0.5, 0.5], radius: [0.1, 0.1])\n"
                "graph a\n")
    with pytest.raises(TrackError, match="step"):
        track_program(prog, "clip.mov", "a", 0, 10, step=0)


def test_output_lut_name_cannot_break_the_dctl_string():
    from cierra.checker import load
    from cierra.compile.dctl import compile_dctl
    prog = load("cierra 0.1\noutput aces2(display: rec1886)\nnode a = exposure(stops: 0.5)\ngraph a\n")
    part, = compile_dctl(prog, input="rec1886_photo", output="rec1886", stem='x"); //\n../y')
    name, = part["luts"]
    assert all(ch.isalnum() or ch in "._-" for ch in name) and not name.startswith(".")
    assert f'DEFINE_LUT(cr_out, "{name}");' in part["dctl"]


def test_host_still_with_nan_is_not_verified():
    from cierra.hosts.grading_app import verify
    ref = np.zeros((3, 4, 4))
    img = ref.copy()
    img[0, 0, 0] = np.nan
    ok = {"ok": True, "readback": "Cierra/x.dctl", "stills": [{"ok": True}]}
    assert verify(ok, [img], [ref])["status"] == "unverified"


def test_apply_script_takes_30_fps_timecodes_and_refuses_rooted_luts():
    from cierra.hosts.grading_app import HostError, apply_script, tc_to_frames
    apply_script("p", "cierra_v1", "Cierra/x.dctl", [("00:00:00:29", "C:/out/a.tif")])
    for lut in ("/tmp/x.dctl", "//server/share/x.dctl"):
        with pytest.raises(HostError):
            apply_script("p", "cierra_v1", lut, [])
    with pytest.raises(HostError):
        tc_to_frames("00:00:60:00", 24)


def test_tuner_counts_a_nan_measure_as_unmet():
    from cierra.tuner import _deficits
    row = [None, "warmth", None, None, None, None, 1, 0.1]
    assert _deficits([row], {"warmth": float("nan")}, {"warmth": 0.0})["warmth"] > 100
    assert _deficits([row], {"warmth": float("nan")}, {"warmth": float("nan")})["warmth"] == 0.0


def test_bound_from_a_look_with_one_keyed_component():
    from cierra.checker import load
    from cierra.tuner import bound
    look = load("cierra 0.1\nnode w = wheels(lift: [keys(linear, 0: 0.01, 10: 0.02), 0, 0])\ngraph w\n")
    prog = load("cierra 0.1\nnode w = wheels(lift: [0.3, 0, 0])\ngraph w\n")
    out, moved = bound(prog, start=look)
    assert out.nodes["w"].call.args["lift"][0] < 0.3 and moved
