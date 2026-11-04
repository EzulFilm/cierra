"""Reference fitting (cierra/fit.py): a reference look becomes readable curve / white balance / saturation values."""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.engine import render
from cierra.fit import _keys, _smooth, detect_shots, fit_reference, fit_sequence
from cierra.format import format_program
from conftest import LOOKS, frame, small

LOOK = load("cierra 0.1\nnode w = white_balance(temperature: 0.15, tint: 0.05)\n"
            "node c = curve(points: [[0.0729, 0.0729], [0.3, 0.27], [0.4136, 0.43], [0.6, 0.64], [1, 1]])\n"
            "node s = saturation(amount: 0.85)\ngraph w -> c -> s\n")


# Mean dE00 the fit must reach per frame. Measured on the CC0 frames (2026-10-28): crowd 0.61, pens 1.40 (from 4.7
# and 5.0 ungraded). A frame with less variety (a green felt) pins the curve and white balance less well. On the
# window frame (1.23) the fit reads saturation 0.96 for 0.85: too little colour in it to separate saturation from
# the curve. The per-hue / per-brightness stages (2026-10-30) take pens to 0.77 (a hue_vs_lum: the frame is mostly
# one hue, whose tone the luma curve misses); crowd keeps its three-node program.
FIT_LIMIT = {"crowd": 0.7, "pens": 1.5}
# hue_vs_hue / hue_vs_sat on hues the crowd frame shows (chroma share 0: 15 %, 120: 38 %, 180: 35 %)
HUE_LOOK = load("cierra 0.1\nnode h = hue_vs_hue(points: [[180, 12, 60]])\n"
                "node s = hue_vs_sat(points: [[120, 1.35, 60], [0, 0.75, 60]])\ngraph h -> s\n")


@pytest.mark.parametrize("name", list(FIT_LIMIT))
def test_fit_recovers_a_known_look(name):
    img = small(frame(name), 4)
    ref = render(LOOK, img)
    prog, info = fit_reference(img, ref)
    de0 = C.delta_e00_display(render(load("cierra 0.1\n"), img), ref, "srgb").mean()
    de1 = C.delta_e00_display(render(prog, img), ref, "srgb").mean()
    assert de1 < FIT_LIMIT[name] and de1 < de0 / 3.5, (de0, de1)
    assert abs(info["temperature"] - 0.15) < 0.08 and abs(info["saturation"] - 0.85) < 0.1
    assert format_program(load(format_program(prog))) == format_program(prog)


def test_fit_of_a_simple_look_is_unchanged_by_the_hue_stages():
    img = small(frame("crowd"), 4)
    ref = render(LOOK, img)
    prog, info = fit_reference(img, ref)
    assert info["stages"] == [] and info["de00"][0] == info["de00"][1]
    assert format_program(prog) == format_program(fit_reference(img, ref, fine=False)[0])


def _curve(prog, op):
    n = [n for n in prog.nodes.values() if n.call.name == op]
    return {round(float(p[0])): float(p[1]) for p in n[0].call.args["points"]} if n else {}


def test_fit_recovers_known_hue_curves():
    img = small(frame("crowd"), 4)
    ref = render(HUE_LOOK, img)
    prog, info = fit_reference(img, ref)
    de0, de1 = info["de00"]  # measured 1.44 -> 0.35 (2026-10-30)
    assert de0 > 1.2 and de1 < 0.45 and info["stages"] == ["match_hue", "match_hue_sat"], info
    assert de1 == pytest.approx(C.delta_e00_display(render(prog, img), ref, "srgb").mean(), abs=1e-3)
    shift = _curve(prog, "hue_vs_hue")
    assert abs(shift[180] - 12) < 1.5 and all(abs(v) < 4 for h, v in shift.items() if h != 180), shift
    # gain by hue: the fitted saturation times hue_vs_sat (the saturation node takes the mean)
    gain = {h: info["saturation"] * _curve(prog, "hue_vs_sat").get(h, 1.0) for h in range(0, 360, 60)}
    want = {0: 0.75, 120: 1.35}
    assert all(abs(g - want.get(h, 1.0)) < (0.1 if h in (0, 120) else 0.12) for h, g in gain.items()), gain


@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern"])
def test_fit_approaches_the_shipped_looks(look):
    prog_look = load_file(LOOKS / f"{look}.cierra")
    for name in ("pens", "crowd"):
        img = small(frame(name), 4)
        ref = render(prog_look, img)
        prog, _ = fit_reference(img, ref)
        de0 = C.delta_e00_display(render(load("cierra 0.1\n"), img), ref, "srgb").mean()
        de1 = C.delta_e00_display(render(prog, img), ref, "srgb").mean()
        assert de1 < 0.6 * de0, (look, name, de0, de1)


def test_sequence_keeps_a_steady_shot_steady_and_isolates_a_flash():
    crowd = small(frame("crowd"), 4)
    w = crowd.shape[2] - 4
    steady = [crowd[:, :, i * 2:i * 2 + w] for i in range(3)]  # a slow pan: three crops 2 px apart
    ref = render(load_file(LOOKS / "beauty_modern.cierra"), steady[1])
    seq = [(0, steady[0]), (12, steady[1]), (24, steady[2])]
    progs = fit_sequence(seq, ref)
    assert list(progs) == [0] and "keys(" not in format_program(progs[0])  # nothing moved more than a step
    flash = seq[:2] + [(18, seq[1][1] * 2.0)] + seq[2:]
    assert detect_shots(flash) == [0, 0, 1, 2]
    progs = fit_sequence(flash, ref)
    p0 = progs[0]
    for f, im in seq[:2]:
        assert C.delta_e00_display(render(p0, im, frame=f), ref if f == 12 else render(
            load_file(LOOKS / "beauty_modern.cierra"), im), "srgb").mean() < 1.0


def test_smoothing_and_keys():
    assert list(_smooth([1, 1, 5, 1, 1], 5)) == [1, 1, 1, 1, 1]  # a one-frame outlier is gone
    assert _keys([0, 1, 2], np.array([0.5, 0.505, 0.5]), 0.01) == "0.5017"
    assert _keys([0, 10, 20], np.array([0.0, 0.05, 0.1]), 0.01).startswith("keys(linear, 0: 0, 10: 0.05, 20: 0.1")


def test_cli_fit(tmp_path):
    from PIL import Image

    from cierra.__main__ import main
    img = small(frame("crowd"), 4)
    np.save(tmp_path / "in.npy", img)
    ref = (np.clip(render(LOOK, img), 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(ref).save(tmp_path / "ref.png")
    main(["fit", str(tmp_path / "in.npy"), str(tmp_path / "ref.png"), str(tmp_path / "out.cierra")])
    assert "match_tone = curve" in (tmp_path / "out.cierra").read_text()


def test_cli_fit_records_how_the_image_was_read(tmp_path, capsys):
    """--range changes the picture the fit sees, so the fit output says which range (and input) it used."""
    import json

    import tifffile

    from cierra.__main__ import main
    img = (small(frame("crowd"), 4).clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    tifffile.imwrite(tmp_path / "in.tif", img, photometric="rgb")
    tifffile.imwrite(tmp_path / "ref.tif", img, photometric="rgb")
    capsys.readouterr()
    main(["fit", str(tmp_path / "in.tif"), str(tmp_path / "ref.tif"), str(tmp_path / "out.cierra"),
          "--input", "srgb_photo", "--range", "video"])
    info = json.loads(capsys.readouterr().out)
    assert info["range"] == "video" and info["input"] == "srgb_photo"


def test_fit_refuses_a_reference_of_another_size():
    """The per-hue / per-brightness stages pair pixels, so a reference of another size is a plain CierraError (one
    line on the command line), not a broadcasting crash; without them (as fit_sequence fits) any size works."""
    from cierra.errors import CierraError
    img = np.full((3, 4, 6), 0.18)
    with pytest.raises(CierraError, match="same size"):
        fit_reference(img, np.full((3, 4, 5), 0.5))
    fit_reference(img, np.full((3, 4, 5), 0.5), fine=False)


# one look per remaining stage, on the crowd frame (measured 2026-11-01: hue_vs_lum 0.83 -> 0.57 dE00, offset 0.039
# found at hue 120 for 0.05; lum_vs_sat 0.44 -> 0.10, gain 0.77 at luma 0.2 and 1.16 at 0.6)
@pytest.mark.parametrize("text,stage", [
    ("node l = hue_vs_lum(points: [[120, 0.05, 60]])", "match_hue_lum"),
    ("node l = lum_vs_sat(points: [[0.15, 0.7], [0.41, 1.0], [0.75, 1.3]])", "match_lum_sat"),
])
def test_fit_recovers_brightness_stages(text, stage):
    img = small(frame("crowd"), 4)
    ref = render(load(f"cierra 0.1\n{text}\ngraph l\n"), img)
    prog, info = fit_reference(img, ref)
    de0, de1 = info["de00"]
    assert info["stages"] == [stage] and de1 < 0.75 * de0, info
    if stage == "match_hue_lum":
        off = _curve(prog, "hue_vs_lum")
        assert 0.025 < off[120] < 0.06 and all(abs(v) < 0.02 for h, v in off.items() if h != 120), off
    else:
        pts = [n for n in prog.nodes.values() if n.call.name == "lum_vs_sat"][0].call.args["points"]
        g = {round(float(p[0]), 1): float(p[1]) for p in pts}
        assert g[0.2] < 0.85 and g[0.6] > 1.1, g
