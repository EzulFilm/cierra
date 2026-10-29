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
# the curve. Improving that is TODO G8 (per-hue / per-brightness fit), not a reason to loosen the crowd.
FIT_LIMIT = {"crowd": 0.7, "pens": 1.5}


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
