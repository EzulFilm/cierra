"""Damage checks (cierra/qc.py): a grade that clips, crushes, lifts the blacks or posterises needs a human, however
well it met the notes; a gentle grade and the modern looks pass on every test frame."""

import numpy as np
import pytest

from cierra import load
from cierra.checker import load_file
from cierra.qc import check_program
from conftest import LOOKS, frame

FRAMES = ("chart", "pens", "crowd", "window_0", "window_1", "window_2")


def _flags(src, name):
    return check_program(load("cierra 0.1\n" + src), frame(name))["flags"]


@pytest.mark.parametrize("name", FRAMES)
def test_gentle_grade_and_modern_looks_pass(name):
    assert _flags("node a = exposure(stops: 0.3)\nnode b = saturation(amount: 1.1)\ngraph a -> b\n", name) == []
    for look in ("beauty_modern", "cinematic_modern"):
        assert check_program(load_file(LOOKS / f"{look}.cierra"), frame(name))["flags"] == [], look


@pytest.mark.parametrize("src,what,name", [
    ("node a = exposure(stops: 2)\ngraph a\n", "clipped highlights", "pens"),
    ("node a = exposure(stops: -2)\ngraph a\n", "crushed blacks", "window_1"),
    ("node a = wheels(master_lift: 0.08)\ngraph a\n", "milky blacks", "pens"),
    ("node a = curve(points: [[0, 0], [0.40, 0.30], [0.42, 0.55], [1, 1]])\ngraph a\n", "posterisation", "chart"),
])
def test_damage_is_flagged(src, what, name):
    flags = _flags(src, name)
    assert any(f.startswith(what) for f in flags), flags


def test_known_issue_teal_night_clips():
    """teal_night (display_encode output, no tone scale) clips on bright frames: an open question for this look
    (a tone-mapped version). The gate must keep saying so until that is decided."""
    q = check_program(load_file(LOOKS / "teal_night.cierra"), frame("pens"))
    assert q["needs_human"] and any(f.startswith("clipped highlights") for f in q["flags"])


def test_banding_added_by_a_steep_grade_of_an_8_bit_sky_not_a_16_bit_one():
    """A sky case: a smooth sky-like gradient, graded x3 around mid-grey. From 8-bit codes the
    grade spreads each step into a visible band (index +0.09, over the 0.06 threshold); from 16-bit codes it stays
    smooth."""
    from cierra.banding import banding_index
    from cierra.qc import damage
    x = np.linspace(0.15, 0.75, 480)[None, :].repeat(270, 0)
    sky = 0.45 + 0.08 * x

    def rgb(y):
        return np.stack([y, y * 0.98, y * 1.02]).clip(0, 1)

    for bits, banded in ((16, False), (8, True)):
        src = np.round(sky * (2 ** bits - 1)) / (2 ** bits - 1)
        graded = 0.5 + (src - 0.49) * 3
        flags = damage(rgb(src), rgb(graded))["flags"]
        assert any(f.startswith("banding added") for f in flags) == banded, (bits, flags)
    smooth, _ = banding_index(rgb(x))
    posterised, _ = banding_index(rgb(np.round(x * 31) / 31))
    assert smooth == 0.0 and 0.15 <= posterised <= 0.3


def test_noisy_16_bit_sky_graded_steeply_adds_no_banding_at_the_preview_size():
    """Sensor noise (1 LSB of 8 bits, Gaussian) on a 1920 px 16-bit sky before a
    x3 grade is not banding. Through the 480 px preview the gate measures on, the delta stays near 0 (measured -0.016..0
    for 0.3-2 LSB). Known limit: noise of 1 LSB or more still present AT 480 px (a tiny source, or coarse grain) reads
    as bands, +0.07-0.11, a false "banding added"."""
    from cierra.llm import preview
    from cierra.qc import THRESHOLDS, damage
    rng = np.random.default_rng(3)
    x = np.linspace(0.15, 0.75, 1920)[None, :].repeat(1080, 0)
    sky = np.round((0.40 + 0.12 * x + rng.normal(0, 1 / 255, x.shape)) * 65535) / 65535
    full = np.stack([sky, sky * 0.98, sky * 1.02]).clip(0, 1)
    rgb, graded = preview(full, 480), preview((0.5 + (full - 0.49) * 3).clip(0, 1), 480)
    r = damage(rgb, graded)
    added = r["after"]["banding"] - r["before"]["banding"]
    assert added < THRESHOLDS["banding"] and not any(f.startswith("banding") for f in r["flags"]), added
    assert added < THRESHOLDS["banding_watch"] and r["watch"] == [], added


def test_grain_at_the_preview_size_amplified_by_a_steep_grade_is_not_banding():
    """1-2 LSB of noise still present at 480 px (a small source, coarse grain), graded x3 from 16 bits,
    read as bands (+0.07-0.11) while the noise blur was capped at 4 px; at 8 px it stays under the watch band."""
    from cierra.qc import THRESHOLDS, damage
    x = np.linspace(0.15, 0.75, 480)[None, :].repeat(270, 0)
    for lsb, seed in ((1.0, 1), (2.0, 1), (1.0, 2)):
        rng = np.random.default_rng(seed)
        sky = np.round((0.40 + 0.12 * x + rng.normal(0, lsb / 255, x.shape)) * 65535) / 65535
        rgb = np.stack([sky, sky * 0.98, sky * 1.02]).clip(0, 1)
        r = damage(rgb, (0.5 + (rgb - 0.49) * 3).clip(0, 1))
        assert r["after"]["banding"] - r["before"]["banding"] < THRESHOLDS["banding_watch"], (lsb, seed, r["flags"])
