"""Damage checks (cierra/qc.py): a grade that clips, crushes, lifts the blacks or posterises needs a human, however
well it met the notes; a gentle grade and the modern looks pass on every test frame."""

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
