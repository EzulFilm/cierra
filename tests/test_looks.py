"""The looks against Cierra's own golden frames (regenerate with tools/make_golden.py after an intended change).

Where tests/golden/origin/ is present (reference renders of the looks' original engine, not distributed), the looks are
also held to ΔE00 <= 1.0 (mean and p95) against it on every test frame; without it those tests skip."""

import json

import numpy as np
import pytest

from cierra import colour as C
from cierra.checker import load_file
from cierra.engine import render
from conftest import FRAMES, LOOKS, ROOT

ORIGIN = ROOT / "tests" / "golden" / "origin"
GOLDEN = ROOT / "tests" / "golden" / "cierra"
LOOK_NAMES = ("cinematic_modern", "beauty_modern", "teal_night")


def cases(folder=GOLDEN):
    out = []
    for look in LOOK_NAMES:
        for p in sorted(folder.glob(f"{look}__*.npy")):
            out.append((look, p.stem.split("__")[1]))
    return out


def cierra_render(look, case):
    """Full-resolution render of a test frame (references are stored subsampled: ORIGIN_STEP, GOLDEN_STEP)."""
    prog = load_file(LOOKS / f"{look}.cierra")
    if case.endswith("_photo"):
        img = np.load(ROOT / "tests" / "data" / f"{case}.npy").astype(np.float64)  # sRGB-encoded chart
        return render(prog, img, input="srgb_photo")
    return render(prog, np.load(FRAMES / f"{case}.npy").astype(np.float64))


ORIGIN_STEP, GOLDEN_STEP = 2, 4  # rendered at full resolution, stored at every 2nd / 4th pixel (repository size)

@pytest.mark.skipif(not ORIGIN.is_dir(), reason="reference renders of the original engine are not distributed")
@pytest.mark.parametrize("look,case", cases(ORIGIN) or [("-", "-")])
def test_matches_the_original_engine(look, case):
    ref = np.load(ORIGIN / f"{look}__{case}.npy").astype(np.float64)
    de = C.delta_e00_display(cierra_render(look, case)[:, ::ORIGIN_STEP, ::ORIGIN_STEP], ref)
    assert de.mean() <= 1.0 and np.quantile(de, 0.95) <= 1.0, (de.mean(), np.quantile(de, 0.95), de.max())


@pytest.mark.parametrize("look,case", [c for c in cases()])
def test_golden(look, case):
    path = GOLDEN / f"{look}__{case}.npy"
    if not path.exists():
        pytest.skip("no golden yet: run tools/make_golden.py")
    out = cierra_render(look, case)[:, ::GOLDEN_STEP, ::GOLDEN_STEP]
    assert np.abs(out - np.load(path)).max() <= 2e-6


@pytest.mark.skipif(not ORIGIN.is_dir(), reason="the parity table needs the original engine's renders")
def test_parity_table_is_current():
    table = json.loads((ROOT / "docs" / "parity.json").read_text())
    assert set(table) >= set(LOOK_NAMES)
