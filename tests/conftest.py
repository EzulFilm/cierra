import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "cache").mkdir(exist_ok=True)  # pytest's --basetemp=cache/pytest (pyproject.toml) needs it in a fresh clone

FRAMES = ROOT / "tests" / "data" / "frames"
LOOKS = ROOT / "looks"


def frame(name):
    return np.load(FRAMES / f"{name}.npy").astype(np.float64)


@pytest.fixture(scope="session")
def frames():
    return {p.stem: frame(p.stem) for p in sorted(FRAMES.glob("*.npy"))}


@pytest.fixture(scope="session")
def chart():
    return frame("chart")


def small(img, step=4):
    return np.ascontiguousarray(img[:, ::step, ::step])
