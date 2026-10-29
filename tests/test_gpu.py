"""The reference engine on the GPU (CuPy) against the same engine on the CPU (NumPy): the same float64 maths, so the
bound is float64 rounding (summation order in filters and matrix products), not a target's float32. Skipped unless
CuPy finds a GPU and CIERRA_GPU_TESTS=1 is set (the GPU may be shared; running on it is an explicit choice)."""

import os

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.backend import gpu_unavailable
from cierra.checker import load_file
from cierra.engine import grade_working, render
from conftest import LOOKS, small
from regression_cases import OP_CASES, op_input, op_source
from test_targets import KEYS_PROGRAM, KITCHEN_SINK, PASS_PROGRAM

why = gpu_unavailable() or (None if os.environ.get("CIERRA_GPU_TESTS") == "1" else "set CIERRA_GPU_TESTS=1")
pytestmark = pytest.mark.skipif(why is not None, reason=f"GPU engine: {why}")
TOL = 1e-9


@pytest.mark.parametrize("name", list(OP_CASES))
def test_gpu_every_op_and_mask(name):
    prog = load(op_source(name))
    code = op_input()
    cpu = grade_working(code, prog, frame=7, px_scale=0.5)
    gpu = grade_working(code, prog, frame=7, px_scale=0.5, device="gpu")
    assert isinstance(gpu, np.ndarray) and np.abs(gpu - cpu).max() < TOL, (name, np.abs(gpu - cpu).max())


@pytest.mark.parametrize("src", [KITCHEN_SINK, PASS_PROGRAM, KEYS_PROGRAM], ids=["kitchen_sink", "pass", "keys"])
def test_gpu_programs(frames, src):
    prog = load(src)
    for name, img in frames.items():
        code = C.to_working(small(img, 2))
        cpu = grade_working(code, prog, frame=5)
        gpu = grade_working(code, prog, frame=5, device="gpu")
        assert np.abs(gpu - cpu).max() < TOL, (name, np.abs(gpu - cpu).max())


@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern", "teal_night"])
def test_gpu_render_looks(frames, look):
    """Whole renders, input and ACES 2.0 output included (OpenColorIO runs on the CPU between them)."""
    prog = load_file(LOOKS / f"{look}.cierra")
    for name, img in frames.items():
        cpu = render(prog, small(img, 2))
        gpu = render(prog, small(img, 2), device="gpu")
        assert isinstance(gpu, np.ndarray) and np.abs(gpu - cpu).max() < 1e-6, (name, np.abs(gpu - cpu).max())


def test_gpu_camera_log_input_and_spatial_ops(chart):
    """A camera log input (OpenColorIO on the way in) with glow, guided detail and a feathered mask."""
    prog = load("cierra 0.1\ninput sony_slog3_sgamut3cine\noutput aces2(display: srgb)\n"
                "node d = guided_detail(amount: -0.7, radius_px: 4, edge: 0.03)\n"
                "node g = glow(threshold: 0.6, radius_px: 12, strength: 0.2, warm: 0.1)\n"
                "node w = exposure(stops: 0.5)\n  mask window(radius: [0.3, 0.3], soft: 0.1, feather_px: 6)\n"
                "graph d -> g -> w\n")
    img = np.clip(small(chart, 2) * 0.5 + 0.2, 0, 1)
    cpu, gpu = render(prog, img), render(prog, img, device="gpu")
    assert np.abs(gpu - cpu).max() < 1e-6, np.abs(gpu - cpu).max()


def test_device_choice(monkeypatch):
    prog = load("cierra 0.1\noutput acescct()\nnode a = exposure(stops: 1)\ngraph a\n")
    code = op_input()
    monkeypatch.setenv("CIERRA_DEVICE", "gpu")
    assert np.abs(grade_working(code, prog) - grade_working(code, prog, device="cpu")).max() < TOL
    with pytest.raises(ValueError):
        grade_working(code, prog, device="tpu")
