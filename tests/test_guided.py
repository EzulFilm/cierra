"""Guided filter (He et al., ECCV 2010) in Cierra: guided_detail, denoise and mask feather_px (spec §5, §6)."""

import numpy as np
import pytest

from cierra import load
from cierra.engine import box_mean, gaussian_blur, grade_working, guided_filter, mask_value


def _step(h=40, w=80, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    y = np.where(np.arange(w)[None, :] < w // 2, 0.3, 0.6) * np.ones((h, 1))
    return y + rng.normal(0, noise, (h, w))


def test_box_mean_and_constants():
    x = np.full((10, 12), 0.4)
    assert np.allclose(box_mean(x, 3), 0.4) and np.allclose(guided_filter(x, x, 3, 1e-3), 0.4)


def test_guided_filter_keeps_edges_and_smooths_noise():
    y = _step()
    q = guided_filter(y, y, 4, 0.03 ** 2)
    g = gaussian_blur(y[None], 4)[0]
    assert q[:, 5:30].std() < 0.6 * y[:, 5:30].std()  # noise smoothed in the flat part
    edge_q, edge_g = q[:, 41].mean() - q[:, 38].mean(), g[:, 41].mean() - g[:, 38].mean()
    assert edge_q > 0.27 and edge_q > edge_g + 0.05  # the step stays a step; the Gaussian smears it


def test_guided_detail_has_no_halo_at_a_hard_edge():
    """Unsharp-style local contrast overshoots next to a hard edge; guided detail barely does."""
    y = _step(noise=0.0)
    code = np.repeat(y[None], 3, axis=0)
    gd = grade_working(code, load("cierra 0.1\nnode a = guided_detail(amount: 1, radius_px: 8, edge: 0.02)\n"
                                  "graph a\n"), px_scale=1.0)
    lc = grade_working(code, load("cierra 0.1\nnode a = local_contrast(amount: 1, radius_px: 8)\ngraph a\n"),
                       px_scale=1.0)
    halo_gd, halo_lc = np.abs(gd - code).max(), np.abs(lc - code).max()
    assert halo_gd < 0.01 and halo_lc > 0.05, (halo_gd, halo_lc)


def test_feathered_matte_follows_the_picture_edge():
    """A window whose soft edge lies near a picture edge: feathering pulls the matte's transition onto that edge."""
    y = _step(noise=0.0)
    code = np.repeat(y[None], 3, axis=0)
    ctx = {"frame": 0, "px_scale": 1.0, "aux": {}}
    plain = load("cierra 0.1\nnode a = exposure()\n  mask window(centre: [0.2, 0.5], radius: [0.35, 2], soft: 0.2)\n"
                 "graph a\n").nodes["a"].mask
    feath = load("cierra 0.1\nnode a = exposure()\n  mask window(centre: [0.2, 0.5], radius: [0.35, 2], soft: 0.2, "
                 "feather_px: 8)\ngraph a\n").nodes["a"].mask
    m0, m1 = mask_value(code, plain, ctx)[20], mask_value(code, feath, ctx)[20]
    jump = lambda m: m[39] - m[40]  # noqa: E731  the picture's edge is between columns 39 and 40
    assert jump(m1) > 2 * jump(m0) and m1.min() >= 0 and m1.max() <= 1


def test_feather_counts_as_a_spatial_pass_and_is_refused_by_pointwise_targets():
    from cierra.targets import dctl_reasons, lut_reasons
    prog = load("cierra 0.1\nnode a = exposure(stops: 1)\n  mask luma(range: 0.2..0.5, feather_px: 4)\ngraph a\n")
    assert any("feather" in r for r in dctl_reasons(prog)) and lut_reasons(prog)


# ---------------------------------------------------------------- denoise (spatial NR; temporal NR is not in v0)

H = "cierra 0.1\noutput acescct()\n"


def _grade(code, node):
    return grade_working(code, load(H + f"node a = {node}\ngraph a\n"), px_scale=1.0)


def _width_10_90(p):
    """Step response width in pixels: distance between the 10 % and 90 % crossings of a rising profile."""
    p = (p - p[:10].mean()) / (p[-10:].mean() - p[:10].mean())
    x = np.arange(len(p), dtype=float)
    return np.interp(0.9, p, x) - np.interp(0.1, p, x)


def test_denoise_is_the_identity_at_strength_0():
    code = np.random.default_rng(3).uniform(0.05, 0.9, (3, 24, 32))
    for node in ("denoise(strength: 0)", "denoise(strength: 0, radius_px: 20, edge: 0.2, chroma: 1)"):
        assert np.array_equal(_grade(code, node), code)


def test_denoise_lowers_noise_on_a_flat_patch():
    """Grey patch + gaussian noise sigma 0.01 cv per channel: at strength 1 luma noise drops >= 4x (5.9x measured),
    chroma noise >= 8x (11.4x; filtered over twice the radius); chroma: 0 leaves chroma noise as it was."""
    from cierra.engine import opponent
    code = 0.4 + np.random.default_rng(1).normal(0, 0.01, (3, 64, 64))
    inner = (slice(None), slice(8, -8), slice(8, -8))
    y0, _, _, cb0, _ = opponent(code[inner])
    y1, _, _, cb1, _ = opponent(_grade(code, "denoise(strength: 1)")[inner])
    assert y0.std() / y1.std() >= 4 and cb0.std() / cb1.std() >= 8, (y0.std() / y1.std(), cb0.std() / cb1.std())
    y2, _, _, cb2, _ = opponent(_grade(code, "denoise(strength: 1, chroma: 0)")[inner])
    assert np.isclose(y2.std(), y1.std()) and np.allclose(cb2, cb0)
    assert abs(y1.mean() - y0.mean()) < 1e-3  # no level shift


def test_denoise_keeps_an_edge_sharp():
    """0.3 -> 0.6 cv step + the same noise: the 10-90 % width of the row-averaged luma grows by <= 0.25 px (0.79 ->
    0.81 measured), while a Gaussian blur that removes about as much noise (sigma 1.5, 5.1x) smears it to > 3 px."""
    from cierra.engine import luma
    h, w = 64, 80
    step = np.where(np.arange(w)[None, :] < w // 2, 0.3, 0.6) * np.ones((h, 1))
    code = np.repeat(step[None], 3, axis=0) + np.random.default_rng(1).normal(0, 0.01, (3, h, w))
    before = _width_10_90(luma(code).mean(0))
    after = _width_10_90(luma(_grade(code, "denoise(strength: 1)")).mean(0))
    blurred = _width_10_90(luma(_grade(code, "blur(radius_px: 1.5)")).mean(0))
    assert after - before <= 0.25 and blurred > 3, (before, after, blurred)


def test_denoise_is_refused_by_pointwise_and_pixel_shader_targets(tmp_path):
    from cierra.compile.cuda import CUDAError, compile_cuda
    from cierra.compile.dctl import DCTLError, compile_dctl
    from cierra.compile.lut import LUTError, export_luts
    from cierra.compile.wgsl import WGSLError, compile_wgsl
    prog = load("cierra 0.1\nnode a = denoise(strength: 0.6)\ngraph a\n")
    with pytest.raises(LUTError, match="spatial"):
        export_luts(prog, tmp_path, "nr")
    for compile_, err in ((compile_dctl, DCTLError), (compile_wgsl, WGSLError), (compile_cuda, CUDAError)):
        with pytest.raises(err, match="spatial"):
            compile_(prog)
    parts = compile_dctl(load("cierra 0.1\nnode a = exposure()\nnode b = denoise()\nnode c = contrast()\n"
                              "graph a -> b -> c\n"), split=True)
    assert [p["dctl"] is None for p in parts] == [False, True, False]


def test_denoise_round_trips_through_the_formatter():
    from cierra import format_program
    prog = load("cierra 0.1\nnode nr = denoise(strength: keys(linear, 0: 0, 24: 0.8), radius_px: 4, edge: 0.03, "
                "chroma: 0.5)\n  mix 0.9\ngraph nr\n")
    text = format_program(prog)
    assert "denoise(" in text and format_program(load(text)) == text
