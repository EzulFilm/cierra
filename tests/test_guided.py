"""Guided filter (He et al., ECCV 2010) in Cierra: guided_detail and mask feather_px (spec §5, §6)."""

import numpy as np

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
