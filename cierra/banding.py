"""Banding index: how much a display image shows bands, flat regions that meet a neighbouring flat region a few code
steps away where that step is visible. Written (2026-10-29) from the published idea of CAMBI
(Tandon et al., "CAMBI: Contrast-aware Multiscale Banding Index", Netflix, 2021); not a port of libvmaf and not
bit-exact with CAMBI. Ported here from OpenCV to scipy.ndimage (Cierra has no OpenCV dependency).

    banding_index(display (3, H, W) in 0..1) -> (score, per-pixel map (H, W))

1. Luma (Rec.709 weights on the display values), blurred so that dither and sensor noise average out as they do to
   the eye (sigma from the image's own noise, 1..8 px: at a 4 px cap, grain that a steep
   grade amplified 3x still read as bands), then 10-bit code values.
2. Flat pixels: a zero-gradient neighbourhood (5x5 at full size, 3x3 at coarser scales).
3. For each flat pixel with value v and each step range in STEP_RANGES: p0 = share of flat pixels in a window with
   value v, pk = share a..b codes above or below (the larger); c = p0 * pk / (p0 + pk), weighted by how visible that
   step is at that luminance (display-linear Weber contrast / 10 %, capped at 1). The pixel keeps the largest c.
4. Four scales, halving each time, with a window of a tenth of the image width; the score is the largest, over
   scales, of the mean of the worst 20 % of flat pixels.

Higher means more banding: 0 for smooth or dithered gradients, about 0.2 for 5- to 8-bit posterisation (it detects
banding, it does not rank how bad it is). Use the difference graded - ungraded at the 480 px preview (cierra.qc);
full size is slow and less reliable on noisy footage.
"""

import numpy as np
from scipy import ndimage as ndi

BITS = 10
STEP_RANGES = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 40))  # 10-bit codes: 5- to 8-bit steps, not real edges
WINDOW_FRAC = 1 / 10
TOP_SHARE = 0.20


def _blur(y, sigma):
    return ndi.gaussian_filter(y, sigma, mode="reflect")


def _luma_codes(rgb):
    y = (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).astype(np.float32)
    m = 2 ** BITS - 1
    hp = y - _blur(y, 1.0)
    noise_codes = 1.4826 * float(np.median(np.abs(hp - np.median(hp)))) * m / 0.6  # high-pass keeps ~60 % of noise
    y = _blur(y, float(np.clip(noise_codes * 0.94, 1.0, 8.0)))  # residual noise ~0.3 codes; 20 px bands survive
    return np.clip(np.rint(np.clip(y, 0, 1) * m), 0, m).astype(np.int32)


def _visibility(v, k):
    """0..1: how visible a step of k codes is at code value v (an 8-bit step at mid-grey counts ~0.27, 6-bit 1)."""
    m = 2 ** BITS - 1
    lo = max((v / m) ** 2.2, 1e-4)
    hi = max(((v + k) / m) ** 2.2, 1e-4)
    return float(np.clip((hi - lo) / (lo + 0.003) / 0.10, 0, 1))  # 0.003: a dark floor (flare), near black


def _box(a, win):
    return ndi.uniform_filter(a, win, mode="reflect") * (win * win)


def _scale_map(q, flat_size):
    H, W = q.shape
    flat = ndi.maximum_filter(q, flat_size) == ndi.minimum_filter(q, flat_size)
    out = np.zeros((H, W), np.float32)
    if not flat.any():
        return out, flat
    win = max(5, int(round(W * WINDOW_FRAC)) | 1)
    n_flat = _box(flat.astype(np.float32), win)
    cache, reach = {}, max(b for _, b in STEP_RANGES)

    def counts(level):
        if level not in cache:
            cache[level] = _box(((q == level) & flat).astype(np.float32), win)
        return cache[level]

    for v in np.unique(q[flat]).astype(int):
        for old in [L for L in cache if L < v - reach]:
            del cache[old]
        px = flat & (q == v)
        denom = np.maximum(n_flat[px], 1.0)
        p0 = counts(v)[px] / denom
        best = np.zeros_like(p0)
        for a, b in STEP_RANGES:
            up = sum(counts(v + d)[px] for d in range(a, b + 1) if v + d < 2 ** BITS)
            dn = sum(counts(v - d)[px] for d in range(a, b + 1) if v - d >= 0)
            pk = np.maximum(up, dn) / denom
            best = np.maximum(best, p0 * pk / np.maximum(p0 + pk, 1e-6) * _visibility(float(v), float(np.sqrt(a * b))))
        out[px] = best
    return out, flat


def banding_index(display, scales=4):
    """display: (3, H, W) display code values in 0..1. Returns (score, per-pixel map at full size)."""
    q = _luma_codes(np.asarray(display, np.float32))
    H, W = q.shape
    full = np.zeros((H, W), np.float32)
    per_scale = []
    for s in range(scales):
        if min(q.shape) < 16:
            break
        c, flat = _scale_map(q, 5 if s == 0 else 3)
        vals = c[flat]
        per_scale.append(float(np.sort(vals)[-max(1, int(len(vals) * TOP_SHARE)):].mean()) if vals.size else 0.0)
        ri = np.arange(H) * c.shape[0] // H  # nearest-neighbour back to full size
        ci = np.arange(W) * c.shape[1] // W
        full = np.maximum(full, c[ri][:, ci])
        q = ndi.median_filter(q, 3, mode="nearest")[: q.shape[0] // 2 * 2:2, : q.shape[1] // 2 * 2:2]
    return (max(per_scale) if per_scale else 0.0), full  # wide bands only show at coarse scales
