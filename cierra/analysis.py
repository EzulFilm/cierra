"""Image statistics for the LLM: numbers, never pixels. All functions take display code values (3, H, W) in 0..1
(sRGB) unless they say otherwise, and return plain dicts of rounded floats.

Scope conventions: luma Y' = 0.2126 R' + 0.7152 G' + 0.0722 B' on display code values; Cb = (B' - Y') / 1.8556,
Cr = (R' - Y') / 1.5748; vectorscope angle = atan2(Cr, Cb) in degrees (red ~103, skin line 122, yellow ~175,
green ~240, cyan ~283, blue ~0/360, magenta ~60).
"""

import math

import numpy as np

from .colour import delta_e00_display

SKIN_LINE_DEG = 122.0
HUE_BANDS = (("red", 103), ("orange_skin", 125), ("yellow", 168), ("green", 232), ("cyan", 285), ("blue", 345),
             ("magenta", 60))


def _r(v, n=4):
    return None if v is None or not math.isfinite(v) else round(float(v), n)


def _sub(a, n=250_000):
    flat = a.reshape(a.shape[0], -1) if a.ndim == 3 else a.reshape(1, -1)
    step = max(1, flat.shape[1] // n)
    return flat[:, ::step]


def _opp(d):
    y = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
    cb, cr = (d[2] - y) / 1.8556, (d[0] - y) / 1.5748
    return y, cb, cr, np.hypot(cb, cr), np.degrees(np.arctan2(cr, cb)) % 360


def histogram(d, bins=10):
    y = _opp(_sub(d))[0]
    h, edges = np.histogram(np.clip(y, 0, 1), bins=bins, range=(0, 1))
    return {"luma_bins_pct": [_r(100 * v / max(1, h.sum()), 2) for v in h], "bin_width": 1 / bins}


def waveform(d, columns=6):
    """Luma percentiles (p2, p50, p98) per vertical band, left to right."""
    y = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
    columns = max(1, min(columns, y.shape[1]))  # an image narrower than the bands gets one band per column
    out = []
    for band in np.array_split(y, columns, axis=1):
        f = band.flatten()[:: max(1, band.size // 50_000)]
        out.append([_r(np.quantile(f, q), 3) for q in (0.02, 0.5, 0.98)])
    return {"columns": columns, "p2_p50_p98": out}


def vectorscope(d):
    y, cb, cr, c, hue = _opp(_sub(d))
    mask = c > 0.01
    bands = {}
    for name, centre in HUE_BANDS:
        dist = np.abs(((hue - centre + 180) % 360) - 180)
        sel = mask & (dist < 30)
        bands[name] = {"share_pct": _r(100 * sel.mean(), 2),
                       "chroma_p50": _r(np.median(c[sel]), 4) if sel.any() else None}
    return {"chroma_p50": _r(np.median(c), 4), "chroma_p90": _r(np.quantile(c, 0.9), 4),
            "chroma_p99": _r(np.quantile(c, 0.99), 4), "saturation_by_hue_band": bands}


def neutral_cast(d):
    y, cb, cr, c, _ = _opp(_sub(d))
    sel = (c < 0.03) & (y > 0.2) & (y < 0.85)
    if not sel.any():
        return {"share_pct": 0.0, "reads_as": "no near-neutral pixels"}
    ncb, ncr = float(cb[sel].mean()), float(cr[sel].mean())
    mag = math.hypot(ncb, ncr)
    ang = math.degrees(math.atan2(ncr, ncb)) % 360
    names = [(0, "blue"), (60, "magenta"), (105, "red"), (150, "warm/orange"), (180, "yellow"), (240, "green"),
             (285, "cyan/teal"), (360, "blue")]
    name = min(names, key=lambda t: min(abs(ang - t[0]), 360 - abs(ang - t[0])))[1]
    level = "neutral" if mag < 0.004 else ("slight " if mag < 0.012 else "clear " if mag < 0.03 else "strong ") + name
    return {"share_pct": _r(100 * sel.mean(), 1), "angle_deg": _r(ang, 1), "magnitude": _r(mag, 4), "reads_as": level}


def skin(d):
    y, cb, cr, c, hue = _opp(_sub(d))
    dev = ((hue - SKIN_LINE_DEG + 180) % 360) - 180
    sel = (np.abs(dev) < 20) & (c > 0.02) & (c < 0.25) & (y > 0.15) & (y < 0.9)
    if sel.sum() < 50:
        return {"share_pct": _r(100 * sel.mean(), 2), "note": "too few skin-like pixels to measure"}
    return {"share_pct": _r(100 * sel.mean(), 2), "hue_offset_deg_p50": _r(np.median(dev[sel]), 2),
            "hue_offset_deg_spread": _r(np.quantile(dev[sel], 0.75) - np.quantile(dev[sel], 0.25), 2),
            "luma_p50": _r(np.median(y[sel]), 3), "chroma_p50": _r(np.median(c[sel]), 4),
            "note": "offset > 0 = toward yellow, < 0 = toward red/magenta"}


def clipping(d):
    mx = d.max(0)
    return {"clipped_white_pct": _r(100 * (mx >= 0.995).mean(), 3), "crushed_black_pct": _r(100 * (mx <= 0.005).mean(), 3),
            "any_channel_clipped_pct": _r(100 * (d >= 0.995).any(0).mean(), 3),
            "any_channel_crushed_pct": _r(100 * (d <= 0.002).any(0).mean(), 3)}


def tone(d):
    y = _opp(_sub(d))[0]
    q = np.quantile(y, [0.005, 0.05, 0.5, 0.95, 0.995])
    return {"black_p0.5": _r(q[0], 3), "p5": _r(q[1], 3), "p50": _r(q[2], 3), "p95": _r(q[3], 3),
            "white_p99.5": _r(q[4], 3), "mean": _r(y.mean(), 3), "mid_grey_reference": 0.461}


def stats(d):
    """Everything the LLM gets about one frame."""
    return {"units": "display sRGB code values 0..1; vectorscope degrees (skin line 122)", "tone": tone(d),
            "histogram": histogram(d), "waveform": waveform(d), "vectorscope": vectorscope(d),
            "neutral_cast": neutral_cast(d), "skin": skin(d), "clipping": clipping(d)}


def flicker(frames):
    """Frames: list of display images. Mean-luma change frame to frame (display code values)."""
    means = [float((0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2]).mean()) for f in frames]
    steps = np.abs(np.diff(means)) if len(means) > 1 else np.zeros(1)
    return {"frames": len(means), "mean_luma": [_r(m, 4) for m in means], "max_step": _r(steps.max(), 4),
            "mean_step": _r(steps.mean(), 4)}


def delta_e(d, ref):
    """CIEDE2000 between two display images of the same size."""
    de = delta_e00_display(d, ref).flatten()
    return {"mean": _r(de.mean(), 3), "p95": _r(np.quantile(de, 0.95), 3), "max": _r(de.max(), 3)}


def measure(d, ref=None, skin_target_deg=0.0):
    """The loop's scorecard: ΔE00 to the reference look (if any), skin hue error, clipping."""
    out = {"clipping": clipping(d), "skin": skin(d)}
    off = out["skin"].get("hue_offset_deg_p50")
    out["skin_hue_error_deg"] = None if off is None else _r(abs(off - skin_target_deg), 2)
    if ref is not None:
        out["delta_e00_to_reference"] = delta_e(d, ref)
    return out
