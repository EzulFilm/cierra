"""Damage checks: did the grade break the picture? Compared on the display output, graded against ungraded, so a
frame that was already clipped or crushed is not blamed on the grade. Any flag means "needs a human": a loop or tuner
must not report the grade as done, however well it met the notes. Simple measures with no perceptual model; none is
validated against viewers (plain pixel percentages, and the posterisation test
is a heuristic, not a published method; CAMBI (Netflix, BSD-3) is the published candidate). They veto damage only;
they never rank looks:

- clipped highlights: % of pixels with a channel at display white (>= 0.998);
- crushed blacks: % of pixels with every channel at display black (<= 0.005);
- milky blacks: the darkest 1 % of luma lifted (washed out);
- posterisation: empty 8-bit luma histogram bins inside the image's own p1..p99 luma range that were filled before
  (a curve or contrast step that tears smooth gradients apart).

Thresholds are judgement, set on the test frames (a +0.3 stop, saturation 1.1 grade passes on every frame; +-2
stops, a hard curve step or lifted blacks do not); clipping and crushing must also rise by more than 15 % of what
the ungraded frame already had (the spark frames start 14-19 % white). They are in THRESHOLDS and in each flag.
"""

import numpy as np

THRESHOLDS = {"clipped": 1.0, "crushed": 1.0, "relative": 0.15, "milky": 0.04, "posterised": 6}


def _luma(d):
    return 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]


def measure(d):
    """The damage measures of one display image (3, H, W) in 0..1."""
    d = np.clip(d, 0, 1)
    y = _luma(d)
    lo, hi = np.quantile(y, [0.01, 0.99])
    hist = np.bincount(np.clip((y * 255 + 0.5).astype(int), 0, 255).ravel(), minlength=256)
    a, b = int(lo * 255 + 0.5), int(hi * 255 + 0.5)
    return {"clipped": float(100 * (d.max(0) >= 0.998).mean()), "crushed": float(100 * (d.max(0) <= 0.005).mean()),
            "milky": float(lo), "empty_bins": int((hist[a:b + 1] == 0).sum()), "range_bins": b - a + 1}


def damage(ungraded, graded):
    """{"flags": [...], "needs_human": bool, "before": measures, "after": measures} for two display images of the
    same frame."""
    m0, m1 = measure(ungraded), measure(graded)
    t, flags = THRESHOLDS, []
    if m1["clipped"] - m0["clipped"] > max(t["clipped"], t["relative"] * m0["clipped"]):
        flags.append(f"clipped highlights: {m1['clipped']:.1f} % of pixels at white (ungraded {m0['clipped']:.1f} %, "
                     f"allowed +{max(t['clipped'], t['relative'] * m0['clipped']):.1f})")
    if m1["crushed"] - m0["crushed"] > max(t["crushed"], t["relative"] * m0["crushed"]):
        flags.append(f"crushed blacks: {m1['crushed']:.1f} % of pixels at black (ungraded {m0['crushed']:.1f} %, "
                     f"allowed +{max(t['crushed'], t['relative'] * m0['crushed']):.1f})")
    if m1["milky"] - m0["milky"] > t["milky"]:
        flags.append(f"milky blacks: darkest 1 % of luma at {m1['milky']:.3f} (ungraded {m0['milky']:.3f}, "
                     f"allowed +{t['milky']})")
    gaps = m1["empty_bins"] - min(m0["empty_bins"], m1["range_bins"])
    if gaps > t["posterised"]:
        flags.append(f"posterisation: {m1['empty_bins']} empty 8-bit luma levels inside the image's range "
                     f"(ungraded {m0['empty_bins']}, allowed +{t['posterised']})")
    return {"flags": flags, "needs_human": bool(flags), "before": m0, "after": m1}


def check_program(prog, image, input=None, width=480):
    """damage() of a program on a frame (rendered at a preview width; input defaults to the program's own)."""
    import copy

    from .engine import render
    from .llm import preview
    input = input or prog.input
    base = copy.copy(prog)  # ungraded = the same program without its graph: same input, photo decode and output
    base.graph = None
    small = preview(np.asarray(image, dtype=np.float64), width)
    return damage(render(base, small, input=input), render(prog, small, input=input))
