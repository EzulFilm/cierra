"""LUT fitting: turn a 3D LUT (for example a grading application's whole clip grade, exported as a .cube) back into a
short, readable Cierra program, and say how far the program is from the LUT.

A LUT is the only complete record of a clip's grade that the application's scripting API gives out: wheel, curve,
window and qualifier values cannot be read. fit_lut(lut, input="acescct") reads the grade back as ops a colourist
recognises, fitted in this order (each stage on what the stages before it leave):

  primary   the neutral axis (r = g = b) of the LUT, three ways, and the one that fits the whole lattice best is kept:
            (a) one parametric primary: printer_lights, contrast or cdl (slope / offset / power per channel,
                least squares; only when it fits the neutral axis within 0.0015 cv);
            (b) curve per channel (red, green, blue, or one rgb curve when the three agree);
            (c) curve(channel: luma) plus wheels (lift / gamma / gain / offset, a linear least-squares fit of the
                colour of the neutral axis on the wheels' own tonal weights).
            (a) and (b) act on each channel alone, (c) keeps colour: which one the LUT is like shows off the axis.
  colour    saturation, hue_vs_hue, hue_vs_sat (12 hue points 30 degrees apart: their raised-cosine weights sum to
            one, so the fit is linear least squares), lum_vs_sat (a curve over luma), hue_vs_lum, from the chroma
            ratio, hue angle and luma of the LUT against the primary's output (weighted by chroma). Points that change
            nothing are left out, and an op that changes nothing is not written.
  refine    every value at once, by robust least squares on CIE Lab differences after the ACES 2.0 output (the
            stages above fit one after another, and a later one changes what an earlier one should have been);
            then nodes the fit does not need are left out.
  masked    only with hints (the application's node list): for a node that uses a qualifier, a qualifier-masked
            saturation, white balance or exposure where the fit misses most, kept only when it lowers the measured
            error by 5 % or more.

The fit sample is a sub-lattice of the LUT's own nodes plus 3000 fixed colours like those in pictures (mostly low
chroma), so the many saturated lattice corners do not outweigh skin, greys and foliage.

The residual is measured, not assumed: CIEDE2000 between the LUT and the fitted program after the program's ACES 2.0
output, over every lattice node and over test frames when given (report["lattice"], report["frames"]). A LUT holds
one pointwise function of colour, so spatial effects (blur, glow, grain), windows and keyframes are not in it and
cannot come back out of it.

The LUT maps the domain `input` to the same domain: ACEScct (the default), ACEScg or linear Rec.709; the program
works in ACEScct like every Cierra program.
"""

import functools
import math
import re
from pathlib import Path

import numpy as np

from . import colour as C
from .checker import load
from .engine import GREY, grade_working, luma, monotone_curve, opponent, smoothstep

PRINTER_POINT = 0.025 / (17.52 * math.log10(2))
DOMAINS = {
    "acescct": (lambda v: np.asarray(v, dtype=np.float64), lambda v: np.asarray(v, dtype=np.float64)),
    # negative values keep the ACEScct toe's straight line (a LUT may map a colour below black and back)
    "acescg": (C.acescct_encode, C.acescct_decode),
    "linear_rec709": (lambda v: C.acescct_encode(C.apply_matrix(C.M709_AP1, v)), C.from_working),
}
HUES = np.arange(0.0, 360.0, 30.0)  # hue points of the fitted hue curves (width 30: a partition of unity)
LUM_KNOTS = np.array([0.0, 0.1, 0.2, 0.3, GREY, 0.5, 0.6, 0.7, 0.85, 1.0])
TOL_PARAM = 0.0015   # cv: a parametric primary must fit the neutral axis this well
TOL_GAIN = 0.004     # cv: the same for linear gains, whose kink at the ACEScct toe a fitted lattice rounds off
TOL_CURVE = 0.0008   # cv: greedy curve knots are added until the neutral axis fits this well (or 12 points)
QUALIFIER_TOOLS = ("qualifier", "hsl", "key", "3d")
MAX_CUBE = 129


class LUTFitError(ValueError):
    pass


# ---------------------------------------------------------------- reading and applying a LUT

def read_cube(path):
    """A .cube 3D LUT -> (lut (3, N, N, N) indexed [channel, b, g, r], domain_min (3,), domain_max (3,))."""
    try:
        return _parse_cube(Path(path).read_text(encoding="ascii", errors="replace"))
    except LUTFitError:
        raise
    except (ValueError, IndexError) as e:
        raise LUTFitError(f"not a readable .cube: {e}") from None


def _parse_cube(text):
    size, dmin, dmax, rows = None, np.zeros(3), np.ones(3), []
    for line in text.splitlines():
        s = line.split("#", 1)[0].strip()
        if not s:
            continue
        key = s.split()[0].upper()
        if key == "TITLE":
            continue
        if key == "LUT_1D_SIZE":
            raise LUTFitError("a 1D LUT: fit_lut needs a 3D LUT (LUT_3D_SIZE)")
        if key == "LUT_3D_SIZE":
            size = int(s.split()[1])
            if not 2 <= size <= MAX_CUBE:
                raise LUTFitError(f"LUT_3D_SIZE {size} outside 2..{MAX_CUBE}")
        elif key == "DOMAIN_MIN":
            dmin = np.array([float(x) for x in s.split()[1:4]])
        elif key == "DOMAIN_MAX":
            dmax = np.array([float(x) for x in s.split()[1:4]])
        elif key == "LUT_3D_INPUT_RANGE":
            lo, hi = (float(x) for x in s.split()[1:3])
            dmin, dmax = np.full(3, lo), np.full(3, hi)
        else:
            try:
                vals = [float(v) for v in s.split()]
            except ValueError:
                continue  # other keywords (LUT_IN_VIDEO_RANGE, ...) do not change a full-range float LUT
            if len(vals) != 3:
                raise LUTFitError(f"bad LUT row {line.strip()!r}")
            rows.append(vals)
            if size and len(rows) > size ** 3:
                raise LUTFitError("more rows than LUT_3D_SIZE ** 3")
    if size is None:
        raise LUTFitError("no LUT_3D_SIZE: not a 3D .cube")
    if len(rows) != size ** 3:
        raise LUTFitError(f"{len(rows)} rows, LUT_3D_SIZE {size} needs {size ** 3}")
    a = np.array(rows, dtype=np.float64)
    if not np.all(np.isfinite(a)) or len(dmin) != 3 or len(dmax) != 3 or np.any(dmax <= dmin):
        raise LUTFitError("non-finite LUT values or a bad domain")
    return a.reshape(size, size, size, 3).transpose(3, 0, 1, 2), dmin, dmax  # rows: red fastest -> [c, b, g, r]


def apply_lut(lut, pts, dmin=(0, 0, 0), dmax=(1, 1, 1)):
    """Tetrahedral interpolation of lut (3, N, N, N) at domain values pts (3, ...) (clipped to the domain)."""
    from .compile.lut import _tetra
    size = lut.shape[1]
    dmin, dmax = np.asarray(dmin, float).reshape(3, 1), np.asarray(dmax, float).reshape(3, 1)
    shape = pts.shape
    unit = (np.asarray(pts, dtype=np.float64).reshape(3, -1) - dmin) / (dmax - dmin)
    idx, w = _tetra(size, unit)
    flat = lut.reshape(3, -1)
    return sum(w[k] * flat[:, idx[k]] for k in range(4)).reshape(shape)


def _nodes(size, dmin, dmax):
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    unit = np.stack([r, g, b])
    return dmin.reshape(3, 1, 1, 1) + unit * (dmax - dmin).reshape(3, 1, 1, 1)


# ---------------------------------------------------------------- program text

def _num(v, nd=4):
    s = f"{float(v):.{nd}f}".rstrip("0").rstrip(".") if nd else f"{float(v):.0f}"
    return "0" if s in ("-0", "") else s


def _pts(points, nds=(4, 4, 0)):
    return "[" + ", ".join("[" + ", ".join(_num(x, nds[i]) for i, x in enumerate(p)) + "]" for p in points) + "]"


def _rgb(v, nd=4):
    return "[" + ", ".join(_num(x, nd) for x in v) + "]"


def _text(nodes, input="acescct", name="", about=""):
    def q(s):
        return '"' + " ".join(re.sub(r"[^ !#-\[\]-~]", " ", str(s)).split())[:380] + '"'
    lines = ["cierra 0.1"]
    if name:
        lines.append(f"name {q(name)}")
    if about:
        lines.append(f"about {q(about)}")
    lines += [f"input {input}", "output aces2()", ""]
    for n in nodes:
        lines.append(f"node {n['id']} = {n['call']}")
        if n.get("label"):
            lines.append(f"  label {q(n['label'])}")
        if n.get("mask"):
            lines.append(f"  mask {n['mask']}")
    lines.append("")
    if nodes:
        lines.append("graph " + " -> ".join(n["id"] for n in nodes))
    return "\n".join(lines) + "\n"


@functools.lru_cache(maxsize=512)
def _load(text):
    return load(text)  # read-only use: the refinement loads its own copy to change


def _run(nodes, x):
    """The chain of node dicts on ACEScct points x (3, M)."""
    if not nodes:
        return x
    prog = _load(_text(nodes))
    return grade_working(x.reshape(3, -1, 1), prog, px_scale=1.0).reshape(x.shape)


# ---------------------------------------------------------------- the primary (neutral axis)

def _greedy_curve(x, y, tol=TOL_CURVE, max_pts=12):
    """Fewest monotone-curve points (from x) through which the curve follows y within tol (or max_pts)."""
    x, y = np.asarray(x, float), np.clip(np.asarray(y, float), -0.2, 1.5)
    pick = sorted({0, len(x) // 2, len(x) - 1})
    while True:
        pts = [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in pick]
        err = np.abs(monotone_curve(x, pts) - y)
        i = int(np.argmax(err))
        if err[i] <= tol or len(pick) >= max_pts or i in pick:
            return pts
        pick = sorted(set(pick) | {i})


def _cdl_channel(x, y):
    from scipy.optimize import least_squares

    def f(p):
        v = x * p[0] + p[1]
        return np.where(v >= 0, np.power(np.maximum(v, 0), p[2]), v)
    r = least_squares(lambda p: f(p) - y, [1.0, 0.0, 1.0], bounds=([0.0, -0.5, 0.2], [4.0, 0.5, 4.0]))
    return r.x, float(np.abs(f(r.x) - y).max())


def _gains(x, n):
    """Exposure / white balance (linear-light gains, engine._linear_gain) for the neutral axis, or None."""
    lin_x, lin_n = C.acescct_decode(x), C.acescct_decode(n)
    use = lin_x > 0.01  # gain = ratio of linear values, read where they are well above zero
    if use.sum() < 4:
        return None
    g = np.log2(np.median(lin_n[:, use] / lin_x[use], axis=1))
    e, t = (g[0] + g[2]) / 2, (g[0] - g[2]) / 2
    n_ = e - g[1]
    model = C.acescct_encode(lin_x[None] * (2.0 ** np.array([e + t, e - n_, e - t]))[:, None])
    if np.abs(model - n).max() > TOL_GAIN:
        return None
    label = "primary from the LUT's neutral axis (linear-light gains)"
    out = []
    if abs(e) > 1e-3:
        out.append({"id": "exposure", "call": f"exposure(stops: {_num(e)})", "label": label})
    if abs(t) > 1e-3 or abs(n_) > 1e-3:
        out.append({"id": "balance", "call": f"white_balance(temperature: {_num(t)}, tint: {_num(n_)})",
                    "label": "" if out else label})
    return out or None


def _parametric(x, n):
    """One parametric primary for the neutral axis n (3, K) at x (K,): a node dict, or None."""
    gains = _gains(x, n)
    if gains:
        return gains
    fits = [_cdl_channel(x, n[c]) for c in range(3)]
    if max(e for _, e in fits) > TOL_PARAM:
        return None
    s, o, p = (np.array([f[0][k] for f in fits]) for k in range(3))
    label = "primary from the LUT's neutral axis"
    if np.allclose(p, 1, atol=2e-3):
        A = np.stack([x, np.ones_like(x)], 1)
        so = np.array([np.linalg.lstsq(A, n[c], rcond=None)[0] for c in range(3)])
        s, o = so[:, 0], so[:, 1]
        if np.ptp(s) < 2e-3:
            a = float(s.mean())
            if abs(a - 1) < 2e-3:
                pts = o / PRINTER_POINT  # an offset in ACEScct (exposure is a linear gain, not an offset)
                if np.abs(pts).max() <= 24:
                    return [{"id": "balance", "call": f"printer_lights(red: {_num(pts[0], 3)}, green: "
                             f"{_num(pts[1], 3)}, blue: {_num(pts[2], 3)})", "label": label}]
            elif 0.3 <= a <= 3:
                piv = float(o.mean() / (1 - a))
                rest = (o - piv * (1 - a)) / PRINTER_POINT
                if 0 <= piv <= 1 and np.abs(rest).max() <= 24:
                    out = [{"id": "contrast", "call": f"contrast(amount: {_num(a)}, pivot: {_num(piv)})",
                            "label": label}]
                    if np.abs(rest).max() * PRINTER_POINT > 2e-4:  # after the contrast: a x + piv (1 - a) + rest
                        out.append({"id": "balance", "call": f"printer_lights(red: {_num(rest[0], 3)}, "
                                    f"green: {_num(rest[1], 3)}, blue: {_num(rest[2], 3)})"})
                    return out
        p = np.ones(3)
    call = f"cdl(slope: {_rgb(s, 5)}, offset: {_rgb(o, 5)}"
    call += ")" if np.allclose(p, 1) else f", power: {_rgb(p, 5)})"
    return [{"id": "cdl", "call": call, "label": label}]


def _curves(x, n):
    pts = [_greedy_curve(x, n[c]) for c in range(3)]
    if all(np.abs(monotone_curve(x, pts[0]) - n[c]).max() <= TOL_CURVE * 1.5 for c in range(3)):
        return [{"id": "tone", "call": f"curve(points: {_pts(pts[0])}, channel: rgb)",
                 "label": "tone, each channel alike (from the LUT's neutral axis)"}]
    return [{"id": f"tone_{ch}", "call": f"curve(points: {_pts(p)}, channel: {ch})",
             "label": f"{ch} channel (from the LUT's neutral axis)" if ch == "red" else ""}
            for ch, p in zip(("red", "green", "blue"), pts)]


def _luma_wheels(x, n):
    yl = luma(n)
    pts = _greedy_curve(x, yl)
    out = [{"id": "tone", "call": f"curve(points: {_pts(pts)})", "label": "tone, colour kept (luma curve)"}]
    y = np.clip(monotone_curve(x, pts), 0, 1)
    basis = np.stack([1 - smoothstep(0.1, GREY + 0.05, y), np.exp(-((y - GREY) / 0.18) ** 2),
                      smoothstep(GREY - 0.05, 0.75, y), np.ones_like(y)], 1)
    d = n - yl[None]
    coef = np.clip(np.array([np.linalg.lstsq(basis, d[c], rcond=None)[0] for c in range(3)]), -0.3, 0.3)  # (3, 4)
    if np.abs(coef).max() >= 5e-4:
        args = [f"{k}: {_rgb(coef[:, j])}" for j, k in enumerate(("lift", "gamma", "gain", "offset"))
                if np.abs(coef[:, j]).max() >= 2e-4]
        out.append({"id": "balance", "call": "wheels(" + ", ".join(args) + ")",
                    "label": "colour of the neutral axis (wheels)"})
    return out


# ---------------------------------------------------------------- colour

def _hue_basis(hue):
    d = (hue[:, None] - HUES[None, :] + 180) % 360 - 180
    return 0.5 + 0.5 * np.cos(np.pi * np.clip(d / 30.0, -1, 1))


def _lum_basis(y):
    k = LUM_KNOTS
    yc = np.clip(y, k[0], k[-1])
    out = np.zeros((len(y), len(k)))
    i = np.clip(np.searchsorted(k, yc) - 1, 0, len(k) - 2)
    t = (yc - k[i]) / (k[i + 1] - k[i])
    out[np.arange(len(y)), i] = 1 - t
    out[np.arange(len(y)), i + 1] = t
    return out


def _wls(B, y, w, ridge=1e-3, prior=None):
    """Weighted least squares with a small ridge pull towards `prior` (keeps unobserved points neutral)."""
    prior = np.zeros(B.shape[1]) if prior is None else prior
    Bw = B * w[:, None]
    lhs = B.T @ Bw + ridge * w.sum() / len(w) * np.eye(B.shape[1])
    return np.linalg.solve(lhs, Bw.T @ y + ridge * w.sum() / len(w) * prior)


def _colour(p, t, wt):
    """Colour nodes taking the primary's output p (3, M) towards the target t (3, M); wt (M,) point weights."""
    yp, hp, cp, _, _ = opponent(p)
    _, ht, ct, _, _ = opponent(t)
    ok = (cp > 0.01) & (ct > 0.005) & (wt > 0)
    out = []
    if ok.sum() < 50:
        return out
    w = (cp * wt)[ok]
    # hue_vs_hue on the primary's hue
    dh = (ht - hp + 180) % 360 - 180
    shift = np.clip(_wls(_hue_basis(hp[ok]), dh[ok], w), -90, 90)
    hue_pts = [[h, round(float(v), 2)] for h, v in zip(HUES, shift) if abs(v) >= 0.3]
    h2 = (hp + _hue_basis(hp) @ np.where(np.abs(shift) >= 0.3, shift, 0)) % 360
    # saturation x hue_vs_sat(hue after the shift) x lum_vs_sat(luma), alternating weighted least squares
    rho = (ct / np.maximum(cp, 1e-9))[ok]
    Bh, Bl = _hue_basis(h2[ok]), _lum_basis(yp[ok])
    g = np.ones(len(LUM_KNOTS))
    for _ in range(6):
        v = _wls(Bh, rho / (Bl @ g), w, prior=np.ones(len(HUES)) * np.median(rho))
        g = _wls(Bl, rho / (Bh @ v), w, prior=np.ones(len(LUM_KNOTS)))
    g = np.clip(g, 0.0, 3.0)
    at_grey = float(np.interp(GREY, LUM_KNOTS, g))
    s = float(np.exp(np.average(np.log(np.maximum(v, 1e-3)), weights=np.bincount(
        np.argmax(Bh, 1), weights=w, minlength=len(HUES)) + 1e-9))) * at_grey
    v, g = v * at_grey / s, g / at_grey
    if abs(s - 1) >= 0.003:
        out.append({"id": "colour", "call": f"saturation(amount: {_num(min(max(s, 0), 3), 4)})",
                    "label": "overall saturation"})
    if hue_pts:
        out.append({"id": "hue_shift", "call": f"hue_vs_hue(points: {_pts([p + [30] for p in hue_pts], (0, 2, 0))})",
                    "label": "hue shifts"})
    sat_pts = [[h, round(float(np.clip(x, 0, 3)), 3), 30] for h, x in zip(HUES, v) if abs(x - 1) >= 0.008]
    if sat_pts:
        out.append({"id": "hue_sat", "call": f"hue_vs_sat(points: {_pts(sat_pts, (0, 3, 0))})",
                    "label": "saturation by hue"})
    if np.abs(g - 1).max() >= 0.008:
        out.append({"id": "lum_sat", "call": f"lum_vs_sat(points: {_pts(np.stack([LUM_KNOTS, g], 1), (4, 3))})",
                    "label": "saturation by luma"})
    return out


def _hue_lum(q, t, wt):
    """hue_vs_lum for the luma that is still off after the colour nodes (q: their output)."""
    yq, hq, cq, _, _ = opponent(q)
    w = smoothstep(0.0, 0.06, cq)
    ok = (w > 0.05) & (wt > 0)
    if ok.sum() < 50:
        return []
    B = _hue_basis(hq[ok]) * w[ok, None]
    off = np.clip(_wls(B, (luma(t) - yq)[ok], wt[ok]), -0.3, 0.3)
    pts = [[h, round(float(v), 4), 30] for h, v in zip(HUES, off) if abs(v) >= 0.002]
    if not pts:
        return []
    return [{"id": "hue_lum", "call": f"hue_vs_lum(points: {_pts(pts, (0, 4, 0))})", "label": "brightness by hue"}]


# ---------------------------------------------------------------- masked nodes (hints)

def _wants_qualifier(hints):
    out = []
    for i, h in enumerate((hints or {}).get("nodes", []) or []):
        tools = " ".join(str(t) for t in (h.get("tools") or [])).lower()
        if any(k in tools for k in QUALIFIER_TOOLS):
            out.append(h.get("label") or f"node {h.get('index', i + 1)}")
    return out


def _region(q, t):
    """The points the fit misses most (q: fit output, t: target): a qualifier string for them, or None."""
    err = np.abs(q - t).max(0)
    thr = max(0.004, 4 * float(np.median(err)))
    bad = err > thr
    if bad.mean() < 0.002:
        return None
    y, h, c, _, _ = opponent(q)
    hb, cb, yb = h[bad], c[bad], y[bad]
    ang = np.radians(hb)
    centre = math.degrees(math.atan2(np.average(np.sin(ang), weights=cb), np.average(np.cos(ang), weights=cb))) % 360
    d = (hb - centre + 180) % 360 - 180
    lo_d, hi_d = np.quantile(d, [0.05, 0.95])
    lo_d, hi_d = min(lo_d, -5.0), max(hi_d, 5.0)  # never an empty hue range
    if hi_d - lo_d > 300:
        hue = None
    else:
        hue = (round((centre + lo_d) % 360, 1), round((centre + hi_d) % 360, 1))
    sat = np.clip(np.quantile(cb, [0.05, 0.95]), 0, 1)
    lum = np.clip(np.quantile(yb, [0.05, 0.95]), -0.1, 1.5)
    parts = []
    if hue is not None:
        parts.append(f"hue: {_num(hue[0], 1)}..{_num(hue[1], 1)}")
    if sat[1] - sat[0] > 1e-3:
        parts.append(f"sat: {_num(sat[0], 3)}..{_num(sat[1], 3)}")
    if lum[1] - lum[0] > 1e-3:
        parts.append(f"lum: {_num(lum[0], 3)}..{_num(lum[1], 3)}")
    return f"qualifier({', '.join(parts)}, soft: 0.05)" if parts else None


def _masked(base, x, t, label):
    """A qualifier-masked node after `base` where it misses most, fitted on the points: node dict or None."""
    from scipy.optimize import minimize_scalar
    q = _run(base, x)
    mask = _region(q, t)
    if mask is None:
        return None
    best = None
    for op, lo, hi, call in (("saturation", 0.2, 3.0, "saturation(amount: {})"),
                             ("white_balance", -1.0, 1.0, "white_balance(temperature: {})"),
                             ("exposure", -1.0, 1.0, "exposure(stops: {})")):
        def cost(v, call=call):
            n = {"id": "keyed", "call": call.format(_num(v)), "mask": mask}
            return float(np.square(_run(base + [n], x) - t).sum())
        r = minimize_scalar(cost, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3})
        if best is None or r.fun < best[0]:
            best = (r.fun, call.format(_num(r.x)))
    if best[0] >= float(np.square(q - t).sum()) * 0.9:
        return None
    return {"id": "keyed", "call": best[1], "mask": mask, "label": f"keyed correction ({label})"}


# ---------------------------------------------------------------- measuring

def residual(prog, lut, input="acescct", dmin=(0, 0, 0), dmax=(1, 1, 1), frames=()):
    """ΔE00 (mean, p95, max) between the LUT and the program after the program's output: over every lattice node, and
    over frames (images in the `input` domain) when given."""
    to_w, _ = DOMAINS[input]
    dmin, dmax = np.asarray(dmin, float), np.asarray(dmax, float)
    nodes = _nodes(lut.shape[1], dmin, dmax).reshape(3, -1)
    eotf = C.output_eotf(prog.output) or "srgb"
    # an LMT program (output acescct) is compared through the ACES 2.0 SDR output, as it would be watched
    show = C.aces2_display if prog.output.name == "acescct" else (lambda v: C.output_from_working(v, prog.output))

    def de(x_dom, lut_out):
        x = to_w(x_dom)
        a = show(grade_working(x.reshape(3, -1, 1), prog, px_scale=1.0))
        b = show(to_w(lut_out).reshape(3, -1, 1))
        e = C.delta_e00_display(a, b, eotf).ravel()
        return {"mean": round(float(e.mean()), 4), "p95": round(float(np.quantile(e, 0.95)), 4),
                "max": round(float(e.max()), 4)}
    out = {"lattice": de(nodes, lut.reshape(3, -1))}
    if frames is not None and len(frames):
        pix = np.concatenate([np.asarray(f, dtype=np.float64).reshape(3, -1) for f in frames], 1)
        pix = np.clip(pix, dmin[:, None], dmax[:, None])
        out["frames"] = de(pix, apply_lut(lut, pix, dmin, dmax))
    return out


def _score(nodes, x, t):
    """Fit-sample error after the ACES 2.0 output: mean + p95 / 2 (ΔE00)."""
    a = C.aces2_display(_run(nodes, x).reshape(3, -1, 1))
    b = C.aces2_display(t.reshape(3, -1, 1))
    e = C.delta_e00_display(a, b).ravel()
    return float(e.mean() + 0.5 * np.quantile(e, 0.95))


# ---------------------------------------------------------------- joint refinement

HUE_OPS = ("hue_vs_hue", "hue_vs_sat", "hue_vs_lum")


def _slots(prog, chain):
    """The values the refinement may move, as (node id, key, index, lo, hi): every value written in the chain's
    calls, except the hue positions / widths of hue curves and the x of curves (the knots stay where they are)."""
    from .ops import OPS
    out = []
    for n in chain:
        node = prog.nodes[n["id"]]
        spec = OPS[node.call.name]
        for key in re.findall(r"(\w+):", n["call"]):
            p, v = spec.params[key], node.call.args[key]
            if p.kind == "num":
                out.append((n["id"], key, None, p.lo, p.hi))
            elif p.kind == "rgb":
                out += [(n["id"], key, j, p.lo, p.hi) for j in range(3)]
            elif p.kind == "points":
                out += [(n["id"], key, (j, 1), p.cols[1][0], p.cols[1][1]) for j in range(len(v))]
    return out


def _get(prog, s):
    v = prog.nodes[s[0]].call.args[s[1]]
    return v if s[2] is None else v[s[2]] if isinstance(s[2], int) else v[s[2][0]][s[2][1]]


def _set(prog, s, val):
    args = prog.nodes[s[0]].call.args
    if s[2] is None:
        args[s[1]] = float(val)
    elif isinstance(s[2], int):
        v = list(args[s[1]])
        v[s[2]] = float(val)
        args[s[1]] = tuple(v)
    else:
        rows = [list(r) for r in args[s[1]]]
        rows[s[2][0]][s[2][1]] = float(val)
        args[s[1]] = tuple(tuple(r) for r in rows)


def _refine(chain, x, t, max_rounds=6):
    """All the chain's values at once: robust least squares (scipy, soft L1) on CIE Lab differences after the ACES
    2.0 output, which ΔE00 follows closely. The stages fit one after another, and a later stage changes what an
    earlier one should have been (a saturation tints an already tinted neutral axis further); a robust loss keeps a
    region the ops cannot express (a keyed node) from pulling everything else. Returns the refined chain."""
    from scipy.optimize import least_squares
    from .format import fcall
    from .ops import OPS
    prog = load(_text(chain))
    slots = _slots(prog, chain)
    if not slots:
        return chain
    lo = np.array([s[3] for s in slots], dtype=float)
    hi = np.array([s[4] for s in slots], dtype=float)
    x0 = np.clip([float(_get(prog, s)) for s in slots], lo + 1e-9, hi - 1e-9)
    lab_t = C.srgb_to_lab(C.aces2_display(t.reshape(3, -1, 1)))
    code = x.reshape(3, -1, 1)

    def res(v):
        for s, val in zip(slots, v):
            _set(prog, s, val)
        out = C.aces2_display(grade_working(code, prog, px_scale=1.0))
        return (C.srgb_to_lab(out) - lab_t).ravel()
    r = least_squares(res, x0, bounds=(lo, hi), diff_step=1e-3, max_nfev=max_rounds * (len(x0) + 1),
                      x_scale=np.maximum(np.abs(x0), 0.01), loss="soft_l1", f_scale=1.0,
                      tr_solver="lsmr" if len(x0) > 8 else "exact",
                      ftol=1e-5, xtol=1e-6)
    for s, val in zip(slots, r.x):
        _set(prog, s, round(float(val), 5))
    return [dict(n, call=fcall(prog.nodes[n["id"]].call, OPS[prog.nodes[n["id"]].call.name])) for n in chain]

# ---------------------------------------------------------------- fitting

def fit_lut(lut, input="acescct", hints=None, frames=None, name="", about="", sample=17):
    """(Program, report) for a 3D LUT. lut: a .cube path, or an array (3, N, N, N) indexed [channel, b, g, r] over
    the domain 0..1 (cierra.compile.lut.bake's layout). input: the LUT's domain (DOMAINS). hints: {"nodes": [{"index",
    "label", "tools"}]} from the application's node list (a node using a qualifier may add a masked node). frames:
    images (3, H, W) in the domain to report the residual on. sample: lattice points per axis the fit uses."""
    if input not in DOMAINS:
        raise LUTFitError(f"unknown LUT domain {input!r} (one of {', '.join(DOMAINS)})")
    to_w, _ = DOMAINS[input]
    if isinstance(lut, (str, Path)):
        lut, dmin, dmax = read_cube(lut)
    else:
        lut, dmin, dmax = np.asarray(lut, dtype=np.float64), np.zeros(3), np.ones(3)
        if lut.ndim != 4 or lut.shape[0] != 3 or not lut.shape[1] == lut.shape[2] == lut.shape[3] >= 2:
            raise LUTFitError(f"a LUT array is (3, N, N, N), not {lut.shape}")
    size = lut.shape[1]
    # the neutral axis: every diagonal node
    diag = np.arange(size)
    dom_axis = dmin[:, None] + np.linspace(0, 1, size)[None] * (dmax - dmin)[:, None]
    x_axis = to_w(dom_axis).mean(0)
    n_axis = to_w(lut[:, diag, diag, diag])
    x, t = _sample(lut, dmin, dmax, sample, to_w)
    xs, ts = _sample(lut, dmin, dmax, max(5, sample // 2 + 1), to_w)  # the joint refinement's (smaller) sample
    xn, tn = _natural(lut, dmin, dmax, input)
    x, t = np.concatenate([x, xn], 1), np.concatenate([t, tn], 1)
    xs, ts = np.concatenate([xs, xn[:, ::3]], 1), np.concatenate([ts, tn[:, ::3]], 1)
    wt = np.ones(x.shape[1])

    candidates = {}
    par = _parametric(x_axis, n_axis)
    if par:
        candidates["parametric"] = par
    candidates["curves"] = _curves(x_axis, n_axis)
    candidates["luma"] = _luma_wheels(x_axis, n_axis)
    fitted, scores = {}, {}
    for key, prim in candidates.items():
        chain = prim + _colour(_run(prim, x), t, wt)
        chain += _hue_lum(_run(chain, x), t, wt)
        fitted[key], scores[key] = chain, _score(chain, x, t)
    # a parametric primary reads best (and is a CDL): kept unless another fits clearly better
    best = min(scores, key=lambda k: scores[k] - (0.05 if k == "parametric" else 0.0))
    keep = len(candidates[best])
    chain = fitted[best]
    if best == "parametric":
        # a saturation after a tinted primary tints the neutral axis further: also try settling the primary and the
        # overall saturation together first (a few values), the rest of the colour fitted on what they leave, and
        # go on with whichever start fits better
        pre = _refine(candidates[best] + [n for n in chain if n["id"] == "colour"], xs, ts)
        rest = [n for n in _colour(_run(pre, x), t, wt) if n["id"] != "colour"]
        rest += _hue_lum(_run(pre + rest, x), t, wt)
        if _score(pre + rest, x, t) < scores[best]:
            chain = pre + rest
    chain = _settle(chain, x, t, xs, ts, keep)
    if best == "parametric" and [n["id"] for n in chain] == ["cdl", "colour"]:  # cdl then saturation: one CDL
        amt = chain[1]["call"].split("amount: ")[1].rstrip(")")
        chain = [dict(chain[0], call=chain[0]["call"][:-1] + f", sat: {amt})")]
        keep = 1
    used, trials = [], {"none": round(_score(chain, x, t), 4)}
    for label in _wants_qualifier(hints)[:1]:  # one masked node at most: kept simple, and measured
        # two places for it, both measured: after the whole fit, keyed where it misses most; or right after the
        # primary, keyed where the primary alone misses most, with the colour nodes fitted again after it
        prim = candidates[best]
        options = {}
        node = _masked(chain, x, t, label)
        if node is not None:
            options["last"] = chain + [node]
        node = _masked(prim, x, t, label)
        if node is not None:
            base = prim + [node]
            base += _colour(_run(base, x), t, wt)
            options["after_primary"] = base + _hue_lum(_run(base, x), t, wt)
        for key, trial in options.items():
            trial = _settle(trial, x, t, xs, ts, keep, protect="keyed")
            trials[key] = round(_score(trial, x, t), 4)
            if trials[key] < _score(chain, x, t) * 0.95 and "keyed" in [n["id"] for n in trial]:
                chain, used = trial, [label]
    prog = load(_text(chain, input, name, about))
    report = residual(prog, lut, input, dmin, dmax, frames)
    report.update({"primary": best, "candidates": {k: round(v, 4) for k, v in scores.items()},
                   "nodes": [n["id"] for n in chain], "keyed": used, "domain": input, "size": size})
    if len(trials) > 1:
        report["keyed_trials"] = trials
    return prog, report


def _sample(lut, dmin, dmax, per_axis, to_w):
    """(x, t): ACEScct inputs and outputs at a sub-lattice of the LUT's nodes (the nodes themselves, no
    interpolation), about per_axis points per axis, both ends included."""
    size = lut.shape[1]
    step = max(1, (size - 1) // max(per_axis - 1, 1))
    sub = np.arange(0, size, step)
    if sub[-1] != size - 1:
        sub = np.append(sub, size - 1)
    nodes = _nodes(size, dmin, dmax)[:, sub][:, :, sub][:, :, :, sub].reshape(3, -1)
    return to_w(nodes), to_w(lut[:, sub][:, :, sub][:, :, :, sub].reshape(3, -1))


def _natural(lut, dmin, dmax, input, n=3000):
    """(x, t) at n fixed pseudo-random colours like those in pictures (luma 0.02..0.95 ACEScct, chroma mostly
    under 0.1), the LUT interpolated there: a uniform lattice is mostly saturated colours no picture has, so the
    fit would favour them."""
    from .engine import from_opponent
    to_w, from_w = DOMAINS[input]
    rng = np.random.default_rng(20261028)
    y = rng.uniform(0.02, 0.95, n)
    c = np.abs(rng.normal(0.0, 0.05, n))
    h = rng.uniform(0, 2 * np.pi, n)
    dom = np.clip(from_w(from_opponent(y, c * np.cos(h), c * np.sin(h))), dmin[:, None], dmax[:, None])
    return to_w(dom), to_w(apply_lut(lut, dom, dmin, dmax))


def _settle(chain, x, t, xs, ts, keep, protect=None):
    """Prune, refine jointly, prune again, and refine once more when the second pruning took a node out."""
    chain = _refine(_prune(chain, x, t, keep, protect), xs, ts)
    pruned = _prune(chain, x, t, keep, protect)
    return _refine(pruned, xs, ts) if len(pruned) < len(chain) else pruned


def _prune(chain, x, t, keep, protect=None):
    """Leave out the nodes after the first `keep` (the primary) that the fit does not need: a node goes when it moves
    no sample by 0.001 cv, or when the error without it is within 2 % + 0.02 ΔE00 of the error with it (50 times
    below a just visible difference)."""
    chain = list(chain)
    i = keep
    while i < len(chain):
        if chain[i]["id"] != protect:
            without = chain[:i] + chain[i + 1:]
            if (np.abs(_run(without, x) - _run(chain, x)).max() < 0.001
                    or _score(without, x, t) <= _score(chain, x, t) * 1.02 + 0.02):
                chain = without
                continue
        i += 1
    return chain
