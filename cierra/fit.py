"""Reference fitting: turn "make it look like this reference" into Cierra's existing parameters, not into a LUT.

fit_reference(image, reference) returns a readable program:
  curve(channel: luma)                   from histogram matching of ACEScct luma, T = H_ref^-1(H_in), sampled at
                                         fixed quantiles of the input; the reference is taken back to ACEScct through
                                         the inverse ACES 2.0 output (ill-conditioned near white: the top quantile is
                                         left out) (Bonneel et al., "Example-Based Video Color
                                         Grading", SIGGRAPH 2013, use the same transfer idea per shot);
  white_balance(temperature, tint)      from the mean opponent colour of the mid-tones, solved by a few secant steps
                                         on the rendered result (deterministic);
  saturation(amount)                     from the mean chroma ratio, also solved on the rendered result.
All three are measured on the program's own output (display values), so whatever the output transform does is
included. Values stay inside the tuner's gentle caps (cierra.tuner.CAPS) and the curve is monotone by construction.

Then, on what those three leave (the program's ACEScct output against the reference taken back to ACEScct, pixels the
reference clips left out), one stage after another, each on the output of the stages kept before it:
  hue_vs_hue    hue angle difference, weighted by chroma;
  hue_vs_sat    chroma out = gain(hue) x chroma in;
  hue_vs_lum    the luma still off, by hue (where the op acts: chroma above ~0.06);
  lum_vs_sat    chroma out = gain(luma) x chroma in, 6 knots over ACEScct luma.
The hue stages use 6 points 60 degrees apart, width 60 (raised cosines that sum to one, so each fit is linear least
squares). Each point is pulled towards identity with RIDGE (5 %) of an average point's data weight: a hue the frame
hardly shows stays at identity. Points within a small step of identity (1 degree, gain 0.02, offset 0.004) are left
out. A stage is kept only when the mean CIEDE2000 of the rendered program against the reference falls by at least
MIN_GAIN (0.25) and by a tenth of the error left. 0.25 is a quarter of a just-noticeable difference averaged over the
frame and above what a stage gains from the base fit's own small misses on a simple reference (0.15 on the crowd
frame for a white balance / curve / saturation look): such references keep exactly the program they had without
the stages. A reference whose tone the base fit misses by hue (a frame of mostly one hue) may still gain a
hue_vs_lum.

fit_sequence(frames, reference, shots) fits a whole sequence: one curve per shot, fitted on the shot's pooled frames
(curve points cannot be keyframed in Cierra 0.1), and per-frame white balance / saturation that are median-smoothed
within the shot and written as keyframes only where they change by more than a visible step, so the grade does not
flicker. A shot boundary resets the smoothing (a cut may change everything); without the host's cuts, detect_shots
splits at jumps in mean level, so a flash frame gets its own fit instead of skewing the shot's curve. Sequence fits
do not write the per-hue / per-brightness stages (their points cannot be keyframed either, and a stage kept on one
frame and not the next would flicker).
"""

import numpy as np

from . import colour as C
from .checker import load
from .engine import GREY, grade_working, opponent, render, smoothstep
from .errors import CierraError
from .format import format_program

QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)
CAPS = {"temperature": 0.3, "tint": 0.3, "saturation": 0.4}
STAGES = ("hue_vs_hue", "hue_vs_sat", "hue_vs_lum", "lum_vs_sat")
HUES = np.arange(0.0, 360.0, 60.0)  # 6 hue points; width 60: their raised cosines sum to one (linear least squares)
HUE_WIDTH = 60.0
LUM_KNOTS = np.array([0.0, 0.2, GREY, 0.6, 0.8, 1.0])
RIDGE = 0.05         # pull of each control point towards identity, as a share of an average point's data weight
MIN_GAIN = 0.25      # dE00: a stage is kept only when it lowers the mean error this much ...
MIN_GAIN_REL = 0.1   # ... and by a tenth of the error left


def _luma(d):
    return 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]


def _opp(d):
    y = _luma(d)
    return y, (d[2] - y) / 1.8556, (d[0] - y) / 1.5748


def _mids(d):
    y = _luma(d)
    lo, hi = np.quantile(y, [0.2, 0.8])
    return (y >= lo) & (y <= hi)


def luma_curve(src_code, ref_code):
    """ACEScct luma curve points [[x, y], ...] mapping src's luma distribution onto ref's (monotone; outside the
    quantiles the curve op extends with its end slopes)."""
    ys, yr = _luma(src_code).ravel(), _luma(ref_code).ravel()
    pts = []
    for q in QUANTILES:
        x, y = float(np.quantile(ys, q)), float(np.quantile(yr, q))
        if not pts or x - pts[-1][0] >= 0.01:
            pts.append([round(x, 4), round(max(y, pts[-1][1]) if pts else y, 4)])
    if len(pts) < 2:
        pts = [[0.0729, 0.0729], [1.0, 1.0]]
    return pts


def to_code(display):
    """A display image (ACES 2.0 sRGB output) back to ACEScct working values."""
    from . import colour as C
    return C.aces2_inverse(np.ascontiguousarray(np.clip(display, 0, 1)))


def _working(image, input):
    from .colour import input_to_working
    return input_to_working(np.asarray(image, dtype=np.float64), input)[0]


def _program(curve, temperature=0.0, tint=0.0, saturation=1.0, input="linear_rec709", extra=()):
    """extra: (id, call, label) nodes chained after match_colour (the per-hue / per-brightness stages)."""
    pts = ", ".join(f"[{x:g}, {y:g}]" for x, y in curve)
    nodes = "".join(f"node {i} = {call}\n  label \"{label}\"\n" for i, call, label in extra)
    chain = "".join(f" -> {i}" for i, _, _ in extra)
    return load(f"cierra 0.1\ninput {input}\n"
                f"node match_tone = curve(points: [{pts}])\n"
                f"  label \"tone matched to the reference (luma histogram)\"\n"
                f"node match_balance = white_balance(temperature: {temperature:.4g}, tint: {tint:.4g})\n"
                f"node match_colour = saturation(amount: {saturation:.4g})\n"
                f"{nodes}graph match_balance -> match_tone -> match_colour{chain}\n")


# ---------------------------------------------------------------- per-hue / per-brightness stages

def _hue_basis(hue):
    d = (hue[:, None] - HUES[None, :] + 180) % 360 - 180
    return 0.5 + 0.5 * np.cos(np.pi * np.clip(d / HUE_WIDTH, -1, 1))


def _lum_basis(y):
    k = LUM_KNOTS
    yc = np.clip(y, k[0], k[-1])
    i = np.clip(np.searchsorted(k, yc) - 1, 0, len(k) - 2)
    t = (yc - k[i]) / (k[i + 1] - k[i])
    out = np.zeros((len(y), len(k)))
    out[np.arange(len(y)), i] = 1 - t
    out[np.arange(len(y)), i + 1] = t
    return out


def _ridge(B, y, w, neutral):
    """Weighted least squares pulled towards `neutral`: each control point gets a pseudo-observation at neutral
    worth RIDGE times the data weight of an average point, so a hue or brightness the frame hardly shows stays at
    identity and a well covered one is biased by a few per cent at most."""
    Bw = B * w[:, None]
    lhs = B.T @ Bw
    lam = RIDGE * np.trace(lhs) / B.shape[1]
    return np.linalg.solve(lhs + lam * np.eye(B.shape[1]), Bw.T @ y + lam * neutral)


def _points(vals, neutral, tol, nd):
    return [[int(h), round(float(v), nd), int(HUE_WIDTH)] for h, v in zip(HUES, vals) if abs(v - neutral) >= tol]


def _fmt(pts):
    return "[" + ", ".join("[" + ", ".join(f"{v:g}" for v in p) + "]" for p in pts) + "]"


def _stage(kind, q, t, ok):
    """(id, call, label) or None: one stage fitted on the program's working output q (3, N) against the reference
    t (3, N) taken back to ACEScct; ok (N,) the pixels the reference shows unclipped."""
    yq, hq, cq, _, _ = opponent(q)
    yt, ht, ct, _, _ = opponent(t)
    if kind == "hue_vs_hue":
        m = ok & (cq > 0.01) & (ct > 0.01)
        if m.sum() < 50:
            return None
        dh = (ht - hq + 180) % 360 - 180
        v = np.clip(_ridge(_hue_basis(hq[m]), dh[m], cq[m], 0.0), -90, 90)
        pts = _points(v, 0.0, 1.0, 1)
        return pts and ("match_hue", f"hue_vs_hue(points: {_fmt(pts)})", "hue shifts")
    if kind in ("hue_vs_sat", "lum_vs_sat"):
        m = ok & (cq > 0.005)
        if m.sum() < 50:
            return None
        basis = _hue_basis(hq[m]) if kind == "hue_vs_sat" else _lum_basis(yq[m])
        # chroma out = gain(hue or luma) x chroma in, least squares on chroma itself (weights chroma^2 implicitly)
        g = np.clip(_ridge(basis * cq[m, None], ct[m], np.ones(m.sum()), 1.0), 0, 3)
        if kind == "hue_vs_sat":
            pts = _points(g, 1.0, 0.02, 3)
            return pts and ("match_hue_sat", f"hue_vs_sat(points: {_fmt(pts)})", "saturation by hue")
        if np.abs(g - 1).max() < 0.02:
            return None
        pts = [[float(k), round(float(x), 3)] for k, x in zip(LUM_KNOTS, g)]
        return "match_lum_sat", f"lum_vs_sat(points: {_fmt(pts)})", "saturation by brightness"
    # hue_vs_lum: the luma still off, by hue, where the op acts (chroma above ~0.06)
    s = smoothstep(0.0, 0.06, cq)
    m = ok & (s > 0.05)
    if m.sum() < 50:
        return None
    v = np.clip(_ridge(_hue_basis(hq[m]) * s[m, None], (yt - yq)[m], np.ones(m.sum()), 0.0), -0.3, 0.3)
    pts = _points(v, 0.0, 0.004, 4)
    return pts and ("match_hue_lum", f"hue_vs_lum(points: {_fmt(pts)})", "brightness by hue")


def _de(prog, image, ref, input):
    return float(C.delta_e00_display(render(prog, image, input=input), ref, "srgb").mean())


def _fine(image, ref, base, input):
    """The stages kept after the base program's curve / balance / saturation (see the module docstring), the
    error before and after."""
    code, display_referred = C.input_to_working(np.asarray(image, dtype=np.float64), input)
    t = to_code(ref).reshape(3, -1)
    ok = np.all((ref > 0.004) & (ref < 0.996), axis=0).ravel() & np.all(np.isfinite(t), axis=0)
    extra, de0 = [], _de(base(()), image, ref, input)
    de = de0
    for kind in STAGES:
        q = grade_working(code, base(extra), display_referred=display_referred).reshape(3, -1)
        st = _stage(kind, q, t, ok)
        if not st:
            continue
        d = _de(base(extra + [st]), image, ref, input)
        if de - d >= max(MIN_GAIN, MIN_GAIN_REL * de):
            extra, de = extra + [st], d
    return extra, de0, de


def _solve(f, target, lo, hi, x0=0.0, steps=6):
    """x in [lo, hi] with f(x) ~ target by secant steps from x0 (f monotone increasing in x)."""
    xa, fa = x0, f(x0)
    xb = min(hi, max(lo, x0 + 0.05 * (hi - lo)))
    fb = f(xb)
    for _ in range(steps):
        if abs(fb - fa) < 1e-12:  # flat: keep the better end (x0 when nothing moves)
            if abs(fa - target) <= abs(fb - target):
                xb = xa
            break
        xc = min(hi, max(lo, xb + (target - fb) * (xb - xa) / (fb - fa)))
        xa, fa, xb, fb = xb, fb, xc, f(xc)
        if abs(fb - target) < 1e-5:
            break
    return round(xb, 4)


def _size(a):
    shape = np.shape(a)
    return f"{shape[-1]} x {shape[-2]}" if len(shape) == 3 and shape[0] == 3 else f"of shape {shape}"


def fit_reference(image, reference, input="linear_rec709", curve=None, fine=True):
    """(Program, info). image: the frame in `input`; reference: a display image (3, H, W) in 0..1 of the look;
    curve: a fixed tone curve (a shot's), else fitted on this frame; fine: also fit the per-hue / per-brightness
    stages (info["stages"]: the ones kept, info["de00"]: mean error without and with them)."""
    ref = np.clip(np.asarray(reference, dtype=np.float64), 0, 1)
    if fine and ref.shape != np.shape(image):  # the stages pair image and reference pixel by pixel
        raise CierraError(f"the reference is {_size(ref)} and the image {_size(image)}: fitting the per-hue and "
                          "per-brightness stages needs the reference to be the same frame, at the same size")
    if curve is None:
        curve = luma_curve(_working(image, input), to_code(ref))
    _, rcb, rcr = _opp(ref)
    rm = _mids(ref)
    t_warm, t_tint = float((rcr - rcb)[rm].mean()), float(((rcr + rcb) / 2)[rm].mean())

    def measure(p):
        out = render(p, image, input=input)
        _, cb, cr = _opp(out)
        m = _mids(out)
        return out, float((cr - cb)[m].mean()), float(((cr + cb) / 2)[m].mean())

    temp = _solve(lambda t: measure(_program(curve, t, 0.0, 1.0, input))[1], t_warm, -CAPS["temperature"],
                  CAPS["temperature"])
    tint = _solve(lambda n: -measure(_program(curve, temp, n, 1.0, input))[2], -t_tint, -CAPS["tint"], CAPS["tint"])

    def chroma(p):
        _, cb, cr = _opp(render(p, image, input=input))
        return float(np.hypot(cb, cr).mean())

    want = float(np.hypot(*_opp(ref)[1:]).mean())
    sat = _solve(lambda s: chroma(_program(curve, temp, tint, s, input)), want, 1 - CAPS["saturation"],
                 1 + CAPS["saturation"], x0=1.0)
    info = {"curve": curve, "temperature": temp, "tint": tint, "saturation": sat}
    if not fine:
        return _program(curve, temp, tint, sat, input), info
    extra, de0, de = _fine(image, ref, lambda ex: _program(curve, temp, tint, sat, input, ex), input)
    info.update(stages=[i for i, _, _ in extra], de00=[round(de0, 3), round(de, 3)])
    return _program(curve, temp, tint, sat, input, extra), info


def _smooth(vals, window=5):
    v = np.asarray(vals, dtype=float)
    h = window // 2
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def _keys(frames, vals, step):
    """keys(linear, ...) text through the smoothed values, a key only where the value moved more than `step` since
    the last key (plus the first and last frame); a constant when nothing moves."""
    if np.ptp(vals) <= step:
        return f"{float(np.mean(vals)):.4g}"
    keys, last = [(frames[0], vals[0])], vals[0]
    for f, v in zip(frames[1:-1], vals[1:-1]):
        if abs(v - last) > step:
            keys.append((f, v))
            last = v
    keys.append((frames[-1], vals[-1]))
    return "keys(linear, " + ", ".join(f"{f}: {v:.4g}" for f, v in keys) + ")"


def detect_shots(frames, input="linear_rec709", jump=0.045):
    """Shot index per frame: a new shot where the mean ACEScct luma jumps by more than `jump` (~0.8 stop) from the
    previous frame (a cut or a flash). Hosts that know their cuts pass them to fit_sequence instead."""
    levels = [float(_luma(_working(im, input)).mean()) for _, im in frames]
    shots = [0]
    for a, b in zip(levels, levels[1:]):
        shots.append(shots[-1] + (abs(b - a) > jump))
    return shots


def fit_sequence(frames, reference, shots=None, input="linear_rec709", step=0.01):
    """{shot index: Program} for a list of (frame number, image) and one reference look. shots: list of shot
    indices per frame (default: detect_shots). Per shot: the curve from the shot's pooled frames, white balance and
    saturation fitted per frame, median-smoothed within the shot, then keyframed where they move more than `step`."""
    shots = detect_shots(frames, input) if shots is None else list(shots)
    if len(shots) != len(frames):
        raise ValueError(f"{len(shots)} shot indices for {len(frames)} frames")
    out = {}
    for s in sorted(set(shots)):
        idx = [i for i, k in enumerate(shots) if k == s]
        imgs = [frames[i][1] for i in idx]
        nums = [frames[i][0] for i in idx]
        pooled = np.concatenate([_working(im, input).reshape(3, -1)
                                 for im in imgs], axis=1)[:, :, None]
        curve = luma_curve(pooled, to_code(reference))
        per = [fit_reference(im, reference, input, curve=curve, fine=False)[1] for im in imgs]
        params = {k: _smooth([p[k] for p in per]) for k in ("temperature", "tint", "saturation")}
        prog = _program(curve, 0.0, 0.0, 1.0, input)
        text = format_program(prog)
        bal = (f"white_balance(temperature: {_keys(nums, params['temperature'], step)}, "
               f"tint: {_keys(nums, params['tint'], step)})")
        text = text.replace(text.split("node match_balance = ")[1].split("\n")[0], bal)
        sat = f"saturation(amount: {_keys(nums, params['saturation'], step)})"
        text = text.replace(text.split("node match_colour = ")[1].split("\n")[0], sat)
        out[s] = load(text)
    return out
