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

fit_sequence(frames, reference, shots) fits a whole sequence: one curve per shot, fitted on the shot's pooled frames
(curve points cannot be keyframed in Cierra 0.1), and per-frame white balance / saturation that are median-smoothed
within the shot and written as keyframes only where they change by more than a visible step, so the grade does not
flicker. A shot boundary resets the smoothing (a cut may change everything); without the host's cuts, detect_shots
splits at jumps in mean level, so a flash frame gets its own fit instead of skewing the shot's curve.
"""

import numpy as np

from .checker import load
from .engine import render
from .format import format_program

QUANTILES = (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97)
CAPS = {"temperature": 0.3, "tint": 0.3, "saturation": 0.4}


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


def _program(curve, temperature=0.0, tint=0.0, saturation=1.0, input="linear_rec709"):
    pts = ", ".join(f"[{x:g}, {y:g}]" for x, y in curve)
    return load(f"cierra 0.1\ninput {input}\n"
                f"node match_tone = curve(points: [{pts}])\n"
                f"  label \"tone matched to the reference (luma histogram)\"\n"
                f"node match_balance = white_balance(temperature: {temperature:.4g}, tint: {tint:.4g})\n"
                f"node match_colour = saturation(amount: {saturation:.4g})\n"
                "graph match_balance -> match_tone -> match_colour\n")


def _solve(f, target, lo, hi, x0=0.0, steps=6):
    """x in [lo, hi] with f(x) ~ target by secant steps from x0 (f monotone increasing in x)."""
    xa, fa = x0, f(x0)
    xb = min(hi, max(lo, x0 + 0.05 * (hi - lo)))
    fb = f(xb)
    for _ in range(steps):
        if abs(fb - fa) < 1e-12:
            break
        xc = min(hi, max(lo, xb + (target - fb) * (xb - xa) / (fb - fa)))
        xa, fa, xb, fb = xb, fb, xc, f(xc)
        if abs(fb - target) < 1e-5:
            break
    return round(xb, 4)


def fit_reference(image, reference, input="linear_rec709", curve=None):
    """(Program, info). image: the frame in `input`; reference: a display image (3, H, W) in 0..1 of the look;
    curve: a fixed tone curve (a shot's), else fitted on this frame."""
    ref = np.clip(np.asarray(reference, dtype=np.float64), 0, 1)
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
    prog = _program(curve, temp, tint, sat, input)
    return prog, {"curve": curve, "temperature": temp, "tint": tint, "saturation": sat}


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
    shots = shots or detect_shots(frames, input)
    out = {}
    for s in sorted(set(shots)):
        idx = [i for i, k in enumerate(shots) if k == s]
        imgs = [frames[i][1] for i in idx]
        nums = [frames[i][0] for i in idx]
        pooled = np.concatenate([_working(im, input).reshape(3, -1)
                                 for im in imgs], axis=1)[:, :, None]
        curve = luma_curve(pooled, to_code(reference))
        per = [fit_reference(im, reference, input, curve=curve)[1] for im in imgs]
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
