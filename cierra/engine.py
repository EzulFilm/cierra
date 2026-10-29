"""The reference engine: runs a checked program on an image in float64, with NumPy on the CPU or CuPy on a GPU
(cierra.backend; the same code, and the GPU run is tested against the CPU run). This is the source of truth; every
compiled target is tested against it.

Every op works on ACEScct code values shaped (3, H, W). The maths of each op is written out in docs/spec.md section 5;
the code follows it line for line.
"""

import math

import numpy as np

from . import ast as A
from .backend import const, ndimage, to_device, to_host, xp
from .backend import device as pick_device
from . import colour as C
from .ops import GREY, OPS

STOP = 1.0 / 17.52  # ACEScct code values per stop above the toe
NEUTRAL_C = 1e-5  # opponent chroma below which a qualifier treats a pixel as grey (hue 0) on every target


# ---------------------------------------------------------------- helpers

def luma(code):
    return 0.2126 * code[0] + 0.7152 * code[1] + 0.0722 * code[2]


def opponent(code):
    """Luma, hue (deg, 0..360), chroma, Cb, Cr of the working values."""
    y = luma(code)
    cb = (code[2] - y) / 1.8556
    cr = (code[0] - y) / 1.5748
    nx = xp(code)
    hue = nx.degrees(nx.arctan2(cr, cb)) % 360.0
    return y, hue, nx.sqrt(cb * cb + cr * cr), cb, cr


def from_opponent(y, cb, cr):
    r = y + 1.5748 * cr
    b = y + 1.8556 * cb
    g = (y - 0.2126 * r - 0.0722 * b) / 0.7152
    return xp(y, cb, cr).stack([r, g, b])


def smoothstep(e0, e1, x):
    t = xp(x).clip((x - e0) / max(e1 - e0, 1e-9), 0, 1)
    return t * t * (3 - 2 * t)


def band(x, lo, hi, soft):
    return smoothstep(lo - soft, lo + soft, x) * (1 - smoothstep(hi - soft, hi + soft, x))


def curve_tangents(points):
    """Fritsch-Carlson tangents for a monotone cubic through sorted points [(x, y), ...] -> (xs, ys, ms)."""
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    n = len(xs)
    d = [(ys[k + 1] - ys[k]) / (xs[k + 1] - xs[k]) for k in range(n - 1)]  # x strictly increasing (checker)
    m = [d[0]] + [(d[k] + d[k - 1]) / 2 for k in range(1, n - 1)] + [d[-1]]
    for k in range(n - 1):
        if d[k] == 0:
            m[k] = m[k + 1] = 0.0
        else:
            a, b = m[k] / d[k], m[k + 1] / d[k]
            t = a * a + b * b
            if t > 9:
                s = 3 / math.sqrt(t)
                m[k], m[k + 1] = s * a * d[k], s * b * d[k]
    return xs, ys, m


def monotone_curve(x, points):
    xs, ys, m = curve_tangents(points)
    nx = xp(x)
    xs_a, ys_a, m_a = const(xs, x), const(ys, x), const(m, x)
    n = len(xs)
    xc = nx.clip(x, xs[0], xs[-1])
    i = nx.clip(nx.searchsorted(xs_a, xc, side="left"), 1, n - 1) - 1
    h = xs_a[i + 1] - xs_a[i]
    t = (xc - xs_a[i]) / h
    t2, t3 = t * t, t * t * t
    out = ((2 * t3 - 3 * t2 + 1) * ys_a[i] + (t3 - 2 * t2 + t) * h * m_a[i] + (-2 * t3 + 3 * t2) * ys_a[i + 1]
           + (t3 - t2) * h * m_a[i + 1])
    out = nx.where(x < xs[0], ys[0] + (x - xs[0]) * m[0], out)
    return nx.where(x > xs[-1], ys[-1] + (x - xs[-1]) * m[-1], out)


def hue_curve(hue, points, neutral):
    nx = xp(hue)
    out = nx.full_like(hue, float(neutral))
    weight = nx.zeros_like(hue)
    for pt in points:
        h0, v = pt[0], pt[1]
        width = pt[2] if len(pt) > 2 else 60.0
        dist = ((hue - h0 + 180) % 360) - 180
        w = 0.5 + 0.5 * nx.cos(math.pi * nx.clip(dist / width, -1, 1))
        out = out + w * (v - neutral)
        weight = weight + w
    return neutral + (out - neutral) / nx.maximum(weight, 1.0)


def gaussian_kernel(sigma):
    r = max(1, int(3 * sigma))
    t = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x, sigma):
    """Separable Gaussian, radius int(3 sigma), edges replicated; x is (C, H, W). sigma < 0.3: unchanged."""
    if sigma < 0.3:
        return x
    k, nd = const(gaussian_kernel(sigma), x), ndimage(x)
    y = nd.correlate1d(x, k, axis=-1, mode="nearest")
    return nd.correlate1d(y, k, axis=-2, mode="nearest")


def box_mean(x, r):
    """Mean over a (2r + 1) square window, edges replicated; x is (H, W). Linear time in the radius."""
    r = int(round(r))
    return x if r < 1 else ndimage(x).uniform_filter(x, size=2 * r + 1, mode="nearest")


def guided_filter(guide, src, r, eps):
    """He, Sun and Tang, "Guided Image Filtering" (ECCV 2010), gray guide: q = mean(a) * I + mean(b) with
    a = cov(I, p) / (var(I) + eps), b = mean(p) - a * mean(I) over (2r + 1) windows. Edge-preserving, linear time."""
    mi, mp = box_mean(guide, r), box_mean(src, r)
    a = (box_mean(guide * src, r) - mi * mp) / (box_mean(guide * guide, r) - mi * mi + eps)
    b = mp - a * mi
    return box_mean(a, r) * guide + box_mean(b, r)


def _rgb(v, like):
    return const(v, like).reshape(3, 1, 1)


# ---------------------------------------------------------------- ops

def op_photo_to_scene(code, ctx, recovered_highlights, knee_stops, max_stops):
    nx = xp(code)
    lin = nx.maximum(C.from_working(code), 0)
    over = nx.maximum(lin.max(0, keepdims=True), 1.0)
    dl = lin / over
    disp = nx.clip(C.srgb_encode(dl), 0, 1)
    full = C.aces2_inverse(disp)
    dm = disp.max(0, keepdims=True)
    sat = (dm - disp.min(0, keepdims=True)) / nx.maximum(dm, 1e-6)
    w = smoothstep(0.70, 0.92, dm) * (1 - smoothstep(0.10, 0.30, sat))
    if float(w.max()) > 0:
        yd = (dl * const(C.REC709_Y, code).reshape(3, 1, 1)).sum(0, keepdims=True)
        ys = C.from_working(C.aces2_inverse(nx.ascontiguousarray(
            nx.broadcast_to(nx.clip(C.srgb_encode(yd), 0, 1), dl.shape))))[:1]
        keep = C.to_working(nx.maximum(dl / nx.maximum(yd, 1e-9) * ys, 0))
        full = full + w * (keep - full)
    scene = full
    if recovered_highlights:
        scene = scene + nx.log2(nx.maximum(lin.min(0, keepdims=True), 1.0)) / 17.52
    k, m = GREY + knee_stops / 17.52, GREY + max_stops / 17.52
    top = scene.max(0, keepdims=True)
    rolled = nx.where(top > k, k + (m - k) * nx.tanh((top - k) / (m - k)), top)
    return scene + (rolled - top)


def _linear_gain(code, log2_gain):
    """A linear-light gain on ACEScct code values: decode, multiply by 2^log2_gain (one value, or one per AP1
    channel), encode. Above the ACEScct toe (linear 0.0078, ~4.5 stops under grey) this is adding log2_gain / 17.52 to
    the code; below it ACEScct is a straight line, where adding a constant would lift black (a +1 stop push went milky),
    so the gain is applied in linear light (as the looks' original engine does since 2026-10-27)."""
    g = const(2.0 ** np.asarray(log2_gain, dtype=np.float64), code)
    if g.ndim == 1:
        g = g.reshape(3, *([1] * (code.ndim - 1)))
    return C.acescct_encode(C.acescct_decode(code) * g)


def op_exposure(code, ctx, stops):
    return code if stops == 0 else _linear_gain(code, stops)


def op_white_balance(code, ctx, temperature, tint):
    if temperature == 0 and tint == 0:
        return code
    return _linear_gain(code, [temperature, -tint, -temperature])


def op_printer_lights(code, ctx, red, green, blue):
    s = 0.025 / (17.52 * math.log10(2))  # 1 point = 0.025 log10 exposure = 0.025 / log10(2) stops
    return code + _rgb([red * s, green * s, blue * s], code)


def op_wheels(code, ctx, lift, gamma, gain, offset, master_lift, master_gamma, master_gain, master_offset):
    nx = xp(code)
    y = nx.clip(luma(code), 0, 1)[None]
    shadows = 1 - smoothstep(0.1, GREY + 0.05, y)
    mids = nx.exp(-((y - GREY) / 0.18) ** 2)
    highs = smoothstep(GREY - 0.05, 0.75, y)
    return (code + (_rgb(lift, code) + master_lift) * shadows + (_rgb(gamma, code) + master_gamma) * mids
            + (_rgb(gain, code) + master_gain) * highs + _rgb(offset, code) + master_offset)


def op_cdl(code, ctx, slope, offset, power, sat):
    nx = xp(code)
    v = code * _rgb(slope, code) + _rgb(offset, code)
    v = nx.where(v >= 0, nx.power(nx.maximum(v, 0), _rgb(power, code)), v)
    y = luma(v)[None]
    return y + sat * (v - y)


def op_contrast(code, ctx, amount, pivot):
    return (code - pivot) * amount + pivot


def op_saturation(code, ctx, amount, protect_skin):
    nx = xp(code)
    y, hue, _, cb, cr = opponent(code)
    k = nx.full_like(y, float(amount))
    if protect_skin:
        skin = 1 - smoothstep(15, 40, nx.abs(((hue - 141 + 180) % 360) - 180))
        k = k + (1 - k) * skin * protect_skin
    return from_opponent(y, cb * k, cr * k)


def op_curve(code, ctx, points, channel, space):
    if space == "display":
        nx = xp(code)
        disp = nx.clip(C.display_encode(code), 0, 1)
        return C.from_display(nx.clip(op_curve(disp, ctx, points, channel, "log"), 0, 1))
    if channel == "luma":
        y = luma(code)
        return code + (monotone_curve(y, points) - y)[None]
    if channel == "rgb":
        return monotone_curve(code, points)
    out = code.copy()
    i = {"red": 0, "green": 1, "blue": 2}[channel]
    out[i] = monotone_curve(code[i], points)
    return out


def op_hue_vs_hue(code, ctx, points):
    nx = xp(code)
    y, hue, _, cb, cr = opponent(code)
    shift = nx.radians(hue_curve(hue, points, 0.0))
    cos, sin = nx.cos(shift), nx.sin(shift)
    return from_opponent(y, cb * cos - cr * sin, cb * sin + cr * cos)


def op_hue_vs_sat(code, ctx, points):
    y, hue, _, cb, cr = opponent(code)
    k = hue_curve(hue, points, 1.0)
    return from_opponent(y, cb * k, cr * k)


def op_hue_vs_lum(code, ctx, points):
    y, hue, c, cb, cr = opponent(code)
    dy = hue_curve(hue, points, 0.0) * smoothstep(0.0, 0.06, c)
    return from_opponent(y + dy, cb, cr)


def op_lum_vs_sat(code, ctx, points):
    y, _, _, cb, cr = opponent(code)
    k = monotone_curve(y, points)
    return from_opponent(y, cb * k, cr * k)


def op_sat_vs_sat(code, ctx, points):
    y, _, c, cb, cr = opponent(code)
    k = monotone_curve(c, points) / xp(c).maximum(c, 1e-6)
    return from_opponent(y, cb * k, cr * k)


def op_sat_vs_lum(code, ctx, points):
    y, _, c, cb, cr = opponent(code)
    return from_opponent(y + monotone_curve(c, points), cb, cr)


def op_rgb_mixer(code, ctx, red, green, blue, preserve_luma):
    m = const([red, green, blue], code)
    out = xp(code).einsum("ij,jhw->ihw", m, code)
    if preserve_luma:
        out = out + (luma(code) - luma(out))
    return out


def op_density(code, ctx, amount):
    y, _, c, cb, cr = opponent(code)
    return from_opponent(y - amount * c, cb, cr)


def _rotate(cb, cr, degrees):
    t = np.radians(degrees)  # a scalar: NumPy scalars work with GPU arrays too
    return cb * np.cos(t) - cr * np.sin(t), cb * np.sin(t) + cr * np.cos(t)


def op_hue_rotate(code, ctx, degrees):
    y, _, _, cb, cr = opponent(code)
    return from_opponent(y, *_rotate(cb, cr, degrees))


def op_vibrance(code, ctx, amount):
    y, _, c, cb, cr = opponent(code)
    k = 1 + amount * (1 - smoothstep(0.0, 0.12, c))
    return from_opponent(y, cb * k, cr * k)


def op_split_tone(code, ctx, shadow_hue, shadow_amount, highlight_hue, highlight_amount, balance):
    y, _, _, cb, cr = opponent(code)
    pivot = GREY + balance
    sh = (1 - smoothstep(0.15, pivot, y)) * shadow_amount
    hi = smoothstep(pivot, 0.75, y) * highlight_amount
    hs, hh = math.radians(shadow_hue), math.radians(highlight_hue)
    return from_opponent(y, cb + sh * math.cos(hs) + hi * math.cos(hh), cr + sh * math.sin(hs) + hi * math.sin(hh))


def gamut_compress_linear(lin, threshold, limit, power):
    """Reference gamut compression on linear RGB (3, ...): distances from the achromatic axis beyond threshold are
    compressed so that limit maps to 1 (the gamut boundary)."""
    nx = xp(lin)
    ach = nx.max(lin, axis=0)
    a = nx.abs(ach)
    d = nx.where(a > 0, (ach - lin) / nx.where(a > 0, a, 1), 0.0)
    out = nx.empty_like(lin)
    for i in range(3):
        t, lim = threshold[i], limit[i]
        s = (lim - t) / (((1 - t) / (lim - t)) ** -power - 1) ** (1 / power)
        x = (d[i] - t) / s
        cd = nx.where(d[i] < t, d[i], t + s * x / (1 + nx.maximum(x, 0) ** power) ** (1 / power))
        out[i] = ach - cd * a
    return out


def op_gamut_compress(code, ctx, threshold, limit, power):
    return C.acescct_encode(gamut_compress_linear(C.acescct_decode(code), threshold, limit, power))


def _rotated(dx, dy, h, w, angle):
    """Offsets from a centre (normalised x, y; y down) in a frame rotated counter-clockwise by angle (deg), in units
    of the frame width: (u along the angle, v across it, k = height / width in those units)."""
    k = (h - 1) / (w - 1) if w > 1 and h > 1 else 1.0  # a one-pixel row or column has no aspect: 1
    t = math.radians(angle)
    up = -dy * k  # y up, in frame widths
    return dx * math.cos(t) + up * math.sin(t), -dx * math.sin(t) + up * math.cos(t), k


def _grid(h, w, like):
    """Normalised pixel coordinates, 0..1 over pixel centres (x = i / (w - 1)), origin top-left, on like's device."""
    nx = xp(like)
    xx = nx.linspace(0, 1, w)[None, :] * nx.ones((h, 1))
    yy = nx.linspace(0, 1, h)[:, None] * nx.ones((1, w))
    return xx, yy


def op_vignette(code, ctx, amount, roundness, softness):
    _, h, w = code.shape
    xx, yy = _grid(h, w, code)
    xx, yy = xx * 2 - 1, yy * 2 - 1
    r = xp(code).sqrt((xx * (w / h) ** (1 - roundness)) ** 2 + yy ** 2) / math.sqrt(2)
    return code - amount * smoothstep(1 - softness, 1.2, r)[None] * STOP * 4


def hash_u32(*keys):
    """Integer hash (lowbias32 rounds) of uint32 arrays/ints, combined in order; deterministic on every target."""
    nx = xp(*keys)
    h = np.uint32(0x9E3779B9)
    with np.errstate(over="ignore"):
        for k in keys:
            h = nx.asarray(h ^ nx.asarray(k, dtype=np.uint32), dtype=np.uint32)
            h = h ^ (h >> np.uint32(16))
            h = (h * np.uint32(0x7FEB352D)).astype(np.uint32)
            h = h ^ (h >> np.uint32(15))
            h = (h * np.uint32(0x846CA68B)).astype(np.uint32)
            h = h ^ (h >> np.uint32(16))
    return h


def _cell_noise(ix, iy, frame, seed, ch):
    """Approximately N(0, 1) per lattice cell: sum of 4 uniforms from the hash (Irwin-Hall, centred, scaled)."""
    h = hash_u32(ix, iy, frame, seed, ch)
    s = 0.0
    for j in range(4):
        h = hash_u32(h, j)
        s = s + (h >> np.uint32(8)).astype(np.float64) / 16777216.0
    return (s - 2.0) * math.sqrt(3.0)


def op_grain(code, ctx, amount, size_px, chroma, seed):
    nx = xp(code)
    _, h, w = code.shape
    size = max(size_px * ctx["px_scale"], 1e-3)
    px = nx.arange(w, dtype=np.float64)[None, :] * nx.ones((h, 1))
    py = nx.arange(h, dtype=np.float64)[:, None] * nx.ones((1, w))
    gx, gy = px / size, py / size
    x0, y0 = nx.floor(gx), nx.floor(gy)
    fx, fy = gx - x0, gy - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    x0i, y0i = (x0.astype(np.int64) & 0xFFFFFFFF), (y0.astype(np.int64) & 0xFFFFFFFF)
    frame = ctx["frame"]

    def noise(ch):
        n00 = _cell_noise(x0i, y0i, frame, seed, ch)
        n10 = _cell_noise(x0i + 1, y0i, frame, seed, ch)
        n01 = _cell_noise(x0i, y0i + 1, frame, seed, ch)
        n11 = _cell_noise(x0i + 1, y0i + 1, frame, seed, ch)
        return (n00 * (1 - fx) + n10 * fx) * (1 - fy) + (n01 * (1 - fx) + n11 * fx) * fy

    mono = noise(3)
    n = nx.stack([mono + chroma * (noise(c) - mono) for c in range(3)])
    y = nx.clip(luma(code), 0, 1)
    resp = 4 * y * (1 - y)
    return code + amount * 0.02 * n * resp[None]


def op_glow(code, ctx, threshold, radius_px, strength, warm):
    nx = xp(code)
    lin = C.from_working(code)
    bright = nx.maximum(lin - float(C.acescct_decode(np.array(threshold))), 0)
    halo = gaussian_blur(bright, radius_px * ctx["px_scale"]) * strength
    halo = halo * _rgb([1 + warm, 1.0, 1 - warm], code)
    return C.acescct_encode(nx.maximum(C.acescct_decode(code) + C.apply_matrix(C.M709_AP1, halo), 0))


def op_halation(code, ctx, threshold, radius_px, strength, tint):
    nx = xp(code)
    lin = C.from_working(code)
    bright = nx.maximum(lin - float(C.acescct_decode(np.array(threshold))), 0)
    yb = (bright * const(C.REC709_Y, code).reshape(3, 1, 1)).sum(0, keepdims=True)
    halo = gaussian_blur(yb, radius_px * ctx["px_scale"]) * strength * _rgb(tint, code)
    return C.acescct_encode(nx.maximum(C.acescct_decode(code) + C.apply_matrix(C.M709_AP1, halo), 0))


def op_diffusion(code, ctx, radius_px, strength):
    nx = xp(code)
    lin = nx.maximum(C.acescct_decode(code), 0)
    spread = nx.maximum(gaussian_blur(lin, radius_px * ctx["px_scale"]) - lin, 0)
    return C.acescct_encode(lin + strength * spread)


def op_blur(code, ctx, radius_px):
    return gaussian_blur(code, radius_px * ctx["px_scale"])


def op_sharpen(code, ctx, amount, radius_px, coring):
    y = luma(code)
    detail = y - gaussian_blur(y[None], radius_px * ctx["px_scale"])[0]
    nx = xp(code)
    cored = nx.sign(detail) * nx.maximum(nx.abs(detail) - coring, 0)
    return code + (amount * cored)[None]


def op_local_contrast(code, ctx, amount, radius_px, protect_blacks):
    y = luma(code)
    delta = amount * (y - gaussian_blur(y[None], radius_px * ctx["px_scale"])[0])
    if protect_blacks:
        delta = xp(code).where(delta < 0, delta * smoothstep(0.05, 0.25, y), delta)
    return code + delta[None]


def op_guided_detail(code, ctx, amount, radius_px, edge, protect_blacks):
    y = luma(code)
    delta = amount * (y - guided_filter(y, y, radius_px * ctx["px_scale"], edge * edge))
    if protect_blacks:
        delta = xp(code).where(delta < 0, delta * smoothstep(0.05, 0.25, y), delta)
    return code + delta[None]


def op_highlights_shadows(code, ctx, highlights, shadows, radius_px):
    base = gaussian_blur(luma(code)[None], radius_px * ctx["px_scale"])[0]
    hi = smoothstep(GREY + 0.05, 0.75, base)
    lo = 1 - smoothstep(0.15, GREY, base)
    return code + (highlights * 0.1 * hi + shadows * 0.1 * lo)[None]


IMPL = {name[3:]: fn for name, fn in globals().items() if name.startswith("op_")}
assert set(IMPL) == set(OPS), set(OPS) ^ set(IMPL)


# ---------------------------------------------------------------- masks

def hue_span(r):
    """Width of a hue range in degrees: hi - lo, or hi - lo + 360 when it wraps (330..30 = 60); 0..360 = 360.
    None when there is no hue range."""
    if r is None:
        return None
    return r.hi - r.lo if r.hi > r.lo else r.hi - r.lo + 360


def mask_value(code, m, ctx):
    if isinstance(m, A.MaskNot):
        return 1 - mask_value(code, m.m, ctx)
    if isinstance(m, A.MaskBin):
        a, b = mask_value(code, m.a, ctx), mask_value(code, m.b, ctx)
        if m.op == "and":
            return a * b
        if m.op == "+":
            return xp(code).minimum(a + b, 1.0)
        if m.op == "-":
            return xp(code).maximum(a - b, 0.0)
        return xp(code).maximum(a, b)
    if isinstance(m, A.MaskRef):  # a node's key carried from where it ran, or a key statement made here; gain, offset
        if m.carried:
            if m.name not in ctx.get("keys", {}):
                raise ValueError(f"key({m.name}) is read, but node {m.name} has not run in this graph (a split part?)")
            base = ctx["keys"][m.name]
        else:
            base = mask_value(code, m.target, ctx)
        k = resolve_args({"gain": m.gain, "offset": m.offset}, ctx["frame"])
        return xp(code).clip(k["gain"] * base + k["offset"], 0.0, 1.0)
    a = resolve_args(m.call.args, ctx["frame"])
    kind = m.call.name
    nx = xp(code)
    _, h, w = code.shape
    if kind in ("qualifier", "skin"):
        if kind == "skin":
            half = a["width"] / 2
            a = {"hue": A.Range(141 - half, 141 + half), "sat": A.Range(0.006, 0.09), "lum": A.Range(0.15, 0.7),
                 "soft": a["soft"], "blur_px": a["blur_px"], "feather_px": a["feather_px"],
                 "clip_black": a["clip_black"], "clip_white": a["clip_white"]}
        y, hue, c, _, _ = opponent(code)
        hue = nx.where(c < NEUTRAL_C, 0.0, hue)  # hue of a grey is 0 deg (atan2(0, 0)), not rounding noise
        soft = a["soft"]
        mv = nx.ones((h, w))
        span = hue_span(a.get("hue"))
        if span is not None and span < 360:  # 0..360 (span 360) selects every hue: no hue term, no chroma gate
            lo = a["hue"].lo
            centre, half = (lo + span / 2) % 360, span / 2
            dist = nx.abs(((hue - centre + 180) % 360) - 180)
            mv = mv * (1 - smoothstep(half, half + soft * 360, dist))
        if a.get("sat") is not None:
            mv = mv * band(c, a["sat"].lo, a["sat"].hi, soft * 0.2)
        if a.get("lum") is not None:
            mv = mv * band(y, a["lum"].lo, a["lum"].hi, soft)
    elif kind == "luma":
        mv = band(luma(code), a["range"].lo, a["range"].hi, a["soft"])
    elif kind == "window":
        xx, yy = _grid(h, w, code)
        (cx, cy), (rx, ry) = a["centre"], a["radius"]
        if a["angle"]:
            u, v, k = _rotated(xx - cx, yy - cy, h, w, a["angle"])
            d = nx.sqrt((u / rx) ** 2 + (v / (k * ry)) ** 2)
        else:
            d = nx.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        mv = 1 - smoothstep(1 - a["soft"], 1 + a["soft"], d)
    elif kind == "box":
        xx, yy = _grid(h, w, code)
        (cx, cy), (sx, sy) = a["centre"], a["size"]
        u, v, k = _rotated(xx - cx, yy - cy, h, w, a["angle"])
        n = 2 + 18 * (1 - a["roundness"])
        d = (nx.abs(u / sx) ** n + nx.abs(v / (k * sy)) ** n) ** (1 / n)
        mv = 1 - smoothstep(1 - a["soft"], 1 + a["soft"], d)
    elif kind == "polygon":
        xx, yy = _grid(h, w, code)
        k = (h - 1) / (w - 1) if w > 1 and h > 1 else 1.0
        px, py = xx, yy * k
        pts = [(float(p[0]), float(p[1]) * k) for p in a["points"]]
        inside = nx.zeros((h, w), dtype=bool)
        d2 = nx.full((h, w), nx.inf)
        for i in range(len(pts)):
            (xi, yi), (xj, yj) = pts[i], pts[i - 1]
            cross = ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / ((yj - yi) or 1.0) + xi)
            inside ^= cross
            ex, ey = xj - xi, yj - yi
            t = nx.clip(((px - xi) * ex + (py - yi) * ey) / max(ex * ex + ey * ey, 1e-12), 0, 1)
            d2 = nx.minimum(d2, (px - xi - t * ex) ** 2 + (py - yi - t * ey) ** 2)
        sd = nx.sqrt(d2) * nx.where(inside, 1.0, -1.0)
        mv = smoothstep(-a["soft"], a["soft"], sd)
    elif kind == "graduated":
        xx, yy = _grid(h, w, code)
        (cx, cy), (fb, fa) = a["centre"], a["falloff"]
        s, _, _ = _rotated(xx - cx, yy - cy, h, w, a["angle"])
        mv = smoothstep(-fb, fa, s)
    elif kind == "depth":
        depth = ctx["aux"]["depth"]
        mv = band(nx.log(nx.maximum(depth, 1e-3)), math.log(a["range"].lo), math.log(a["range"].hi), a["soft"])
    elif kind == "segment":
        mv = nx.isin(ctx["aux"]["segments"], const(a["ids"], code, dtype=np.int64)).astype(np.float64)
    elif kind == "rgb_key":
        mv = nx.ones((h, w))
        for i, key in enumerate(("red", "green", "blue")):
            if a.get(key) is not None:
                mv = mv * band(code[i], a[key].lo, a[key].hi, a["soft"])
    else:
        raise ValueError(kind)
    if a.get("blur_px"):
        mv = gaussian_blur(mv[None], a["blur_px"] * ctx["px_scale"])[0]
    if a.get("feather_px"):  # edge-aware: the matte's soft edge snaps to the picture's luma edges
        mv = nx.clip(guided_filter(nx.clip(luma(code), 0, 1), mv, a["feather_px"] * ctx["px_scale"], 1e-3), 0, 1)
    lo, hi = a.get("clip_black", 0.0), a.get("clip_white", 1.0)
    if lo or hi != 1.0:  # matte clip: black..white stretched to 0..1
        mv = nx.clip((mv - lo) / max(hi - lo, 1e-6), 0, 1)
    return mv


# ---------------------------------------------------------------- keyframes

def key_value(k, frame):
    """Value of a Keys at an integer frame: held before the first and after the last key."""
    fr = [f for f, _ in k.keys]
    vals = [v for _, v in k.keys]
    if frame <= fr[0] or len(fr) == 1:
        return vals[0]
    if frame >= fr[-1]:
        return vals[-1]
    if k.interp == "step":
        return vals[max(i for i, f in enumerate(fr) if f <= frame)]
    if k.interp == "linear":
        i = max(i for i, f in enumerate(fr) if f <= frame)
        t = (frame - fr[i]) / (fr[i + 1] - fr[i])
        return vals[i] + (vals[i + 1] - vals[i]) * t
    return float(monotone_curve(np.array([float(frame)]), list(zip(fr, vals)))[0])


def key_poly(k, a, b):
    """The polynomial (numpy.poly1d in the frame) that key_value(k, f) follows for integer frames a <= f < b, where
    a and b lie within one segment of k's keys (or outside its key range). Used to bound animated values exactly."""
    fr = [f for f, _ in k.keys]
    vals = [float(v) for _, v in k.keys]
    if len(fr) == 1 or b <= fr[0]:
        return np.poly1d([vals[0]])
    if a >= fr[-1]:
        return np.poly1d([vals[-1]])
    i = max(j for j, f in enumerate(fr) if f <= a)
    if not b <= fr[i + 1]:
        raise ValueError(f"frames {a}..{b} cross a key of this value")
    x0, x1, y0, y1 = fr[i], fr[i + 1], vals[i], vals[i + 1]
    if k.interp == "step":
        return np.poly1d([y0])
    t = np.poly1d([1.0 / (x1 - x0), -x0 / (x1 - x0)])  # t(f) = (f - x0) / h
    if k.interp == "linear":
        return y0 + (y1 - y0) * t
    _, _, ms = curve_tangents(list(zip(fr, vals)))
    h, m0, m1 = x1 - x0, ms[i], ms[i + 1]
    t2, t3 = t * t, t * t * t
    return (2 * t3 - 3 * t2 + 1) * y0 + (t3 - 2 * t2 + t) * (h * m0) + (-2 * t3 + 3 * t2) * y1 + (t3 - t2) * (h * m1)


def resolve(v, frame):
    if isinstance(v, A.Keys):
        return key_value(v, frame)
    if isinstance(v, tuple) and v and not isinstance(v[0], tuple):
        return tuple(resolve(x, frame) for x in v)
    return v


def resolve_args(args, frame):
    return {k: resolve(v, frame) for k, v in args.items()}


# ---------------------------------------------------------------- graph

def run_node(code, node, ctx):
    mv = None
    if node.mask is not None and node.id in ctx.get("carry", ()):  # its key is read downstream: made here, off or not
        mv = ctx.setdefault("keys", {})[node.id] = mask_value(code, node.mask, ctx)
    if not node.enabled:
        return code
    args = resolve_args(node.call.args, ctx["frame"])
    out = IMPL[node.call.name](code, ctx, **args)
    weight = resolve(node.mix, ctx["frame"])
    if node.mask is not None:
        weight = weight * (mv if mv is not None else mask_value(code, node.mask, ctx))[None]
    return code + (out - code) * weight


def _blend(bottom, top, mode):
    if mode == "normal":
        return top
    nx = xp(bottom, top)
    if mode == "lighten":
        return nx.maximum(bottom, top)
    if mode == "darken":
        return nx.minimum(bottom, top)
    if mode == "screen":
        return 1 - (1 - bottom) * (1 - top)
    if mode == "overlay":
        return nx.where(bottom < 0.5, 2 * bottom * top, 1 - 2 * (1 - bottom) * (1 - top))
    lb, lt = C.acescct_decode(bottom), C.acescct_decode(top)
    if mode == "subtract":
        return C.acescct_encode(nx.maximum(lb - lt, 0))
    if mode == "difference":
        return C.acescct_encode(nx.abs(lb - lt))
    if mode == "add":
        return C.acescct_encode(nx.maximum(lb + lt, 0))
    if mode == "multiply":
        return C.acescct_encode(nx.maximum(lb * lt / 0.18, 0))
    raise ValueError(mode)


def run_graph(code, g, prog, ctx):
    if g is None or isinstance(g, A.Pass):  # an omitted graph and `pass` are the identity
        return code
    if isinstance(g, A.Ref):
        return run_node(code, prog.nodes[g.name], ctx)
    if isinstance(g, A.Serial):
        for item in g.items:
            code = run_graph(code, item, prog, ctx)
        return code
    if isinstance(g, A.Parallel):
        outs = [run_graph(code, b, prog, ctx) for b in g.branches]
        return sum(outs) / len(outs)
    if isinstance(g, A.Layer):
        b = run_graph(code, g.bottom, prog, ctx)
        t = run_graph(code, g.top, prog, ctx)
        return b + resolve(g.opacity, ctx["frame"]) * (_blend(b, t, g.mode) - b)
    raise AssertionError(g)


def make_ctx(prog, width, frame=0, aux=None, px_scale=None):
    from .walk import carried_keys
    return {"frame": int(frame), "aux": aux or {}, "keys": {},
            "carry": {k for n in prog.nodes.values() for k in carried_keys(n.mask)},
            "px_scale": float(px_scale) if px_scale is not None else width / prog.reference_width}


def grade_working(code, prog, frame=0, aux=None, px_scale=None, display_referred=False, device=None):
    """ACEScct in -> graded ACEScct out (photo_graph first when display_referred). device "gpu" (or
    CIERRA_DEVICE=gpu) runs the graph on the GPU; the result comes back as the kind of array that went in."""
    ctx = make_ctx(prog, code.shape[-1], frame, aux, px_scale)
    for r in prog.requires:
        if r not in ctx["aux"]:
            raise ValueError(f"the program needs the {r!r} plane")
    dev, back = pick_device(device), "gpu" if xp(code) is not np else "cpu"
    code = to_device(code, dev)
    ctx["aux"] = {k: to_device(v, dev) if k == "depth" else (to_device(v, dev).astype(np.int64) if dev == "gpu"
                                                            else v) for k, v in ctx["aux"].items()}
    if display_referred and prog.photo_graph is not None:
        code = run_graph(code, prog.photo_graph, prog, ctx)
    return to_device(run_graph(code, prog.graph, prog, ctx), back)


def render(prog, image, frame=0, aux=None, px_scale=None, input=None, device=None):
    """Image (3, H, W) in the program's input space (or `input`) -> output (display code values, or ACEScct for an
    `output acescct()` program), as a NumPy array. device: "cpu" (default) or "gpu" (see grade_working)."""
    dev = pick_device(device)
    image = to_device(image, dev) if dev == "gpu" else np.asarray(image, dtype=np.float64)
    code, display_referred = C.input_to_working(image, input or prog.input)
    graded = grade_working(code, prog, frame, aux, px_scale, display_referred, device=dev)
    return to_host(C.output_from_working(graded, prog.output))


def is_lut_expressible(prog, graph=None):
    """(bool, reasons): can the graph live in a 3D LUT? (rules: cierra.targets.lut_reasons)"""
    from .targets import lut_reasons
    reasons = lut_reasons(prog, graph)
    return not reasons, reasons
