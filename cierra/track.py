"""Tracking: a window, box or graduated mask that follows the picture, written as keys(linear, ...) on its centre
(and its size and angle for a similarity track), so the program stays one readable text that every target already
runs. Polygon points cannot be keyed in v0, so a polygon is not tracked.

The tracker follows one patch of the reference frame (the mask's own region there) through luma frames: a
frame-to-frame normalised cross-correlation finds the coarse shift (the last frame's patch already has the current
size and turn), then Gauss-Newton matches the reference patch itself under a similarity warp, with a gain and an
offset for the light. Matching the reference, not the last frame, is what keeps a long track from drifting; when the
best match falls below `min_match` the track is lost and says where, rather than sliding off onto something else.
"""

import math

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

from . import ast as A
from .engine import resolve_args
from .ops import LIMITS, MASKS

SHAPES = {"window": "radius", "box": "size", "graduated": None}  # trackable mask -> its extent pair (scaled)
INSIDE = {"window": 0.7, "box": 0.9}  # patch half-size / mask extent: the patch stays inside the shape
MOTIONS = ("translate", "similarity")
MAX_BUFFER = 2 ** 31  # bytes of luma held for the frames before the reference (tracked backwards after it)


class TrackError(ValueError):
    """A track that cannot be made as asked, or that was lost; the CLI prints it as one line."""


class Lost(TrackError):
    def __init__(self, why, score=None):
        self.why, self.score, self.frame = why, score, None
        super().__init__(why)


def _zncc_map(region, t):
    """Zero-mean normalised cross-correlation of template t at every position where it fits inside region."""
    t = t - t.mean()
    h, w = t.shape
    num = fftconvolve(region, t[::-1, ::-1], mode="valid")
    c1 = np.pad(region.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    c2 = np.pad((region * region).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s1 = c1[h:, w:] - c1[:-h, w:] - c1[h:, :-w] + c1[:-h, :-w]
    s2 = c2[h:, w:] - c2[:-h, w:] - c2[h:, :-w] + c2[:-h, :-w]
    den = np.sqrt(np.maximum(s2 - s1 * s1 / t.size, 0) * (t * t).sum())
    return np.where(den > 1e-9, num / np.maximum(den, 1e-30), 0.0)


def _peak(m):
    """Arg-max of a correlation map, refined to a fraction of a pixel by a parabola through each axis."""
    iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
    off = []
    for i, n, line in ((iy, m.shape[0], m[:, ix]), (ix, m.shape[1], m[iy, :])):
        d = 0.0
        if 0 < i < n - 1:
            a, b, c = line[i - 1], line[i], line[i + 1]
            den = a - 2 * b + c
            d = 0.5 * (a - c) / den if den < 0 else 0.0
        off.append(i + float(np.clip(d, -0.5, 0.5)))
    return off[0], off[1], float(m[iy, ix])


def _zncc(a, b):
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


class _Crop:
    """A smoothed crop of a frame around a point: cubic spline coefficients for exact-ish samples, and central
    differences for the Gauss-Newton gradient (only the step uses them; the fixed point is set by the samples)."""

    def __init__(self, img, cx, cy, r):
        h, w = img.shape
        self.x0, self.y0 = max(0, int(cx - r)), max(0, int(cy - r))
        x1, y1 = min(w, int(cx + r) + 2), min(h, int(cy + r) + 2)
        if x1 - self.x0 < 4 or y1 - self.y0 < 4:
            raise Lost("the region leaves the picture")
        c = ndimage.gaussian_filter(np.asarray(img[self.y0:y1, self.x0:x1], np.float64), 1.0, mode="nearest")
        self.gy, self.gx = np.gradient(c)
        self.coef = ndimage.spline_filter(c, 3)

    def sample(self, x, y, grad=False):
        at = [y - self.y0, x - self.x0]
        v = ndimage.map_coordinates(self.coef, at, order=3, prefilter=False, mode="nearest")
        if not grad:
            return v
        return (v, ndimage.map_coordinates(self.gx, at, order=1, mode="nearest"),
                ndimage.map_coordinates(self.gy, at, order=1, mode="nearest"))


class Tracker:
    """Follows the patch of `ref` centred on `centre` (pixels, x right, y down), `half` = (hx, hy) pixels, frame by
    frame (step). Pose: centre and the similarity (a, b) = scale·(cos φ, sin φ), φ clockwise on screen."""

    def __init__(self, ref, centre, half, motion="similarity", search=None, min_match=0.6):
        self.hx, self.hy = (max(2, int(round(v))) for v in half)
        qy, qx = np.mgrid[-self.hy:self.hy + 1, -self.hx:self.hx + 1].astype(np.float64)
        self.qx, self.qy = qx.ravel(), qy.ravel()
        self.sim = motion == "similarity"
        self.search = int(search or max(16, self.hx, self.hy))
        self.min_match = min_match
        self.shape = ref.shape
        self.p = np.array([centre[0], centre[1], 1.0, 0.0])
        self.v = np.zeros(2)
        self.T = _Crop(ref, centre[0], centre[1], math.hypot(self.hx, self.hy) + 6).sample(*self._warp(self.p))
        if self.T.std() < 1e-4:
            raise TrackError("the region is flat: there is nothing in it to track")
        self.prev = ref

    def _warp(self, p):
        cx, cy, a, b = p
        return cx + a * self.qx - b * self.qy, cy + b * self.qx + a * self.qy

    def pose(self, score=1.0):
        """(x, y, scale, angle) with the angle in degrees counter-clockwise on screen, and the match."""
        cx, cy, a, b = self.p
        return float(cx), float(cy), math.hypot(a, b), -math.degrees(math.atan2(b, a)), score

    def _coarse(self, img, pred):
        """The previous frame's patch (current size and turn) searched for around the predicted centre."""
        h, w = img.shape
        s = math.hypot(*self.p[2:])
        hx, hy = max(2, int(round(self.hx * s))), max(2, int(round(self.hy * s)))
        px, py = int(round(self.p[0])), int(round(self.p[1]))
        if px - hx < 0 or py - hy < 0 or px + hx >= w or py + hy >= h:
            return pred
        t = np.asarray(self.prev[py - hy:py + hy + 1, px - hx:px + hx + 1], np.float64)
        qx, qy, r = int(round(pred[0])), int(round(pred[1])), self.search
        x0, y0 = max(0, qx - hx - r), max(0, qy - hy - r)
        x1, y1 = min(w, qx + hx + r + 1), min(h, qy + hy + r + 1)
        if x1 - x0 < t.shape[1] or y1 - y0 < t.shape[0]:
            return pred
        iy, ix, score = _peak(_zncc_map(np.asarray(img[y0:y1, x0:x1], np.float64), t))
        if score < 0.3:  # nothing like it near the prediction: let Gauss-Newton start from the prediction
            return pred
        return np.array([x0 + ix + hx + (self.p[0] - px), y0 + iy + hy + (self.p[1] - py)])

    def step(self, img):
        """Pose in the next frame, or Lost."""
        h, w = img.shape
        pred = self.p[:2] + self.v
        p = np.concatenate([self._coarse(img, pred), self.p[2:]])
        crop = _Crop(img, p[0], p[1], math.hypot(self.hx, self.hy) * math.hypot(*p[2:]) + 10)
        g = o = None
        for _ in range(60):
            x, y = self._warp(p)
            if np.mean((x < 0) | (x > w - 1) | (y < 0) | (y > h - 1)) > 0.2:
                raise Lost("the region leaves the picture")
            i, gx, gy = crop.sample(x, y, grad=True)
            if g is None:  # the light: gain and offset of the patch, then solved with the pose
                g, o = np.linalg.lstsq(np.stack([self.T, np.ones_like(self.T)], 1), i, rcond=None)[0]
            r = i - g * self.T - o
            cols = [gx, gy] + ([gx * self.qx + gy * self.qy, gy * self.qx - gx * self.qy] if self.sim else [])
            J = np.stack(cols + [-self.T, -np.ones_like(self.T)], 1)
            d = np.linalg.lstsq(J, -r, rcond=None)[0]
            n = len(cols)
            p[:n] += d[:n]
            g, o = g + d[n], o + d[n + 1]
            if not 0.2 < math.hypot(*p[2:]) < 5:
                raise Lost("the region's size runs away")
            if max(abs(d[0]), abs(d[1]), *(abs(d[2:n]) * max(self.hx, self.hy))) < 1e-4:
                break
            if math.hypot(*(p[:2] - pred)) > 4 * (self.search + max(self.hx, self.hy)):
                raise Lost("the match slides away")
        score = _zncc(crop.sample(*self._warp(p)), self.T)
        if score < self.min_match:
            raise Lost(f"match {score:.2f} < {self.min_match:g}", score)
        self.v, self.p, self.prev = p[:2] - self.p[:2], p, img
        return self.pose(score)


def track(frames, at, centre, half, motion="similarity", search=None, min_match=0.6):
    """{frame: (x, y, scale, angle, match)} over frames ((n, luma) in increasing n): forwards from frame `at` (the
    reference, where the patch is taken), and backwards from it over the frames before it, both against the
    reference patch. Raises Lost with .frame set."""
    back, out, fwd = [], {}, None
    for n, img in frames:
        if n < at:
            back.append((n, img))
        elif fwd is None:
            if n != at:
                raise TrackError(f"frame {at} is not among the frames read (with --step, start the range on it)")
            ref, fwd = img, Tracker(img, centre, half, motion, search, min_match)
            out[n] = fwd.pose()
        else:
            out[n] = _step(fwd, n, img)
    if fwd is None:
        raise TrackError(f"frame {at} is not among the frames read")
    bwd = Tracker(ref, centre, half, motion, search, min_match)
    for n, img in reversed(back):
        out[n] = _step(bwd, n, img)
    return out


def _step(t, n, img):
    try:
        return t.step(img)
    except Lost as e:
        e.frame = n
        raise


def thin(frames, vals, tol):
    """Keys through (frames, vals) such that linear interpolation between the kept keys misses no value by more than
    tol: the first and last frame, then the worst-missed frame, again and again (Douglas-Peucker on one value)."""
    f, v = np.asarray(frames, np.float64), np.asarray(vals, np.float64)
    keep, todo = {0, len(v) - 1}, [(0, len(v) - 1)]
    while todo:
        i, j = todo.pop()
        if j - i < 2:
            continue
        line = v[i] + (v[j] - v[i]) * (f[i + 1:j] - f[i]) / (f[j] - f[i])
        err = np.abs(v[i + 1:j] - line)
        k = int(np.argmax(err))
        if err[k] > tol:
            keep.add(i + 1 + k)
            todo += [(i, i + 1 + k), (i + 1 + k, j)]
    return [(int(frames[i]), float(v[i])) for i in sorted(keep)]


def _value(name, frames, vals, tol, ref, original, digits):
    """The parameter value for a tracked series: the original when the track never moves it by more than tol from
    its value at the reference frame, else thinned linear keys (at most LIMITS["keys"])."""
    vals = np.asarray(vals, np.float64)
    if np.all(np.abs(vals - ref) <= tol):
        return original, 0
    keys = thin(frames, vals, tol)
    if len(keys) > LIMITS["keys"]:
        raise TrackError(f"{name} needs {len(keys)} keyframes to follow the track within the tolerance, more than "
                         f"the {LIMITS['keys']} a value may have: track a shorter range (or raise --tolerance)")
    return A.Keys("linear", tuple((f, round(v, digits)) for f, v in keys)), len(keys)


def _mask_calls(m):
    """The mask calls of a node's own mask expression, in reading order (key(...) references are left out: a key
    statement may serve other nodes, and a carried key is another node's)."""
    if isinstance(m, A.MaskCall):
        return [m]
    if isinstance(m, A.MaskNot):
        return _mask_calls(m.m)
    if isinstance(m, A.MaskBin):
        return _mask_calls(m.a) + _mask_calls(m.b)
    return []


def trackable(prog, node, shape=None):
    """The mask call of `node` a track moves: its only window / box / graduated, or the shape-th (1 = first in
    reading order)."""
    n = prog.nodes.get(node)
    if n is None:
        raise TrackError(f"no node {node!r} (nodes: {', '.join(prog.nodes) or 'none'})")
    calls = [c for c in _mask_calls(n.mask) if c.call.name in SHAPES]
    if not calls:
        poly = any(c.call.name == "polygon" for c in _mask_calls(n.mask))
        raise TrackError(f"node {node} has no window, box or graduated mask to track"
                         + (" (polygon points cannot be keyed)" if poly else ""))
    if shape is None:
        if len(calls) > 1:
            raise TrackError(f"node {node}'s mask has {len(calls)} trackable shapes: pick one with --shape 1..{len(calls)}")
        return calls[0].call
    if not 1 <= shape <= len(calls):
        raise TrackError(f"--shape {shape}: node {node}'s mask has {len(calls)} trackable shape(s)")
    return calls[shape - 1].call


def track_program(prog, clip, node, first, last, at=None, step=1, shape=None, motion="similarity", patch=None,
                  search=None, min_match=0.6, tolerance=0.5):
    """Track node's mask over clip frames first..last (every step-th) from its place at frame `at` (default first)
    and write the motion into the program as keys: centre always, size and angle for a similarity track. Program
    frames are the clip's frame numbers. Returns (program, report); raises TrackError (incl. Lost) or VideoError."""
    from . import video
    if motion not in MOTIONS:
        raise TrackError(f"motion {motion!r}: expected {' or '.join(MOTIONS)}")
    at = first if at is None else at
    if not isinstance(step, int) or step < 1:
        raise TrackError(f"--step {step!r}: a whole number of frames, 1 or more")
    if not first <= at <= last:
        raise TrackError(f"--at {at} is outside the range {first}..{last}")
    if (at - first) % step:
        raise TrackError(f"--at {at} is not on the --step {step} grid from {first}")
    if not 0 < min_match < 1 or not tolerance > 0:
        raise TrackError("--min-match is between 0 and 1, --tolerance above 0")
    call = trackable(prog, node, shape)
    info = video.probe(clip)
    w, h = info["width"], info["height"]
    if (at - first) // step * w * h * 4 > MAX_BUFFER:
        raise TrackError(f"tracking {at - first} frames backwards holds them in memory, more than "
                         f"{MAX_BUFFER >> 30} GiB: place the mask nearer the range start (--at) or use --step")
    a = resolve_args(call.args, at)
    cx, cy = a["centre"][0] * (w - 1), a["centre"][1] * (h - 1)
    ext = SHAPES[call.name]
    if patch:
        half = (patch, patch)
    elif ext:
        ex, ey = a[ext][0] * (w - 1) * INSIDE[call.name], a[ext][1] * (h - 1) * INSIDE[call.name]
        half = (ex, ey) if a["angle"] == 0 else (min(ex, ey),) * 2  # a turned shape: a square inside it
    else:
        half = (0.1 * min(w, h),) * 2  # a graduated filter has no extent: a patch around its centre
    cap = max(2, min(w, h) // 4)
    half = tuple(min(max(v, 2), cap) for v in half)
    try:
        poses = track(video.read_luma(clip, first, last, step), at, (cx, cy), half, motion, search, min_match)
    except Lost as e:
        if e.frame is None:
            raise TrackError(f"{clip}: node {node}'s mask at frame {at}: {e.why}") from None
        side = f"--to {e.frame - step}" if e.frame > at else f"--from {e.frame + step}"
        raise TrackError(f"{clip}: track of node {node} lost at frame {e.frame} ({e.why}); end the range before it "
                         f"({side}) or place the mask there and track from it") from None
    frames = sorted(poses)
    xs, ys, ss, angs, match = (np.array([poses[f][i] for f in frames]) for i in range(5))
    counts = {}

    def put(key, i, vals, tol, ref, digits, lo, hi):
        if np.any(vals < lo) or np.any(vals > hi):
            raise TrackError(f"{clip}: the tracked {call.name} {key} leaves {lo:g}..{hi:g} (the region leaves the "
                             "picture or turns past 180 degrees): track a shorter range")
        orig = call.args[key] if i is None else call.args[key][i]
        v, n = _value(f"{call.name} {key}" + ("" if i is None else f"[{i}]"), frames, vals, tol, ref, orig, digits)
        counts[key + ("" if i is None else f"[{i}]")] = n
        return v

    pc = MASKS[call.name].params["centre"]
    new = dict(call.args)
    new["centre"] = (put("centre", 0, xs / (w - 1), tolerance / (w - 1), a["centre"][0], 6, pc.lo, pc.hi),
                     put("centre", 1, ys / (h - 1), tolerance / (h - 1), a["centre"][1], 6, pc.lo, pc.hi))
    if motion == "similarity":
        reach = max(half) * math.sqrt(2)  # pixels from the centre to the patch corner: an angle's tolerance there
        if ext:
            pe = MASKS[call.name].params[ext]
            new[ext] = tuple(put(ext, i, a[ext][i] * ss, tolerance / (d - 1), a[ext][i], 6, pe.lo, pe.hi)
                             for i, d in ((0, w), (1, h)))
            reach = max(a[ext][0] * (w - 1), a[ext][1] * (h - 1))
        pa = MASKS[call.name].params["angle"]
        new["angle"] = put("angle", None, a["angle"] + angs, math.degrees(tolerance / max(reach, 1)), a["angle"], 4,
                           pa.lo, pa.hi)
    call.args.clear()
    call.args.update(new)
    err = np.hypot(np.diff(xs), np.diff(ys))
    return prog, {"node": node, "shape": call.name, "frames": [frames[0], frames[-1]], "at": at, "step": step,
                  "tracked": len(frames), "motion": motion, "patch_px": [round(2 * v + 1) for v in half],
                  "min_match": round(float(match.min()), 3), "max_shift_px": round(float(err.max()), 2) if len(err) else 0.0,
                  "keys": counts}
