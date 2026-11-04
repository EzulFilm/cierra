"""Tracking (cierra/track.py): the tracker core on synthetic numpy frames of a textured object under known motion,
the keyframe thinning, and `cierra track` end to end on tiny lossless clips made with the system FFmpeg."""

import math
import shutil
import subprocess

import numpy as np
import pytest
from scipy import ndimage

from cierra import ast as A
from cierra.__main__ import main
from cierra.checker import load_file
from cierra.engine import key_value
from cierra.track import Lost, Tracker, TrackError, _value, thin

W, H = 160, 120
FFMPEG = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                            reason="needs ffmpeg and ffprobe on PATH")


def _texture(seed, n, sigma=1.5):
    t = ndimage.gaussian_filter(np.random.default_rng(seed).random((n, n)), sigma)
    return (t - t.min()) / np.ptp(t)


OBJ, BG = _texture(1, 160), _texture(2, 400)[:H, :W] * 0.6 + 0.2


def _frame(x, y, s=1.0, angle=0.0, half=30, obj=OBJ):
    """The background with a textured square (half side `half` at scale 1) at (x, y), scaled by s and turned by
    `angle` degrees counter-clockwise on screen; the object texture is sampled cubic, so sub-pixel motion is real."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    phi = math.radians(angle)  # counter-clockwise on screen = clockwise in y-down pixel coordinates: undo it
    dx, dy = xx - x, yy - y
    u = (math.cos(phi) * dx - math.sin(phi) * dy) / s
    v = (math.sin(phi) * dx + math.cos(phi) * dy) / s
    c = (obj.shape[0] - 1) / 2
    o = ndimage.map_coordinates(obj, [v + c, u + c], order=3, mode="nearest")
    return np.where((abs(u) <= half) & (abs(v) <= half), o, BG).astype(np.float32)


def _path(t):
    return 60 + 2.3 * t, 58 + 6 * math.sin(t / 5), 1 + 0.01 * t, 0.8 * t  # x, y, scale, angle


def test_translation_is_followed_to_a_hundredth_of_a_pixel():
    """Sub-pixel steps, a gentle curve: position within 0.05 px of the truth on every frame."""
    tr = Tracker(_frame(60, 58), (60, 58), (18, 18), motion="translate")
    worst = 0.0
    for t in range(1, 16):
        x, y = 60 + 1.37 * t, 58 + 3 * math.sin(t / 3)
        px, py, s, a, m = tr.step(_frame(x, y))
        worst = max(worst, math.hypot(px - x, py - y))
        assert m > 0.95 and s == 1 and a == 0
    assert worst < 0.05, worst


def test_similarity_follows_scale_and_rotation():
    """Scale 1 -> 1.15 and 12 degrees counter-clockwise over 15 frames, while moving."""
    x0, y0, _, _ = _path(0)
    tr = Tracker(_frame(x0, y0), (x0, y0), (18, 18))
    for t in range(1, 16):
        x, y, s, a = _path(t)
        px, py, ps, pa, m = tr.step(_frame(x, y, s, a))
        assert math.hypot(px - x, py - y) < 0.1, (t, px - x, py - y)
        assert abs(ps - s) < 0.003 and abs(pa - a) < 0.2, (t, ps, s, pa, a)


def test_lost_when_the_object_goes():
    """The object replaced by other texture: the track is lost (a low match), not moved onto something else."""
    tr = Tracker(_frame(60, 58), (60, 58), (18, 18))
    tr.step(_frame(61, 58))
    with pytest.raises(Lost):
        tr.step(_frame(62, 58, obj=_texture(9, 160)))


def test_flat_region_is_refused():
    with pytest.raises(TrackError, match="flat"):
        Tracker(np.full((H, W), 0.5, np.float32), (60, 58), (18, 18))


def test_thin_keeps_corners_and_bounds_the_miss():
    f = np.arange(100)
    v = np.where(f < 40, f * 0.5, 20 - (f - 40) * 0.25)  # two straight lines: three keys
    assert thin(f, v, 0.01) == [(0, 0.0), (40, 20.0), (99, 5.25)]
    s = 10 * np.sin(f / 9)
    keys = thin(f, s, 0.5)
    k = A.Keys("linear", tuple(keys))
    assert max(abs(key_value(k, i) - s[i]) for i in f) <= 0.5
    assert len(keys) < 20


def test_more_than_64_keys_is_said_plainly():
    f = np.arange(300)
    zig = (f % 4 >= 2) * 3.0  # a step every 2 frames: needs ~150 keys
    with pytest.raises(TrackError, match="more than the 64 a value may have"):
        _value("window centre[0]", f, zig, 0.5, 0.0, 0.0, 6)
    assert _value("x", f, np.full(300, 0.2), 0.5, 0.2, "orig", 6) == ("orig", 0)  # still: the value is kept


# ---------------------------------------------------------------- end to end, through FFmpeg

def _clip(path, frames):
    """Frames (float 0..1, H x W) as lossless 16-bit grey FFV1."""
    raw = path.with_suffix(".raw")
    raw.write_bytes(b"".join((np.clip(f, 0, 1) * 65535 + 0.5).astype("<u2").tobytes() for f in frames))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{W}x{H}",
                    "-r", "25", "-i", str(raw), "-c:v", "ffv1", "-pix_fmt", "gray16le", str(path)], check=True)
    return path


def _program(path, x, y, r=20, shape="window"):
    ext = "radius" if shape == "window" else "size"
    path.write_text("cierra 0.1\n\nnode lift = exposure(stops: 0.5)\n"
                    f"  mask {shape}(centre: [{x / (W - 1)}, {y / (H - 1)}], {ext}: [{r / (W - 1)}, {r / (H - 1)}])\n"
                    "\ngraph lift\n", encoding="utf-8")
    return path


def _call(path):
    return load_file(path).nodes["lift"].mask.call


def _at(v, f):
    return key_value(v, f) if isinstance(v, A.Keys) else v


@pytest.mark.parametrize("tol", [0.5, 0.02])
@FFMPEG
def test_track_writes_keys_that_follow_the_motion(tmp_path, capsys, tol):
    """24 frames of curved motion with growth and a turn: on every frame the keyed centre misses the truth by no
    more than the tolerance (plus 0.03 px of tracking error) on each axis, radius and angle follow, the keys are
    thinned (fewer than frames at the default tolerance) and the program still checks."""
    clip = _clip(tmp_path / "c.mkv", [_frame(*_path(t)) for t in range(24)])
    x0, y0, _, _ = _path(0)
    prog = _program(tmp_path / "p.cierra", x0, y0)
    assert not main(["track", str(prog), str(clip), "--node", "lift", "--from", "0", "--to", "23",
                     "--tolerance", str(tol)])
    report = capsys.readouterr().out
    c = _call(prog)
    for t in range(24):
        x, y, s, a = _path(t)
        cx, cy = _at(c.args["centre"][0], t) * (W - 1), _at(c.args["centre"][1], t) * (H - 1)
        assert abs(cx - x) < tol + 0.03 and abs(cy - y) < tol + 0.03, (t, cx - x, cy - y)
        assert abs(_at(c.args["radius"][0], t) * (W - 1) - 20 * s) < tol + 0.03, t
        assert abs(_at(c.args["angle"], t) - a) < 0.1, t
    assert '"tracked": 24' in report
    if tol == 0.5:
        assert all(isinstance(v, A.Keys) and 2 <= len(v.keys) < 12 for v in c.args["centre"]), c.args["centre"]


@FFMPEG
def test_track_backwards_from_the_middle_with_a_step(tmp_path):
    """The mask placed on frame 12 of 0..21, every 3rd frame: tracked forwards and backwards, a box this time, its
    position only (--motion translate), written to another file."""
    clip = _clip(tmp_path / "c.mkv", [_frame(50 + 3 * t, 50 + 0.08 * t * t) for t in range(22)])
    prog = _program(tmp_path / "p.cierra", 50 + 36, 50 + 0.08 * 144, shape="box")
    before = prog.read_text()
    out = tmp_path / "o.cierra"
    assert not main(["track", str(prog), str(clip), "--node", "lift", "--from", "0", "--to", "21", "--at", "12",
                     "--step", "3", "--motion", "translate", "-o", str(out)])
    assert prog.read_text() == before
    c = _call(out)
    for t in range(0, 22, 3):
        cx, cy = _at(c.args["centre"][0], t) * (W - 1), _at(c.args["centre"][1], t) * (H - 1)
        assert math.hypot(cx - (50 + 3 * t), cy - (50 + 0.08 * t * t)) < 0.25, t
    assert [f for f, _ in c.args["centre"][0].keys] == [0, 21]  # straight in x: two keys
    assert all(f % 3 == 0 for f, _ in c.args["centre"][1].keys)
    assert c.args["size"] == (20 / (W - 1), 20 / (H - 1)) and c.args["angle"] == 0


@FFMPEG
def test_lost_track_is_one_line_and_writes_nothing(tmp_path, capsys):
    frames = [_frame(50 + 2 * t, 60) for t in range(10)] + [BG.astype(np.float32)] * 4
    clip = _clip(tmp_path / "c.mkv", frames)
    prog = _program(tmp_path / "p.cierra", 50, 60)
    before = prog.read_text()
    assert main(["track", str(prog), str(clip), "--node", "lift", "--from", "0", "--to", "13"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err
    assert "lost at frame 10" in err and "--to 9" in err
    assert prog.read_text() == before


@FFMPEG
def test_track_refusals_are_one_line(tmp_path, capsys):
    clip = _clip(tmp_path / "c.mkv", [_frame(60, 60)] * 3)
    keyed = tmp_path / "k.cierra"
    keyed.write_text("cierra 0.1\n\nnode a = exposure(stops: 0.5)\n  mask luma(range: 0.2..0.6)\n\ngraph a\n")
    prog = _program(tmp_path / "p.cierra", 60, 60)
    for p, args, msg in ((keyed, ["--node", "a", "--to", "2"], "node a has no window, box or graduated mask"),
                         (prog, ["--node", "b", "--to", "2"], "no node 'b'"),
                         (prog, ["--node", "lift", "--to", "9"], "frame 3: the clip has 3 frames"),
                         (prog, ["--node", "lift", "--to", "2", "--at", "5"], "--at 5 is outside the range 0..2")):
        assert main(["track", str(p), str(clip), "--from", "0"] + args) == 1
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and msg in err and "Traceback" not in err, err


@FFMPEG
def test_read_luma_gives_the_frames_asked_in_one_pass(tmp_path):
    from cierra.video import read_luma
    clip = _clip(tmp_path / "n.mkv", [np.full((H, W), k / 10, np.float32) for k in range(9)])
    got = list(read_luma(clip, 2, 8, 3))
    assert [n for n, _ in got] == [2, 5, 8]
    assert all(abs(float(img.mean()) - n / 10) < 1e-4 and img.shape == (H, W) for n, img in got)
