"""temporal_denoise (spec §5): grain averages out over neighbouring frames where the picture holds still, moving
areas keep the current frame (no ghosts), a pan is undone first; the CLI reads the neighbours from a video."""

import shutil
import subprocess

import numpy as np
import pytest

from cierra import CierraError, load
from cierra.engine import (box_mean, global_shift, global_shift_subpixel, grade_working, shift_frame,
                           shift_frame_subpixel)

H, W, SIGMA = 64, 96, 0.02
HEAD = "cierra 0.1\noutput acescct()\nrequires frames\n"


def _prog(args="strength: 1, frames: 2, noise: 0.02", before=""):
    pre = f"node e = {before}\n" if before else ""
    return load(HEAD + pre + f"node t = temporal_denoise({args})\ngraph " + ("e -> t" if before else "t") + "\n")


def _scene(dx=0, dy=0, box_x=None):
    """Clean (3, H, W) ACEScct: a smooth, non-periodic texture round mid grey moved by (dy, dx), and a bright 16 px
    box at column box_x."""
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    yy, xx = yy - dy, xx - dx
    t = 0.38 + 0.05 * np.sin(xx / 5.3) * np.cos(yy / 7.1) + 0.03 * np.sin((xx + 2 * yy) / 11.7)
    img = np.stack([t * 1.02, t, t * 0.97])
    if box_x is not None:
        img[:, 24:40, box_x:box_x + 16] = [[[0.62]], [[0.55]], [[0.5]]]
    return img


def _noisy(clean, seed):
    return clean + np.random.default_rng(seed).normal(0, SIGMA, clean.shape)


def _sequence(scenes):
    """Frames 0..n-1 with seeded noise, graded at the middle one: (current noisy, {offset: noisy}, clean current)."""
    c = len(scenes) // 2
    noisy = [_noisy(s, 100 + i) for i, s in enumerate(scenes)]
    return noisy[c], {i - c: n for i, n in enumerate(noisy) if i != c}, scenes[c]


def _rms(x):
    return float(np.sqrt((x ** 2).mean()))


def test_static_noise_falls_by_most_of_sqrt_5():
    """Five frames of a still scene: noise std falls by close to sqrt(5) = 2.24 (the weights stay near 1)."""
    cur, near, clean = _sequence([_scene()] * 5)
    out = grade_working(cur, _prog(), frame=2, aux={"frames": near}, px_scale=1.0)
    inner = (slice(None), slice(4, -4), slice(4, -4))
    factor = _rms((cur - clean)[inner]) / _rms((out - clean)[inner])
    print(f"static: noise std {_rms((cur - clean)[inner]):.4f} -> {_rms((out - clean)[inner]):.4f}, "
          f"factor {factor:.2f}")
    assert factor > 1.9, factor


def test_a_moving_box_does_not_ghost():
    """A bright box moving 6 px a frame over a still background: where the box is or was, the result stays on the
    current clean frame (local mean error far below the 0.25 step a ghost would leave a share of), and the still
    background around it is still denoised."""
    scenes = [_scene(box_x=20 + 6 * i) for i in range(5)]
    cur, near, clean = _sequence(scenes)
    for align in ("true", "false"):
        out = grade_working(cur, _prog(f"strength: 1, frames: 2, noise: 0.02, align: {align}"), frame=2,
                            aux={"frames": near}, px_scale=1.0)
        err = box_mean((out - clean).mean(0), 2)
        moving = (slice(20, 44), slice(16, 64))
        ghost = float(np.abs(err[moving]).max())
        naive = np.mean([cur] + list(near.values()), axis=0)  # a plain average: the ghost to avoid
        naive_ghost = float(np.abs(box_mean((naive - clean).mean(0), 2)[moving]).max())
        still = (slice(4, 16), slice(4, -4))
        factor = _rms((cur - clean)[:, still[0], still[1]]) / _rms((out - clean)[:, still[0], still[1]])
        print(f"moving box, align {align}: max local error {ghost:.4f} (plain average {naive_ghost:.4f}), "
              f"rms error in the moving area {_rms((out - clean)[:, moving[0], moving[1]]):.4f} "
              f"(input {_rms((cur - clean)[:, moving[0], moving[1]]):.4f}), still area factor {factor:.2f}")
        assert naive_ghost > 0.08 and ghost < 0.025, (ghost, naive_ghost)
        assert _rms((out - clean)[:, moving[0], moving[1]]) < _rms((cur - clean)[:, moving[0], moving[1]])
        assert factor > 1.8, factor


def test_a_pan_is_undone_by_align():
    """The whole picture moving (2, -3) px a frame: aligned, it denoises nearly as a still; not aligned, the
    texture's motion counts as motion where it is larger than the noise and less is averaged (the error still falls)."""
    scenes = [_scene(dx=-3 * i, dy=2 * i) for i in range(5)]
    cur, near, clean = _sequence(scenes)
    assert global_shift(clean[1], scenes[3][1]) == (-2, 3)
    assert global_shift(cur[1], near[1][1]) == (-2, 3) and global_shift(cur[1], near[-2][1]) == (4, -6)
    inner = (slice(None), slice(12, -12), slice(12, -12))
    got = {}
    for align in ("true", "false"):
        out = grade_working(cur, _prog(f"strength: 1, frames: 2, noise: 0.02, align: {align}"), frame=2,
                            aux={"frames": near}, px_scale=1.0)
        got[align] = _rms((cur - clean)[inner]) / _rms((out - clean)[inner])
    print(f"pan: factor aligned {got['true']:.2f}, not aligned {got['false']:.2f}")
    assert got["true"] > 1.8 and 1.0 < got["false"] < got["true"] - 0.3, got


def test_shift_frame_moves_whole_pixels():
    x = np.arange(12.0).reshape(1, 3, 4)
    assert np.array_equal(shift_frame(x, 0, 1)[0, 0], [0, 0, 1, 2])
    assert np.array_equal(shift_frame(x, 1, 0)[0, :, 0], [0, 0, 4])


def test_shift_frame_subpixel_is_bilinear_and_exact_at_integers():
    x = np.arange(12.0).reshape(1, 3, 4)
    assert np.array_equal(shift_frame_subpixel(x, 1, 1), shift_frame(x, 1, 1))
    assert np.array_equal(shift_frame_subpixel(x, 0, 0.5)[0, 0], [0, 0.5, 1.5, 2.5])
    assert np.array_equal(shift_frame_subpixel(x, 0.5, 0)[0, :, 0], [0, 2, 6])  # out(y) = x(y - dy), top edge kept


def test_subpixel_pan_is_recovered_and_denoised_without_hallucinating_zero_motion():
    scenes = [_scene(dx=-0.43 * i, dy=0.27 * i) for i in range(5)]
    cur, near, clean = _sequence(scenes)
    dy, dx = global_shift_subpixel(clean[1], scenes[3][1])
    assert abs(dy - (-0.27)) < 0.15 and abs(dx - 0.43) < 0.15, (dy, dx)
    out = grade_working(cur, _prog(), frame=2, aux={"frames": near}, px_scale=1.0)
    inner = (slice(None), slice(12, -12), slice(12, -12))
    factor = _rms((cur - clean)[inner]) / _rms((out - clean)[inner])
    print(f"sub-pixel pan: shift {(dy, dx)}, noise factor {factor:.2f}")
    assert factor > 1.8, factor
    cur, near, _ = _sequence([_scene()] * 5)
    assert global_shift_subpixel(cur[1], near[1][1]) == (0.0, 0.0)
    flat = np.full((H, W), 0.4)
    assert global_shift_subpixel(flat, flat) == (0.0, 0.0)


def test_strength_0_and_no_neighbours_are_the_identity():
    cur, near, _ = _sequence([_scene(box_x=30)] * 3)
    for args, frames in (("strength: 0", near), ("strength: 1, frames: 4, noise: 0.2", {}),
                         ("strength: 1, frames: 1", {5: near[1]})):  # 5 is further than `frames`
        out = grade_working(cur, _prog(args), aux={"frames": frames}, px_scale=1.0)
        assert np.array_equal(out, cur), args


def test_neighbours_go_through_the_nodes_before_it():
    """exposure -> temporal_denoise: the neighbours are compared after the exposure too (else every pixel would
    differ by a stop and nothing would average), each at its own frame's keyframe value."""
    cur, near, clean = _sequence([_scene()] * 5)
    for before in ("exposure(stops: 1)", "exposure(stops: keys(linear, 0: 0.3, 4: 0.7))"):
        prog = _prog(before=before)
        out = grade_working(cur, prog, frame=2, aux={"frames": near}, px_scale=1.0)
        ref = grade_working(clean, load(HEAD + f"node e = {before}\ngraph e\n"), frame=2, aux={"frames": {}},
                            px_scale=1.0)
        noisy = grade_working(cur, load(HEAD + f"node e = {before}\ngraph e\n"), frame=2, aux={"frames": {}},
                              px_scale=1.0)
        assert _rms(noisy - ref) / _rms(out - ref) > 1.9, before


def test_display_referred_neighbours_run_photo_graph_too():
    from cierra.engine import render
    prog = load("cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\nrequires frames\n"
                "node p = photo_to_scene()\nnode t = temporal_denoise(strength: 1, noise: 0.03)\n"
                "photo_graph p\ngraph t\n")
    disp = np.clip(_scene() + 0.1, 0, 1)
    frames = {k: np.clip(_noisy(disp, 7 + k), 0, 1) for k in (-2, -1, 1, 2)}
    cur = np.clip(_noisy(disp, 3), 0, 1)
    ref = render(prog, disp, frame=2, aux={"frames": {}}, px_scale=1.0)
    out = render(prog, cur, frame=2, aux={"frames": frames}, px_scale=1.0)
    plain = render(prog, cur, frame=2, aux={"frames": {}}, px_scale=1.0)
    print(f"display-referred: factor {_rms(plain - ref) / _rms(out - ref):.2f}")
    assert _rms(plain - ref) / _rms(out - ref) > 1.7


def test_checker_rules():
    with pytest.raises(CierraError, match="needs the 'frames' data: add `requires frames`"):
        load("cierra 0.1\nnode t = temporal_denoise()\ngraph t\n")
    for g in ("t -> t", "parallel(t, u)"):
        with pytest.raises(CierraError, match="may be used once in a program, in graph"):
            load(HEAD + f"node t = temporal_denoise()\nnode u = temporal_denoise(frames: 1)\ngraph {g}\n")
    with pytest.raises(CierraError, match="may be used once"):
        load("cierra 0.1\ninput srgb_photo\nrequires frames\nnode t = temporal_denoise()\nphoto_graph t\n")
    with pytest.raises(CierraError, match=r"frames: 5 is outside \[1, 4\]"):
        load(HEAD + "node t = temporal_denoise(frames: 5)\ngraph t\n")
    with pytest.raises(ValueError, match="needs the 'frames' plane"):
        grade_working(_scene(), _prog())
    from cierra.engine import frames_needed
    assert frames_needed(_prog("frames: 3")) == 3 and frames_needed(load("cierra 0.1\n")) == 0


def test_refused_by_lut_and_pixel_shader_targets(tmp_path):
    from cierra.compile.cuda import CUDAError, compile_cuda
    from cierra.compile.dctl import DCTLError, compile_dctl
    from cierra.compile.lut import LUTError, export_luts
    from cierra.compile.wgsl import WGSLError, compile_wgsl
    prog = _prog()
    with pytest.raises(LUTError, match="temporal"):
        export_luts(prog, tmp_path, "tnr")
    for compile_, err in ((compile_dctl, DCTLError), (compile_wgsl, WGSLError), (compile_cuda, CUDAError)):
        with pytest.raises(err, match=r"temporal_denoise\(\) is temporal"):
            compile_(prog)


# ---------------------------------------------------------------- end to end through the CLI and FFmpeg

needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                  reason="needs ffmpeg and ffprobe on PATH")
VW, VH = 64, 36


def _noisy_clip(path, n=6):
    """n lossless 8-bit frames of one grey (luma 120) with seeded noise of 6 codes, full range so 0..255 is 0..1."""
    raw = b""
    rng = np.random.default_rng(11)
    for _ in range(n):
        y = np.clip(np.round(120 + rng.normal(0, 6, (VH, VW))), 0, 255).astype(np.uint8)
        raw += y.tobytes() + np.full((VH, VW), 128, np.uint8).tobytes() * 2
    path.with_suffix(".raw").write_bytes(raw)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{VW}x{VH}",
                    "-r", "25", "-i", str(path.with_suffix(".raw")), "-c:v", "ffv1", "-color_range", "pc",
                    "-colorspace", "bt709", str(path)], check=True)
    return path


@needs_ffmpeg
def test_cli_renders_a_clip_with_its_neighbours(tmp_path, capsys):
    from PIL import Image
    from cierra.__main__ import main
    v = _noisy_clip(tmp_path / "n.mkv")
    head = "cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\n"
    (tmp_path / "plain.cierra").write_text(head)
    (tmp_path / "tnr.cierra").write_text(head + "requires frames\nnode t = temporal_denoise(strength: 1, noise: 0.06)"
                                                "\ngraph t\n")
    std = {}
    for frame in ("2", "0", "5"):  # the middle, and both clip ends (fewer neighbours)
        for name in ("plain", "tnr"):
            out = tmp_path / f"{name}{frame}.png"
            assert main(["render", str(tmp_path / f"{name}.cierra"), str(v), str(out), "--frame", frame]) in (0, None)
            std[name + frame] = float(np.asarray(Image.open(out), dtype=float)[..., 1].std())
    with capsys.disabled():
        print("CLI clip: green std", {k: round(s, 2) for k, s in std.items()})
    assert std["plain2"] / std["tnr2"] > 1.8 and std["plain0"] / std["tnr0"] > 1.4, std
    capsys.readouterr()
    np.save(tmp_path / "still.npy", np.full((3, 8, 8), 0.4))
    assert main(["render", str(tmp_path / "tnr.cierra"), str(tmp_path / "still.npy"), str(tmp_path / "s.png")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and "requires frames" in err and "video" in err
    assert main(["render", str(tmp_path / "tnr.cierra"), str(v), str(tmp_path / "x.png"), "--frame", "9"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "6 frames" in err
