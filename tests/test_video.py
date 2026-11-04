"""Video input through the system FFmpeg: exact codes (no double range conversion), the stream's range and matrix,
and frame-accurate selection. Clips are made here from known raw codes, lossless (FFV1), a few frames at 64 x 36."""

import shutil
import subprocess

import numpy as np
import pytest

from cierra.__main__ import ImageInputError, main, read_image

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
                                reason="needs ffmpeg and ffprobe on PATH")
W, H = 64, 36


def _clip(path, frames, pix_fmt, codec=("-c:v", "ffv1"), tags=()):
    """frames: list of raw planes as FFmpeg's rawvideo wants them for pix_fmt (bytes), W x H, 25 fps."""
    raw = path.with_suffix(".raw")
    raw.write_bytes(b"".join(frames))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{W}x{H}", "-r", "25",
                    "-i", str(raw), *codec, *tags, str(path)], check=True)
    return path


def _yuv444(y, cb=128, cr=128, dtype=np.uint8):
    """One yuv444p frame: y a row of luma codes repeated down the picture (or a scalar), flat chroma."""
    y = np.broadcast_to(np.resize(np.asarray(y, dtype), W), (H, W))
    planes = [y, np.full((H, W), cb, dtype), np.full((H, W), cr, dtype)]
    return b"".join(p.astype(dtype if dtype == np.uint8 else "<u2").tobytes() for p in planes)


TV709 = ("-color_range", "tv", "-colorspace", "bt709")
CODES = [4, 16, 126, 235, 250]  # sub-black, legal black, mid, legal white, super-white


def test_legal_range_codes_are_exact_and_not_converted_twice(tmp_path):
    """A stream tagged tv reads legal range by default: 16 is 0, 235 is 1 exactly, grey stays grey, sub-blacks and
    super-whites are kept outside 0..1. (swscale's own expansion put 235 at 65283 / 65535, and expanding there and
    again here would crush black.) --range full reads the same codes as code / 255."""
    v = _clip(tmp_path / "v.mkv", [_yuv444(CODES)], "yuv444p", tags=TV709)
    got = read_image(v, None)
    assert got.shape == (3, H, W) and got.dtype == np.float64
    want = (np.array(CODES) - 16) / 219
    assert np.allclose(got[:, 0, :5], want, atol=1e-12)
    assert got[0, 0, 0] < 0 and got[0, 0, 4] > 1
    assert np.allclose(read_image(v, "video"), got)
    assert np.allclose(read_image(v, "full")[:, 0, :5], np.array(CODES) / 255, atol=1e-12)


def test_10_bit_422_legal_range(tmp_path):
    """ProRes-like 10-bit 4:2:2: 64 is 0 and 940 is 1, read at the stream's own depth."""
    y = np.resize(np.array([64, 502, 940, 1019], np.uint16), W)
    frame = (np.broadcast_to(y, (H, W)).astype("<u2").tobytes() + np.full((H, W // 2), 512, "<u2").tobytes() * 2)
    v = _clip(tmp_path / "t.mkv", [frame], "yuv422p10le", tags=TV709)
    got = read_image(v, None)[:, 0, :4]
    assert np.allclose(got, (np.array([64, 502, 940, 1019]) - 64) / 876, atol=1e-12)


def test_full_range_tag_is_the_default(tmp_path):
    v = _clip(tmp_path / "pc.mkv", [_yuv444([0, 128, 255])], "yuv444p", tags=("-color_range", "pc"))
    assert np.allclose(read_image(v, None)[:, 0, :3], np.array([0, 128, 255]) / 255, atol=1e-12)


@pytest.mark.parametrize("tags,ycbcr", [
    (TV709, (63, 102, 240)),  # BT.709 legal-range 100% red
    (("-color_range", "tv", "-colorspace", "smpte170m"), (81, 90, 240)),  # BT.601 legal-range 100% red
    (("-color_range", "tv"), (81, 90, 240)),  # untagged matrix, 36 lines: SD, so BT.601
])
def test_matrix_from_the_tag(tmp_path, tags, ycbcr):
    """Published code values of 100% red decode to (1, 0, 0) only with the right matrix and the legal chroma scale
    (224 steps, not 219)."""
    v = _clip(tmp_path / "red.mkv", [_yuv444(ycbcr[0], ycbcr[1], ycbcr[2])], "yuv444p", tags=tags)
    rgb = read_image(v, None)[:, 10, 10]
    assert np.allclose(rgb, [1, 0, 0], atol=0.01), rgb


def test_rgb_stream_reads_channels_in_order(tmp_path):
    a = np.zeros((H, W, 3), np.uint8)
    a[..., 0], a[..., 1], a[..., 2] = 200, 100, 7
    v = _clip(tmp_path / "rgb.mkv", [a.tobytes()], "rgb24")
    assert np.allclose(read_image(v, None)[:, 0, 0], np.array([200, 100, 7]) / 255, atol=1e-12)


def _numbered(tmp_path, codec, name):
    """Eight frames, frame k flat luma 16 + 20 k."""
    return _clip(tmp_path / name, [_yuv444(16 + 20 * k) for k in range(8)], "yuv444p", codec=codec, tags=TV709)


@pytest.mark.parametrize("codec", [("-c:v", "ffv1"),
                                   ("-c:v", "libx264", "-qp", "0", "-g", "4", "-bf", "2", "-pix_fmt", "yuv444p")])
def test_frame_number_is_exact(tmp_path, codec):
    """Every frame N comes back as frame N, also from a long-GOP stream with B-frames (frames stored out of
    display order), where a time seek can land on a neighbour."""
    if codec[1] == "libx264" and "libx264" not in subprocess.run(["ffmpeg", "-v", "error", "-encoders"],
                                                                 capture_output=True, text=True).stdout:
        pytest.skip("ffmpeg without libx264")
    v = _numbered(tmp_path, codec, "n.mkv" if codec[1] == "ffv1" else "n.mp4")
    for k in range(8):
        assert np.allclose(read_image(v, "video", k), 20 * k / 219, atol=1e-12), k


def test_frame_past_the_end_is_one_line(tmp_path, capsys):
    v = _numbered(tmp_path, ("-c:v", "ffv1"), "n.mkv")
    with pytest.raises(ImageInputError, match=r"frame 8: the clip has 8 frames \(0..7\)"):
        read_image(v, None, 8)
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\n")
    capsys.readouterr()
    assert main(["render", str(tmp_path / "p.cierra"), str(v), str(tmp_path / "o.png"), "--frame", "9"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and "8 frames" in err


def test_render_a_video_frame(tmp_path):
    """render takes a clip and --frame: the program's own input (here display codes), the stream's legal range, so an
    identity program gives frame 3's grey (luma 76: (76 - 16) / 219 of white) back."""
    from PIL import Image
    v = _numbered(tmp_path, ("-c:v", "ffv1"), "n.mkv")
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\n")
    assert main(["render", str(tmp_path / "p.cierra"), str(v), str(tmp_path / "o.png"), "--frame", "3"]) in (0, None)
    out = np.asarray(Image.open(tmp_path / "o.png"), dtype=float)
    assert np.abs(out - 60 / 219 * 255).max() <= 1


def test_frame_on_a_still_is_refused_for_fit(tmp_path, capsys):
    """fit / loop have no animation: --frame on a still would be ignored, so it is one plain line instead."""
    from PIL import Image
    Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(tmp_path / "s.png")
    capsys.readouterr()
    assert main(["fit", str(tmp_path / "s.png"), str(tmp_path / "s.png"), str(tmp_path / "o.cierra"),
                 "--frame", "2"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "--frame is for video" in err


def test_not_a_video_is_one_line(tmp_path):
    (tmp_path / "bad.mov").write_bytes(b"not a video")
    with pytest.raises(ImageInputError, match="bad.mov"):
        read_image(tmp_path / "bad.mov", None)


def test_loop_history_records_the_video_frame(tmp_path, capsys):
    """loop on a clip: each history entry says which frame, range and input were graded, to reproduce it."""
    import json
    v = _numbered(tmp_path, ("-c:v", "ffv1"), "n.mkv")
    assert main(["loop", "warmer", str(v), str(tmp_path / "h"), "--frame", "5", "--iters", "1", "--driver", "rules",
                 "--input", "srgb_photo"]) in (0, None)
    entry = json.loads((tmp_path / "h" / "log.jsonl").read_text().splitlines()[0])
    assert entry["source"] == {"input": "srgb_photo", "range": "video", "frame": 5}
