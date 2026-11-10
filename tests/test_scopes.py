"""Numeric per-frame and timeline scopes."""

import json
import shutil
import subprocess

import numpy as np
import pytest

from cierra.scopes import timeline_scopes, video_scopes


def test_timeline_scopes_keep_frame_identity_and_measure_flicker():
    frames = [(10, np.full((3, 4, 6), 0.2)), (12, np.full((3, 4, 6), 0.5)),
              (14, np.full((3, 4, 6), 0.35))]
    out = timeline_scopes(frames)
    assert (out["first_frame"], out["last_frame"], out["count"]) == (10, 14, 3)
    assert [f["frame"] for f in out["frames"]] == [10, 12, 14]
    assert [f["tone"]["p50"] for f in out["frames"]] == [0.2, 0.5, 0.35]
    assert out["frames"][0]["waveform"]["columns"] == 6
    assert out["frames"][0]["vectorscope"]["chroma_p50"] == 0
    assert out["flicker"]["max_step"] == 0.3 and out["flicker"]["mean_step"] == 0.225


def test_timeline_scopes_refuse_empty_or_mismatched_frames():
    with pytest.raises(ValueError, match="at least one"):
        timeline_scopes([])
    with pytest.raises(ValueError, match="same shape"):
        timeline_scopes([(0, np.zeros((3, 4, 6))), (1, np.zeros((3, 4, 5)))])


def test_video_scopes_reads_the_requested_frame_grid_and_range(monkeypatch):
    from cierra import video
    calls = []

    def read(path, n, code_range):
        calls.append((path, n, code_range))
        return np.full((3, 2, 2), n / 10)

    monkeypatch.setattr(video, "read_frame", read)
    out = video_scopes("clip.mov", 1, 5, 2, "video")
    assert calls == [("clip.mov", 1, "video"), ("clip.mov", 3, "video"), ("clip.mov", 5, "video")]
    assert [f["frame"] for f in out["frames"]] == [1, 3, 5]
    for args in ((-1, 0, 1), (3, 2, 1), (0, 2, 0)):
        with pytest.raises(ValueError, match="scope frames"):
            video_scopes("clip.mov", *args)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs FFmpeg")
def test_scopes_cli_reads_real_video_frames_end_to_end(tmp_path, capsys):
    from cierra.__main__ import main
    w, h = 8, 6
    raw = b""
    for code in (32, 64, 96):
        raw += (np.full((h, w), code, np.uint8).tobytes()
                + np.full((h, w), 128, np.uint8).tobytes() * 2)
    (tmp_path / "clip.raw").write_bytes(raw)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv444p", "-s", f"{w}x{h}",
                    "-r", "25", "-i", str(tmp_path / "clip.raw"), "-c:v", "ffv1", "-color_range", "pc",
                    "-colorspace", "bt709", str(tmp_path / "clip.mkv")], check=True)
    assert main(["scopes", str(tmp_path / "clip.mkv"), "--from", "0", "--to", "2", "--step", "2"]) in (0, None)
    out = json.loads(capsys.readouterr().out)
    assert [f["frame"] for f in out["frames"]] == [0, 2]
    assert [f["tone"]["p50"] for f in out["frames"]] == [round(32 / 255, 3), round(96 / 255, 3)]
    assert out["flicker"]["max_step"] == round(64 / 255, 4)
    assert main(["scopes", str(tmp_path / "clip.mkv"), "--from", "2", "--to", "1"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and "first <= last" in err
