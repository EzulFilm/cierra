"""tools/raw_to_video.py without a camera file: downscaling, the frame loop with a keyframed program, and the FFmpeg
encode (skipped when ffmpeg is not installed; it is an external program, not part of Cierra)."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from cierra import load
from cierra.engine import render
from conftest import ROOT

sys.path.insert(0, str(ROOT / "tools"))
import raw_to_video as rv  # noqa: E402


def test_downscale_is_at_most_the_width_and_even():
    img = np.random.default_rng(0).random((3, 101, 333))
    out = rv.downscale(img, 100)
    assert out.shape[2] <= 100 and out.shape[1] % 2 == 0 and out.shape[2] % 2 == 0
    assert abs(out.mean() - img.mean()) < 0.01


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_encode_a_keyframed_program(tmp_path):
    prog = load("cierra 0.1\ninput aces2065_1\noutput aces2(display: rec1886)\n"
                "node up = exposure(stops: keys(linear, 0: 0, 3: 1))\ngraph up\n")
    img = np.full((3, 48, 64), 0.18)
    frames = [render(prog, img, frame=i, input="aces2065_1") for i in range(4)]
    assert frames[3].mean() > frames[0].mean()  # the grade moves with the frame
    out = tmp_path / "t.mov"
    assert rv.encode(iter(frames), out, 24, "prores") == 4
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                       "stream=codec_name,width,height,nb_frames,color_primaries", "-of", "json",
                                       str(out)], capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert (probe["codec_name"], probe["width"], probe["height"], probe["nb_frames"]) == ("prores", 64, 48, "4")
    assert probe["color_primaries"] == "bt709"
