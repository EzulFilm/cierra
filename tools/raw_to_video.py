"""A whole pipeline: camera RAW -> scene-linear ACES2065-1 -> a Cierra program -> a movie file.

    python tools/raw_to_video.py examples/tungsten_fix.cierra shot.dng --frames 72 --out out/pipeline/shot.mov
    python tools/raw_to_video.py look.cierra frame_*.dng --out shot.mov --codec h264

RAW stills (DNG, CR2 / CR3, NEF, ARW, RAF, ...) are decoded with LibRaw through rawpy (`pip install -e ".[raw]"`):
camera white balance, no auto brightness, linear, ACES2065-1 primaries. Several RAW files become one frame each; a
single file is held for --frames frames, so a keyframed program animates over it. The program runs in Cierra's
reference engine with input aces2065_1, frame by frame, and FFmpeg (an external program, not part of Cierra) encodes
the display-referred result:
  prores   ProRes-compatible 422 HQ, 10-bit (FFmpeg's prores_ks encoder), .mov
  h264     H.264 4:2:0 8-bit (needs an FFmpeg built with libx264), .mp4
Next to the movie it writes <out>_before.png (the ungraded frame through the program's output transform) and
<out>_after.png (the last graded frame). Cine RAW formats (BRAW, R3D, ARRIRAW, ...) need their vendors' SDKs and
are not read here: convert them to EXR or DNG first.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra.checker import load_file  # noqa: E402
from cierra.engine import render  # noqa: E402
from cierra.format import format_program  # noqa: E402
from cierra.llm import ungraded  # noqa: E402

CODECS = {
    "prores": ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le"],
    "h264": ["-c:v", "libx264", "-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"],
}
# Rec.709 tags: the program's output should be aces2(display: rec1886) for video (gamma 2.4 display)
TAGS = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"]


def decode_raw(path, wb="camera"):
    """A RAW still -> (3, H, W) float64 scene-linear ACES2065-1, no auto brightness. wb "camera": the white balance
    the camera recorded; "daylight": LibRaw's daylight balance, so the light's own colour stays in the picture for
    the program to correct."""
    import rawpy
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(output_color=rawpy.ColorSpace.ACES, gamma=(1, 1), no_auto_bright=True,
                              output_bps=16, use_camera_wb=(wb == "camera"))
    return (rgb.astype(np.float64) / 65535.0).transpose(2, 0, 1)


def downscale(img, width):
    """Box-average to at most `width` pixels wide (integer factor), then crop to even sizes for 4:2:x video."""
    s = max(1, -(-img.shape[2] // width))
    h, w = img.shape[1] // s * s, img.shape[2] // s * s
    img = img[:, :h, :w].reshape(3, h // s, s, w // s, s).mean(axis=(2, 4))
    return img[:, : img.shape[1] // 2 * 2, : img.shape[2] // 2 * 2]


def to_rgb48(disp):
    return (np.clip(disp, 0, 1).transpose(1, 2, 0) * 65535 + 0.5).astype("<u2")


def encode(frames, out, fps, codec):
    """Frames (iterable of display-referred (3, H, W)) -> a movie through an FFmpeg subprocess."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found: install it (it is not part of Cierra) or write PNG frames instead")
    frames = iter(frames)
    first = next(frames)
    h, w = first.shape[1:]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *CODECS[codec], *TAGS, str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    for disp in _chain(first, frames):
        proc.stdin.write(to_rgb48(disp).tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit(f"ffmpeg failed (codec {codec}); try --codec prores")
    return n


def _chain(first, rest):
    yield first
    yield from rest


def write_png(path, disp):
    from PIL import Image
    Image.fromarray((np.clip(disp, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("program")
    ap.add_argument("raw", nargs="+", help="RAW stills, one per frame (or one, held for --frames)")
    ap.add_argument("--out", default="out/pipeline/shot.mov")
    ap.add_argument("--frames", type=int, default=48, help="frames to hold a single still for (default 48)")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--width", type=int, default=1920, help="largest output width (default 1920)")
    ap.add_argument("--codec", choices=sorted(CODECS), default="prores")
    ap.add_argument("--wb", choices=("camera", "daylight"), default="camera", help="RAW white balance (default camera)")
    a = ap.parse_args(argv)

    prog = load_file(a.program)
    if prog.input != "aces2065_1":
        print(f"note: the program says input {prog.input}; RAW is decoded to aces2065_1 and rendered as that",
              file=sys.stderr)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stills = [downscale(decode_raw(p, a.wb), a.width) for p in a.raw]
    n = a.frames if len(stills) == 1 else len(stills)
    animated = "keys(" in format_program(prog)

    def frames():
        last = None
        for i in range(n):
            img = stills[min(i, len(stills) - 1)]
            if animated or len(stills) > 1 or last is None:
                last = render(prog, img, frame=i, input="aces2065_1")
            yield last

    count = encode(frames(), out, a.fps, a.codec)
    write_png(out.with_name(out.stem + "_before.png"), render(ungraded(prog), stills[0], input="aces2065_1"))
    write_png(out.with_name(out.stem + "_after.png"),
              render(prog, stills[-1], frame=n - 1, input="aces2065_1"))
    print(f"wrote {out} ({count} frames, {stills[0].shape[2]}x{stills[0].shape[1]}, {a.codec}) and before / after PNGs")


if __name__ == "__main__":
    main()
