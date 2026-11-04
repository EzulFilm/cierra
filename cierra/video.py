"""One frame of a video file (H.264, ProRes, MXF, anything the system FFmpeg reads) as (3, H, W) float64.

FFmpeg and ffprobe are called as programs (no Python package). The frame is decoded at the stream's own bit depth,
4:4:4 (only the chroma is upsampled), and the YUV -> R'G'B' matrix and the code range are applied here, in float:
swscale's own range expansion is not exact at 16 bits (legal white 235 came out as 65283 of 65535), its 8 -> 16-bit
step differs between full and legal range, and RGB output clips sub-blacks and super-whites, which stills keep.
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SUFFIXES = {".mov", ".mp4", ".m4v", ".mxf", ".mkv", ".avi", ".webm", ".mts", ".m2ts", ".ts", ".mpg", ".mpeg",
            ".y4m", ".nut", ".dv", ".ogv", ".3gp", ".wmv", ".flv"}
DEPTHS = (8, 9, 10, 12, 14, 16)  # the 4:4:4 planar formats FFmpeg has: yuv444p[N] / gbrp[N]
# Kr, Kb of each YUV matrix FFmpeg tags (color_space)
MATRICES = {"bt709": (0.2126, 0.0722), "bt470bg": (0.299, 0.114), "smpte170m": (0.299, 0.114),
            "bt2020nc": (0.2627, 0.0593), "fcc": (0.30, 0.11), "smpte240m": (0.212, 0.087)}


class VideoError(ValueError):
    """A video Cierra cannot read as asked; read_image turns it into an ImageInputError (one line, no traceback)."""


def is_video(path):
    return Path(path).suffix.lower() in SUFFIXES


def _tool(name, p):
    exe = shutil.which(name)
    if not exe:
        raise VideoError(f"{p}: reading a video needs {name} (FFmpeg) on PATH")
    return exe


def probe(path):
    """The first video stream: width, height, pix_fmt, bit depth, rgb, range, matrix, fps (no decode)."""
    p = Path(path)
    if not p.is_file():
        raise VideoError(f"{p}: no such file")
    r = subprocess.run([_tool("ffprobe", p), "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,pix_fmt,color_range,color_space,r_frame_rate,nb_frames", "-of", "json",
                        str(p)], capture_output=True, text=True)
    streams = json.loads(r.stdout or "{}").get("streams") if r.returncode == 0 else None
    if not streams:
        raise VideoError(f"{p}: FFmpeg finds no video stream ({(r.stderr.strip().splitlines() or ['no output'])[-1]})")
    s = streams[0]
    fmt = _pix_fmt(s.get("pix_fmt"), p)
    num, _, den = s.get("r_frame_rate", "0/1").partition("/")
    num, den = float(num or 0), float(den or 1)
    return {"width": s["width"], "height": s["height"], "pix_fmt": s.get("pix_fmt"), "depth": fmt["depth"],
            "rgb": fmt["rgb"], "color_range": s.get("color_range", "unknown"),
            "color_space": s.get("color_space", "unknown"),
            "fps": num / den if den else 0.0, "nb_frames": s.get("nb_frames")}


def _pix_fmt(name, p):
    """Bit depth and rgb flag of a pixel format, from FFmpeg's own table (names like p010 or v210 do not say)."""
    r = subprocess.run([_tool("ffprobe", p), "-v", "error", "-show_pixel_formats", "-of", "json"],
                       capture_output=True, text=True)
    for f in json.loads(r.stdout or "{}").get("pixel_formats", []):
        if f["name"] == name:
            flags = f.get("flags", {})
            if flags.get("float") or flags.get("hwaccel") or flags.get("bayer"):
                raise VideoError(f"{p}: pixel format {name} is not read (8 to 16-bit integer YUV / RGB / grey only)")
            depth = max(c["bit_depth"] for c in f["components"][:3])
            return {"depth": depth, "rgb": bool(flags.get("rgb") or flags.get("palette"))}
    raise VideoError(f"{p}: unknown pixel format {name!r}")


def default_range(info):
    """The code range when the user gave none: the stream's tag; untagged YUV is legal range (the broadcast norm,
    and what FFmpeg itself assumes), untagged RGB full."""
    return {"tv": "video", "pc": "full"}.get(info["color_range"], "full" if info["rgb"] else "video")


def frame_count(path):
    """Frames in the first video stream, by counting packets (a demux, no decode)."""
    r = subprocess.run([_tool("ffprobe", path), "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return None


def read_frame(path, frame=0, code_range=None):
    """Frame `frame` (0 = first, in presentation order) as (3, H, W) float64 R'G'B', the codes mapped to 0..1 by
    code_range ("full", "video", or None: default_range). Frame-accurate: the select filter keeps decoded frame n
    exactly, where a time seek can land on a neighbour (rounded timestamps, B-frames, a start offset); it decodes
    every frame before it, so late frames of a long clip take a while. Codes outside the range are kept below 0 /
    over 1, as for stills."""
    got = read_frames(path, frame, 1, code_range)
    if not got:
        n = frame_count(path)
        has = f"the clip has {n} frames (0..{n - 1})" if n else "the clip has no such frame"
        raise VideoError(f"{Path(path)}: frame {frame}: {has}")
    return got[0]


def read_frames(path, first=0, count=1, code_range=None):
    """Frames first .. first + count - 1 as a list of (3, H, W), read as read_frame reads one, in one decode (a
    temporal op reads a frame's neighbours); shorter when the clip ends first."""
    p = Path(path)
    info = probe(p)
    code_range = code_range or default_range(info)
    depth = next((d for d in DEPTHS if d >= info["depth"]), None)
    if depth is None:
        raise VideoError(f"{p}: {info['depth']}-bit video is not read (8 to 16 bits)")
    out = ("gbrp" if info["rgb"] else "yuv444p") + ("" if depth == 8 else f"{depth}le")
    w, h = info["width"], info["height"]
    # same depth in and out and no range given to swscale: it only upsamples chroma, the codes pass unchanged
    cmd = [_tool("ffmpeg", p), "-v", "error", "-nostdin", "-noautorotate", "-i", str(p), "-map", "0:v:0",
           "-vf", f"select=between(n\\,{first}\\,{first + count - 1}),"
                  "scale=flags=accurate_rnd+full_chroma_int+bitexact",
           "-fps_mode", "passthrough", "-frames:v", str(count), "-f", "rawvideo", "-pix_fmt", out, "-"]
    r = subprocess.run(cmd, capture_output=True)
    dtype = np.uint8 if depth == 8 else np.dtype("<u2")
    size = 3 * h * w * np.dtype(dtype).itemsize
    if r.returncode != 0:
        raise VideoError(f"{p}: FFmpeg could not decode it ({(r.stderr.decode(errors='replace').strip().splitlines() or ['no message'])[-1]})")
    if len(r.stdout) % size:
        raise VideoError(f"{p}: FFmpeg returned a partial frame ({len(r.stdout)} bytes, frames are {size})")
    frames = np.frombuffer(r.stdout, dtype).reshape(-1, 3, h, w).astype(np.float64)
    if info["rgb"]:
        return [_unit(a[[2, 0, 1]], depth, code_range) for a in frames]  # gbrp planes: G, B, R
    m = _matrix(info, p)
    return [_yuv_to_rgb(a, depth, code_range, m) for a in frames]


def read_luma(path, first, last, step=1):
    """Frames first, first + step, ... up to last as (n, (H, W) float32 luma 0..1), decoded in one FFmpeg pass that
    stops after the last frame: a tracker reads hundreds of frames, and read_frame per frame would decode the clip
    from its start every time. Luma as swscale makes it (gray16): the tracker's matching ignores gain and offset, so
    range and matrix do not matter there."""
    p = Path(path)
    info = probe(p)
    w, h = info["width"], info["height"]
    count = (last - first) // step + 1
    sel = f"select=between(n\\,{first}\\,{last})*not(mod(n-{first}\\,{step}))"
    cmd = [_tool("ffmpeg", p), "-v", "error", "-nostdin", "-noautorotate", "-i", str(p), "-map", "0:v:0",
           "-vf", f"{sel},scale=flags=accurate_rnd+bitexact", "-fps_mode", "passthrough", "-frames:v", str(count),
           "-f", "rawvideo", "-pix_fmt", "gray16le", "-"]
    size = h * w * 2
    with tempfile.TemporaryFile() as err:  # a file, not a pipe: FFmpeg cannot block on a full stderr
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            for i in range(count):
                buf = proc.stdout.read(size)
                if len(buf) != size:
                    proc.wait()
                    err.seek(0)
                    msg = err.read().decode(errors="replace").strip().splitlines()
                    if proc.returncode:
                        raise VideoError(f"{p}: FFmpeg could not decode it ({(msg or ['no message'])[-1]})")
                    n = frame_count(p)
                    has = f"the clip has {n} frames (0..{n - 1})" if n else "the clip has no such frame"
                    raise VideoError(f"{p}: frame {first + i * step}: {has}")
                yield first + i * step, np.frombuffer(buf, "<u2").reshape(h, w).astype(np.float32) / 65535
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()


def _matrix(info, p):
    """Kr, Kb from the stream's tag; untagged: BT.709 from 720 lines up (HD), BT.601 below (SD), as players do."""
    cs = info["color_space"]
    if cs in MATRICES:
        return MATRICES[cs]
    if cs in ("unknown", "unspecified", "reserved", "gbr", None):  # gbr: a grey stream tagged as RGB
        return MATRICES["bt709" if info["height"] >= 720 else "smpte170m"]
    raise VideoError(f"{p}: colour matrix {cs} is not read (BT.709, BT.601, BT.2020 non-constant, FCC, SMPTE 240M)")


def _unit(a, depth, code_range):
    """Integer codes at `depth` bits to 0..1: full 0..2^n-1, video 16..235 scaled to n bits (as _codes_to_unit)."""
    if code_range == "video":
        step = 2.0 ** (depth - 8)
        return (a - 16 * step) / (219 * step)
    return a / (2 ** depth - 1)


def _yuv_to_rgb(a, depth, code_range, matrix):
    """Y'CbCr codes to R'G'B' 0..1. Legal-range chroma spans 16..240 (224 steps, not luma's 219), so the range is
    applied to Y and to Cb / Cr separately before the matrix."""
    y = _unit(a[0], depth, code_range)
    if code_range == "video":
        step = 2.0 ** (depth - 8)
        cb, cr = (a[1] - 128 * step) / (224 * step), (a[2] - 128 * step) / (224 * step)
    else:
        cb, cr = (a[1] - 2 ** (depth - 1)) / (2 ** depth - 1), (a[2] - 2 ** (depth - 1)) / (2 ** depth - 1)
    kr, kb = matrix
    kg = 1 - kr - kb
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b])
