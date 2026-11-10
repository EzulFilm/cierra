"""Command line:

    python -m cierra check  look.cierra                 check, print warnings and the canonical form
    python -m cierra render look.cierra in.npy out.png  [--input KIND] [--range full|video] [--frame N]
    python -m cierra dctl   look.cierra out_dir [--frame N] [--split] [--in KIND] [--out program|srgb|rec1886]
    python -m cierra wgsl   look.cierra out_dir [--frame N] [--split]    WebGPU compute shader
    python -m cierra cuda   look.cierra out_dir [--frame N] [--split]    CUDA C++ kernel (.cu)
    python -m cierra lut    look.cierra out_dir [--size 65]
    python -m cierra cdl    look.cierra out_dir                ASC CDL .cc / .ccc / .cdl (programs that are one CDL)
    python -m cierra fit    in.npy reference.png out.cierra [--input KIND] [--range R] [--frame N]   a reference look as curve / balance / colour (+ hue curves)
    python -m cierra import-lut grade.cube out.cierra [--input acescct]   a 3D LUT back as a readable program
    python -m cierra loop   "notes" in.npy history_dir [--reference ref.png] [--iters 4] [--range R] [--frame N]
    python -m cierra track  look.cierra clip.mov --node NAME --from F0 --to F1 [--at F] [--step K] [-o out.cierra]
                            a node's window / box / graduated mask follows the picture: keys on its centre (and size,
                            angle with --motion similarity, the default); the program is rewritten in canonical form
    python -m cierra scopes clip.mov [--from F0] [--to F1] [--step K] [--range R]  numeric per-frame scopes + flicker
Images: .npy (3, H, W) float, or PNG / TIFF (8, 16-bit or float) read as display codes: the program's own input
when it takes display codes (srgb_photo, rec1886_photo), else srgb_photo; --input overrides.
--range full|video says how the input image's integer codes are coded: full (default: 0..2^n-1 is 0..1) or video
(legal range: 16..235 at 8 bits, 4096..60160 in a 16-bit file, is 0..1; codes outside stay below 0 / over 1). It
applies to the input image only, not to a reference; a float image (.npy, float TIFF) with --range video is refused.
Video (.mov, .mp4, .mxf, .mkv, ... through the system FFmpeg): --frame N reads frame N (0 = first, frame-accurate; on
render it is also the program's frame). Its codes are the program's own input, as a .npy's are; --range defaults to
the stream's tag (untagged YUV: video, untagged RGB: full); YUV becomes R'G'B' by the stream's matrix (untagged:
BT.709 from 720 lines up, BT.601 below). A program that `requires frames` (temporal_denoise) renders from a video
only: the neighbouring frames it reads are decoded with the frame.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .checker import check, load_file
from .errors import CierraError
from .format import format_program
from .parser import parse

RANGES = ("full", "video")


class ImageInputError(ValueError):
    """An image file Cierra cannot read as asked; main() prints it as one line."""


class ImageRangeError(ImageInputError):
    """An image that cannot be read in the range asked for (video range on float data)."""


def _codes_to_unit(a, code_range):
    """Integer codes to 0..1. full: 0..2^n-1. video (legal range): 16*2^(n-8) is 0 and 235*2^(n-8) is 1, n the
    file's bit depth (a 10-bit picture in a 16-bit file is scaled to 16 bits, so 4096..60160); codes below and
    above (sub-blacks, super-whites) are kept as values below 0 and over 1, not clipped here, so the input decode
    decides what to do with them."""
    if code_range == "video":
        step = 2 ** (np.iinfo(a.dtype).bits - 8)
        return (a.astype(np.float64) - 16 * step) / (219 * step)
    return a.astype(np.float64) / np.iinfo(a.dtype).max


def read_image(path, code_range="full", frame=0):
    """An image as (3, H, W) float64. code_range "full" or "video" says how integer codes map to 0..1 (None: full,
    or a video stream's own tag). Video on float data (.npy, float TIFF) is refused rather than ignored: a float file
    has no code scale to read 16..235 against, and ignoring the flag would hand back a picture the user asked to be
    read differently. frame picks a video's frame; a still is the same picture at every frame."""
    if code_range not in RANGES + (None,):
        raise ImageRangeError(f"range {code_range!r}: expected full or video")
    p = Path(path)
    from . import video
    if video.is_video(p):
        try:
            return video.read_frame(p, frame, code_range)
        except video.VideoError as e:
            raise ImageInputError(str(e)) from None
    code_range = code_range or "full"
    if p.suffix.lower() == ".npy":
        if code_range == "video":
            raise ImageRangeError(f"{p}: --range video is for integer images (PNG, 8 / 16-bit TIFF), not .npy floats")
        return np.load(p).astype(np.float64)
    if p.suffix.lower() in (".tif", ".tiff"):  # Pillow cannot open 16-bit RGB TIFFs; float TIFFs keep their values
        import tifffile
        a = tifffile.imread(p, key=0)  # the first page of a multi-page TIFF
        if a.ndim not in (2, 3):
            raise ImageInputError(f"{p}: a TIFF image of shape {a.shape} (expected H x W, with up to 4 channels)")
        if a.ndim == 3 and a.shape[0] in (2, 3, 4) and a.shape[-1] not in (1, 2, 3, 4):  # planar (C, H, W)
            a = a.transpose(1, 2, 0)
        if a.ndim == 3 and a.shape[-1] < 3:  # grey (+ alpha)
            a = a[..., 0]
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        a = a[..., :3]
        if np.issubdtype(a.dtype, np.integer):
            return _codes_to_unit(a, code_range).transpose(2, 0, 1)
        if code_range == "video":
            raise ImageRangeError(f"{p}: --range video is for integer TIFFs (8 / 16-bit), not {a.dtype} floats")
        return a.astype(np.float64).transpose(2, 0, 1)
    from PIL import Image
    with Image.open(p) as im:
        depth, colour_type = _png_depth(p) if im.format == "PNG" else (8, None)
        if depth == 16 and colour_type != 0:  # Pillow hands 16-bit colour PNGs back as 8-bit RGB
            raise ImageInputError(f"{p}: a 16-bit colour PNG would be read as 8-bit; save it as a 16-bit TIFF")
        if depth == 16:  # 16-bit grey: Pillow keeps it ("I;16"); convert("RGB") would clip it to 8 bits
            a = np.repeat(np.asarray(im).astype(np.uint16)[..., None], 3, axis=-1)
        else:
            a = np.asarray(im.convert("RGB"))
    return _codes_to_unit(a, code_range).transpose(2, 0, 1)


def input_range(path, code_range):
    """The range an input is read in: the one asked for, else full for a still and the stream's tag for a video,
    so fit and loop record the range actually used."""
    from . import video
    if code_range or not video.is_video(path):
        return code_range or "full"
    try:
        return video.default_range(video.probe(path))
    except video.VideoError as e:
        raise ImageInputError(str(e)) from None


def _input_frame(path, frame):
    """fit / loop: --frame picks a video frame; on a still it would be ignored, so it is refused instead."""
    from . import video
    if frame is not None and not video.is_video(path):
        raise ImageInputError(f"{path}: --frame is for video input, not a still image")
    return frame or 0


def _read_with_neighbours(prog, path, code_range, frame):
    """render: the frame, and the neighbouring frames a temporal op reads ({offset: image}, None when the program
    reads none), in one decode; clip ends just have fewer. A still has no neighbours, so it is refused."""
    from . import video
    from .engine import frames_needed
    if "frames" not in prog.requires:
        return read_image(path, code_range, frame), None
    if not video.is_video(path):
        raise ImageInputError(f"{path}: the program requires frames (temporal_denoise): render a video, not a still")
    n = frames_needed(prog)
    first = max(frame - n, 0)
    try:
        got = video.read_frames(path, first, frame + n - first + 1, code_range)
        if len(got) <= frame - first:
            video.read_frame(path, frame, code_range)  # says how many frames the clip has
    except video.VideoError as e:
        raise ImageInputError(str(e)) from None
    return got[frame - first], {first + i - frame: im for i, im in enumerate(got) if first + i != frame}


def _png_depth(p):
    """(bit depth, colour type) from a PNG's header (IHDR is always the first chunk)."""
    with open(p, "rb") as f:
        head = f.read(26)
    return head[24], head[25]


def _int_in(lo, hi):
    """argparse type: an integer in lo..hi."""
    def parse(s):
        v = int(s)
        if not lo <= v <= hi:
            raise argparse.ArgumentTypeError(f"{v} outside {lo}..{hi}")
        return v
    return parse


def write_png(path, disp):
    from PIL import Image
    disp = np.nan_to_num(disp, nan=0.0, posinf=1.0, neginf=0.0)  # NaN has no uint8 value: black, as a display shows it
    a = (np.clip(disp, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    Image.fromarray(a).save(path)


def main(argv=None):
    from .fit_lut import MAX_CUBE
    from .ops import LIMITS
    FRAME = _int_in(0, LIMITS["max_frame"])
    ap = argparse.ArgumentParser(prog="cierra")
    ap.add_argument("--device", choices=("cpu", "gpu"),
                    help="where the engine runs (render, fit, loop, tuner): default CIERRA_DEVICE or cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("program")
    r = sub.add_parser("render")
    r.add_argument("program"), r.add_argument("image"), r.add_argument("out")
    r.add_argument("--input"), r.add_argument("--frame", type=FRAME, default=0)
    RANGE_HELP = ("how the input image's integer codes are coded: full or video (16..235 at 8 bits is 0..1); default "
                  "full, or a video stream's own tag")
    FRAME_HELP = "the frame of a video input to read (0 = first)"
    r.add_argument("--range", dest="code_range", choices=RANGES, help=RANGE_HELP)
    d = sub.add_parser("dctl")
    d.add_argument("program"), d.add_argument("out_dir")
    d.add_argument("--frame", type=FRAME, default=0), d.add_argument("--split", action="store_true")
    from .compile.dctl_wrap import INPUTS as WRAP_INPUTS, OUTPUTS as WRAP_OUTPUTS
    d.add_argument("--in", dest="wrap_in", choices=WRAP_INPUTS, default="acescct",
                   help="decode the node's input from this kind to ACEScct (a Rec.709 / YRGB project: rec1886_photo, "
                        "or the camera log); default acescct (an ACEScct timeline)")
    d.add_argument("--out", dest="wrap_out", choices=WRAP_OUTPUTS,
                   help="encode for a display through a 3D LUT written next to the DCTL: the program's own output, "
                        "or srgb / rec1886; default acescct, or program when --in is not acescct")
    d.add_argument("--lut-size", type=int, default=97, choices=range(2, 130), metavar="2..129",
                   help="points per side of the output LUT (default 97, fitted for tetrahedral interpolation)")
    w = sub.add_parser("wgsl")
    w.add_argument("program"), w.add_argument("out_dir")
    w.add_argument("--frame", type=FRAME, default=0), w.add_argument("--split", action="store_true")
    cu = sub.add_parser("cuda")
    cu.add_argument("program"), cu.add_argument("out_dir")
    cu.add_argument("--frame", type=FRAME, default=0), cu.add_argument("--split", action="store_true")
    lu = sub.add_parser("lut")
    lu.add_argument("program"), lu.add_argument("out_dir"), lu.add_argument("--size", type=_int_in(2, MAX_CUBE), default=65)
    cd = sub.add_parser("cdl")
    cd.add_argument("program"), cd.add_argument("out_dir")
    fi = sub.add_parser("fit")
    fi.add_argument("image"), fi.add_argument("reference"), fi.add_argument("out")
    fi.add_argument("--input", default="linear_rec709")
    fi.add_argument("--range", dest="code_range", choices=RANGES, help=RANGE_HELP)
    fi.add_argument("--frame", type=FRAME, help=FRAME_HELP)
    il = sub.add_parser("import-lut")
    il.add_argument("cube"), il.add_argument("out")
    il.add_argument("--input", default="acescct", help="the LUT's domain: acescct (default), acescg, linear_rec709, srgb_display, rec709_display (gamma 2.4)")
    lo = sub.add_parser("loop")
    lo.add_argument("notes"), lo.add_argument("image"), lo.add_argument("history")
    lo.add_argument("--input", default="linear_rec709"), lo.add_argument("--reference")
    lo.add_argument("--iters", type=_int_in(1, 50), default=4), lo.add_argument("--start")
    lo.add_argument("--driver", choices=("auto", "openweight", "anthropic", "rules"), help="default: CIERRA_DRIVER or auto")
    lo.add_argument("--range", dest="code_range", choices=RANGES, help=RANGE_HELP)
    lo.add_argument("--frame", type=FRAME, help=FRAME_HELP)
    lo.add_argument("--against", choices=("ungraded", "start"), default="ungraded",
                    help="judge the notes against the ungraded frame (default) or the --start program, e.g. a look")
    lo.add_argument("--viewing-width", type=_int_in(64, 16384),
                    help="score after box-downsampling the render to this width (default: the 480 px render preview)")
    tr = sub.add_parser("track", help="make a node's mask follow the picture of a video clip (keys on its centre)")
    tr.add_argument("program"), tr.add_argument("clip")
    tr.add_argument("--node", required=True, help="the node whose window / box / graduated mask is tracked")
    tr.add_argument("--from", dest="first", type=FRAME, required=True, help="first frame of the range")
    tr.add_argument("--to", dest="last", type=FRAME, required=True, help="last frame of the range")
    tr.add_argument("--at", type=FRAME, help="the frame where the mask sits on its target (default --from); the "
                    "track runs forwards and backwards from it")
    tr.add_argument("--step", type=_int_in(1, 1000), default=1, help="track every K-th frame")
    tr.add_argument("--shape", type=_int_in(1, LIMITS["mask_calls"]), help="which trackable shape of the mask (1 = first)")
    tr.add_argument("--motion", choices=("similarity", "translate"), default="similarity",
                    help="similarity: position, size and angle (default); translate: position only")
    tr.add_argument("--patch", type=_int_in(2, 4096), help="half-size in pixels of the patch followed (default: "
                    "inside the mask's shape)")
    tr.add_argument("--search", type=_int_in(1, 4096), help="pixels a frame may move from the predicted place")
    tr.add_argument("--min-match", type=float, default=0.6, help="correlation below which the track is lost (0.6)")
    tr.add_argument("--tolerance", type=float, default=0.5,
                    help="pixels the keys may miss the track by between keyframes (0.5)")
    tr.add_argument("-o", "--out", help="write here instead of rewriting the program")
    sc = sub.add_parser("scopes", help="numeric per-frame scopes and mean-luma flicker of a still or video")
    sc.add_argument("image")
    sc.add_argument("--from", dest="first", type=FRAME, default=0, help="first frame (default 0)")
    sc.add_argument("--to", dest="last", type=FRAME, help="last frame (default: only --from)")
    sc.add_argument("--step", type=_int_in(1, 1000), default=1, help="measure every K-th frame")
    sc.add_argument("--range", dest="code_range", choices=RANGES, help=RANGE_HELP)
    a = ap.parse_args(argv)
    if a.device:
        from .backend import gpu_unavailable
        if a.device == "gpu" and gpu_unavailable():
            print(f"--device gpu: {gpu_unavailable()}", file=sys.stderr)
            return 1
        os.environ["CIERRA_DEVICE"] = a.device
    try:
        return _run(a)
    except CierraError as e:  # a program the parser or checker refuses: say why, no traceback
        print(f"{getattr(a, 'program', '')}: {e}".lstrip(": "), file=sys.stderr)
        return 1
    except ImageInputError as e:  # e.g. --range video on a float image: one line, no traceback
        print(e, file=sys.stderr)
        return 1


def _run(a):
    if a.cmd == "check":
        prog, warnings = check(parse(Path(a.program).read_text(encoding="utf-8")))
        for w in warnings:
            print("warning:", w, file=sys.stderr)
        print(format_program(prog), end="")
    elif a.cmd == "render":
        from .engine import render
        from .video import is_video
        prog = load_file(a.program)
        img, near = _read_with_neighbours(prog, a.image, a.code_range, a.frame)
        # a .npy or a video keeps the program's input (a camera file holds the codes the program declares, log or
        # video); PNG / TIFF are display codes, decoded as the program says when it takes display codes
        # (rec1886_photo from a display LUT fit), else as sRGB
        own = a.image.lower().endswith(".npy") or is_video(a.image) or prog.input in ("srgb_photo", "rec1886_photo")
        kind = a.input or (prog.input if own else "srgb_photo")
        aux = {"frames": near} if near is not None else None
        write_png(a.out, render(prog, img, frame=a.frame, aux=aux, input=kind))
    elif a.cmd in ("dctl", "wgsl", "cuda"):
        from .compile.cuda import CUDAError, compile_cuda
        from .compile.dctl import DCTLError, compile_dctl
        from .compile.wgsl import WGSLError, compile_wgsl
        compile_fn, key = {"dctl": compile_dctl, "wgsl": compile_wgsl, "cuda": compile_cuda}[a.cmd], a.cmd
        ext = {"cuda": "cu"}.get(key, key)
        from .compile.dctl_wrap import WrapError
        stem = Path(a.program).stem
        try:
            if key == "dctl":
                wrap_out = a.wrap_out or ("acescct" if a.wrap_in == "acescct" else "program")
                parts = compile_dctl(load_file(a.program), a.frame, a.split, a.wrap_in, wrap_out, a.lut_size, stem)
            else:
                parts = compile_fn(load_file(a.program), a.frame, a.split)
        except (DCTLError, WGSLError, CUDAError, WrapError) as e:
            print(f"NOT IN {key.upper()}:", e, file=sys.stderr)
            return 1
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for part in parts:
            if part[key]:
                name = stem + ("" if part["name"] == "main" else "_" + part["name"]) + "." + ext
                (out / name).write_text(part[key])
                print("wrote", out / name, "-", part["note"])
                from .compile.lut import write_cube
                for lut_name, lat in part.get("luts", {}).items():
                    write_cube(out / lut_name, lat, f"Cierra output LUT for {name} (ACEScct in, display out)")
                    print("wrote", out / lut_name, "- output LUT; keep it next to", name)
            else:
                print(f"NOT IN {key.upper()}:", part["note"])
    elif a.cmd == "lut":
        from .compile.lut import LUTError, export_luts
        try:
            files = export_luts(load_file(a.program), a.out_dir, Path(a.program).stem, a.size)
        except LUTError as e:
            print("NOT A LUT:", e, file=sys.stderr)
            return 1
        print(json.dumps(files, indent=1))
    elif a.cmd == "cdl":
        from .compile.cdl import CDLError, export_cdl
        try:
            files = export_cdl(load_file(a.program), a.out_dir, Path(a.program).stem)
        except CDLError as e:
            print("NOT A CDL:", e, file=sys.stderr)
            return 1
        print(json.dumps(files, indent=1))
    elif a.cmd == "fit":
        from .fit import fit_reference
        from .video import is_video
        rng, frame = input_range(a.image, a.code_range), _input_frame(a.image, a.frame)
        prog, info = fit_reference(read_image(a.image, rng, frame), read_image(a.reference, None), input=a.input)
        info.update(input=a.input, range=rng)  # how the image was read: needed to reproduce the fit
        if is_video(a.image):
            info.update(frame=frame)
        Path(a.out).write_text(format_program(prog), encoding="utf-8")
        print(json.dumps(info))
    elif a.cmd == "import-lut":
        from .fit_lut import LUTFitError, fit_lut
        try:
            prog, report = fit_lut(a.cube, input=a.input, name=Path(a.cube).stem)
        except (LUTFitError, OSError) as e:
            print(f"{a.cube}: {e}", file=sys.stderr)
            return 1
        Path(a.out).write_text(format_program(prog), encoding="utf-8")
        print(json.dumps(report, indent=1))
    elif a.cmd == "loop":
        from .history import History
        from .llm import grade_loop, make_provider
        from .video import is_video
        ref = read_image(a.reference, None) if a.reference else None
        start = Path(a.start).read_text() if a.start else None
        rng = input_range(a.image, a.code_range)
        frame = _input_frame(a.image, a.frame)
        img = read_image(a.image, rng, frame)
        best, recs = grade_loop(a.notes, img, make_provider(a.driver), History(a.history),
                                input=a.input, reference=ref, start=start, max_iters=a.iters, code_range=rng,
                                against=a.against, video_frame=frame if is_video(a.image) else None,
                                viewing_width=a.viewing_width)
        print(json.dumps(recs, indent=1))
        if best is not None:
            print(format_program(best))
    elif a.cmd == "scopes":
        from .scopes import timeline_scopes, video_scopes
        from .video import is_video, VideoError
        try:
            if is_video(a.image):
                report = video_scopes(a.image, a.first, a.last, a.step, input_range(a.image, a.code_range))
            else:
                if a.first or a.last not in (None, 0) or a.step != 1:
                    raise ImageInputError(f"{a.image}: --from / --to / --step are for video input, not a still image")
                report = timeline_scopes([(0, read_image(a.image, a.code_range, 0))])
        except (VideoError, ValueError) as e:
            raise ImageInputError(str(e)) from None
        print(json.dumps(report, indent=1))
    elif a.cmd == "track":
        from .checker import load
        from .track import TrackError, track_program
        from .video import VideoError
        try:
            prog, report = track_program(load_file(a.program), a.clip, a.node, a.first, a.last, at=a.at, step=a.step,
                                         shape=a.shape, motion=a.motion, patch=a.patch, search=a.search,
                                         min_match=a.min_match, tolerance=a.tolerance)
        except (TrackError, VideoError) as e:
            print(e, file=sys.stderr)
            return 1
        text = format_program(prog)
        load(text)  # the keys written must check before the program is overwritten (a CheckError: a tracker bug)
        dest = Path(a.out or a.program)  # a new file, then a rename: a failed write never leaves the program half written
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, dest)
        print(json.dumps(report))


if __name__ == "__main__":
    sys.exit(main())
