"""Command line:

    python -m cierra check  look.cierra                 check, print warnings and the canonical form
    python -m cierra render look.cierra in.npy out.png  [--input KIND] [--frame N]
    python -m cierra dctl   look.cierra out_dir [--frame N] [--split]
    python -m cierra wgsl   look.cierra out_dir [--frame N] [--split]    WebGPU compute shader
    python -m cierra cuda   look.cierra out_dir [--frame N] [--split]    CUDA C++ kernel (.cu)
    python -m cierra lut    look.cierra out_dir [--size 65]
    python -m cierra cdl    look.cierra out_dir                ASC CDL .cc / .ccc / .cdl (programs that are one CDL)
    python -m cierra fit    in.npy reference.png out.cierra [--input KIND]   a reference look as curve / balance / colour
    python -m cierra import-lut grade.cube out.cierra [--input acescct]   a 3D LUT back as a readable program
    python -m cierra loop   "notes" in.npy history_dir [--reference ref.png] [--iters 4]
Images: .npy (3, H, W) float, or PNG / TIFF (8, 16-bit or float) read as display codes: the program's own input
when it takes display codes (srgb_photo, rec1886_photo), else srgb_photo; --input overrides.
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


def read_image(path):
    p = Path(path)
    if p.suffix == ".npy":
        return np.load(p).astype(np.float64)
    if p.suffix.lower() in (".tif", ".tiff"):  # Pillow cannot open 16-bit RGB TIFFs; float TIFFs keep their values
        import tifffile
        a = tifffile.imread(p, key=0)  # the first page of a multi-page TIFF
        if a.ndim not in (2, 3):
            raise ValueError(f"{p}: a TIFF image of shape {a.shape} (expected H x W, with up to 4 channels)")
        if a.ndim == 3 and a.shape[0] in (2, 3, 4) and a.shape[-1] not in (1, 2, 3, 4):  # planar (C, H, W)
            a = a.transpose(1, 2, 0)
        if a.ndim == 3 and a.shape[-1] < 3:  # grey (+ alpha)
            a = a[..., 0]
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        a = a[..., :3]
        scale = np.iinfo(a.dtype).max if np.issubdtype(a.dtype, np.integer) else 1.0
        return (a.astype(np.float64) / scale).transpose(2, 0, 1)
    from PIL import Image
    a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float64) / 255.0
    return a.transpose(2, 0, 1)


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
    d = sub.add_parser("dctl")
    d.add_argument("program"), d.add_argument("out_dir")
    d.add_argument("--frame", type=FRAME, default=0), d.add_argument("--split", action="store_true")
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
    il = sub.add_parser("import-lut")
    il.add_argument("cube"), il.add_argument("out")
    il.add_argument("--input", default="acescct", help="the LUT's domain: acescct (default), acescg, linear_rec709, srgb_display, rec709_display (gamma 2.4)")
    lo = sub.add_parser("loop")
    lo.add_argument("notes"), lo.add_argument("image"), lo.add_argument("history")
    lo.add_argument("--input", default="linear_rec709"), lo.add_argument("--reference")
    lo.add_argument("--iters", type=_int_in(1, 50), default=4), lo.add_argument("--start")
    lo.add_argument("--driver", choices=("auto", "openweight", "anthropic", "rules"), help="default: CIERRA_DRIVER or auto")
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


def _run(a):
    if a.cmd == "check":
        prog, warnings = check(parse(Path(a.program).read_text(encoding="utf-8")))
        for w in warnings:
            print("warning:", w, file=sys.stderr)
        print(format_program(prog), end="")
    elif a.cmd == "render":
        from .engine import render
        prog = load_file(a.program)
        img = read_image(a.image)
        # a .npy keeps the program's input; PNG / TIFF are display codes, decoded as the program says when it takes
        # display codes (rec1886_photo from a display LUT fit), else as sRGB
        own = a.image.endswith(".npy") or prog.input in ("srgb_photo", "rec1886_photo")
        kind = a.input or (prog.input if own else "srgb_photo")
        write_png(a.out, render(prog, img, frame=a.frame, input=kind))
    elif a.cmd in ("dctl", "wgsl", "cuda"):
        from .compile.cuda import CUDAError, compile_cuda
        from .compile.dctl import DCTLError, compile_dctl
        from .compile.wgsl import WGSLError, compile_wgsl
        compile_fn, key = {"dctl": compile_dctl, "wgsl": compile_wgsl, "cuda": compile_cuda}[a.cmd], a.cmd
        ext = {"cuda": "cu"}.get(key, key)
        try:
            parts = compile_fn(load_file(a.program), a.frame, a.split)
        except (DCTLError, WGSLError, CUDAError) as e:
            print(f"NOT IN {key.upper()}:", e, file=sys.stderr)
            return 1
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(a.program).stem
        for part in parts:
            if part[key]:
                name = stem + ("" if part["name"] == "main" else "_" + part["name"]) + "." + ext
                (out / name).write_text(part[key])
                print("wrote", out / name, "-", part["note"])
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
        prog, info = fit_reference(read_image(a.image), read_image(a.reference), input=a.input)
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
        ref = read_image(a.reference) if a.reference else None
        start = Path(a.start).read_text() if a.start else None
        best, recs = grade_loop(a.notes, read_image(a.image), make_provider(a.driver), History(a.history),
                                input=a.input, reference=ref, start=start, max_iters=a.iters)
        print(json.dumps(recs, indent=1))
        if best is not None:
            print(format_program(best))


if __name__ == "__main__":
    sys.exit(main())
