"""Build the browser-only playground: a plain folder any static host (or `python -m http.server`) can serve.

    python tools/build_playground_static.py [--out out/playground_static]

Writes index.html and app.js (the same page as web/playground: with no /api it switches to this mode), and static/:
  cierra.zip       the cierra package (.py files only): its compiler runs in the browser in Pyodide
                   (cierra.compile.web: numpy and the standard library only)
  manifest.json    the Pyodide release (pinned, loaded from cdn.jsdelivr.net) and the files below
  luts/*.bin       the ACES 2.0 output LUTs (OpenColorIO has no WebAssembly build), 97^3 float32, red fastest
  frames/*.bin     the built-in test frames as ACEScct (CC0 photos and the synthetic chart), the frame payload
What it cannot do without the server, and says so: camera-log inputs on dropped files, a photo_graph
(photo_to_scene), .npy / TIFF drops (PNG / JPEG only), outputs other than aces2 srgb / rec1886, display_encode, acescct.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/"  # Python 3.12.7, numpy 2.0.2 (released 2025-01)
OUTPUTS = ("aces2(display: srgb)", "aces2(display: rec1886)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out" / "playground_static"))
    a = ap.parse_args()
    import playground_server as PS
    from cierra import load
    from cierra.compile.web import LUT_N, frame_bytes
    out = Path(a.out)
    st = out / "static"
    (st / "luts").mkdir(parents=True, exist_ok=True)
    (st / "frames").mkdir(parents=True, exist_ok=True)
    for f in ("index.html", "app.js"):
        (out / f).write_text((ROOT / "web" / "playground" / f).read_text())
    with zipfile.ZipFile(st / "cierra.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((ROOT / "cierra").rglob("*.py")):
            z.write(p, p.relative_to(ROOT).as_posix())
    luts = {}
    for o in OUTPUTS:
        name = o.replace("(display: ", "_").replace(")", "") + f"_{LUT_N}.bin"
        (st / "luts" / name).write_bytes(PS.lut_bytes(o))
        luts[o] = {"file": name, "n": LUT_N}
    frames = []
    empty = load("cierra 0.1\n")
    for name, (path, space) in PS.BUILTIN.items():
        img = PS.downscale(np.load(path).astype(np.float64))
        code, disp = PS.working_frame(empty, img, space)
        (st / "frames" / f"{name}.bin").write_bytes(frame_bytes(code, space, disp))
        frames.append({"name": name, "input": space, "file": f"{name}.bin"})
    (st / "manifest.json").write_text(json.dumps({"pyodide": PYODIDE, "luts": luts, "frames": frames}, indent=1))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"built {out} ({size / 1e6:.1f} MB): {len(frames)} frames, {len(luts)} LUTs, cierra.zip "
          f"{(st / 'cierra.zip').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
