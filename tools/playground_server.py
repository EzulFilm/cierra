"""Cierra web playground: a local server for web/playground/ (standard library only).

    python tools/playground_server.py [--port 8790] [--host 127.0.0.1]
    then open http://127.0.0.1:8790/  (a browser with WebGPU: Chrome / Edge 113+, Safari 26, Firefox 141+)

What runs where:
  - in the browser (WebGPU): the program's WGSL shader (cierra/compile/wgsl.py, the same code as `cierra wgsl`), then
    the program's output transform (display_encode exactly in WGSL; aces2 as a 97^3 LUT, tetrahedral, ACEScct 0..1 in -> display code values out), then the
    scopes (waveform, vectorscope, RGB histogram), all on the graded pixels;
  - here: parsing + checking + compiling the program (the compiler is Python), and turning a dropped frame into
    ACEScct (the program's input space; a display-referred photo also through the program's photo_graph, which is not
    part of the shader, as for DCTL), and building the output LUT.
Nothing is written to disk except a dropped file's temporary copy; the server listens on 127.0.0.1 by default.

API (JSON unless said):
  POST /api/compile   {program, frame?}            -> {ok, wgsl, warnings, output, input} | {ok: false, errors}
  POST /api/frame     {program, name, data (base64)} | {program, builtin}
                                                   -> binary: 4-byte little-endian header length, JSON header
                                                      {width, height, input, display_referred}, float32 ACEScct r,g,b
                                                      interleaved row-major (the shader's buffer layout)
  GET  /api/output?output=<output call>            -> {kind: "wgsl", wgsl} (display_encode: exact, in the shader) or
                                                      {kind: "lut", n} (aces2 through OCIO, acescct: sampled)
  GET  /api/lut?output=<output call>               -> binary float32 [N^3 x 3], r fastest, then g, then b (N = 97)
  GET  /api/frames                                 -> built-in test frames (linear Rec.709 .npy, chart_photo sRGB)
"""

import argparse
import base64
import json
import struct
import sys
import tempfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.errors import CierraError  # noqa: E402
from cierra.parser import parse  # noqa: E402
from cierra.compile.web import (LUT_N, MAX_WIDTH, compile_program, downscale, output_shader,  # noqa: E402,F401
                                output_text, working_frame)

WEB = ROOT / "web" / "playground"
MAX_BODY = 64 << 20
BUILTIN = {**{p.stem: (p, "linear_rec709") for p in sorted((ROOT / "tests" / "data" / "frames").glob("*.npy"))},
           "chart_photo": (ROOT / "tests" / "data" / "chart_photo.npy", "srgb_photo")}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css"}


@lru_cache(maxsize=8)
def lut_bytes(output):
    """The output transform sampled on N^3 ACEScct 0..1, r fastest (float32 r, g, b per entry)."""
    prog = load(f"cierra 0.1\noutput {output}\n")
    v = np.linspace(0.0, 1.0, LUT_N)
    b, g, r = np.meshgrid(v, v, v, indexing="ij")
    code = np.stack([r, g, b]).reshape(3, LUT_N * LUT_N, LUT_N)
    out = C.output_from_working(code, prog.output)
    return np.ascontiguousarray(np.asarray(out, dtype=np.float32).reshape(3, -1).T).tobytes()


def frame_payload(req):
    prog = load(req.get("program") or "cierra 0.1\n")
    if req.get("builtin"):
        path, space = BUILTIN[req["builtin"]]
        img = np.load(path).astype(np.float64)
    else:
        from cierra.__main__ import read_image
        suffix = Path(req.get("name") or "frame.png").suffix.lower() or ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(base64.b64decode(req["data"]))
            f.flush()
            img = read_image(f.name)
        # a float .npy is scene-linear Rec.709 unless the program says otherwise; 8 / 16-bit pictures are photos
        space = prog.input if (suffix == ".npy" or prog.input not in ("linear_rec709",)) else "srgb_photo"
    img = downscale(img)
    code, disp = working_frame(prog, img, space)
    head = json.dumps({"width": code.shape[2], "height": code.shape[1], "input": space,
                       "display_referred": disp}).encode()
    return struct.pack("<I", len(head)) + head + np.ascontiguousarray(code.transpose(1, 2, 0)).tobytes()


class Handler(BaseHTTPRequestHandler):
    server_version = "cierra-playground"

    def log_message(self, fmt, *args):
        pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path, _, query = self.path.partition("?")
        if path == "/api/frames":
            return self.send(200, [{"name": k, "input": v[1]} for k, v in BUILTIN.items()])
        if path == "/api/output":
            from urllib.parse import parse_qs, unquote
            out = unquote(parse_qs(query).get("output", ["aces2(display: srgb)"])[0])
            try:
                code = output_shader(out)
            except (CierraError, ValueError) as e:
                return self.send(400, {"error": str(e)})
            return self.send(200, {"kind": "wgsl", "wgsl": code} if code else {"kind": "lut", "n": LUT_N})
        if path == "/api/lut":
            from urllib.parse import parse_qs, unquote
            out = unquote(parse_qs(query).get("output", ["aces2(display: srgb)"])[0])
            try:
                parse(f"cierra 0.1\noutput {out}\n")
                return self.send(200, lut_bytes(out), "application/octet-stream")
            except (CierraError, ValueError) as e:
                return self.send(400, {"error": str(e)})
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        f = (WEB / name).resolve()
        if WEB.resolve() not in f.parents or not f.is_file():
            return self.send(404, {"error": "not found"})
        return self.send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self.send(413, {"error": "too large"})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/api/compile":
                return self.send(200, compile_program(req.get("program", ""), req.get("frame", 0)))
            if self.path == "/api/frame":
                return self.send(200, frame_payload(req), "application/octet-stream")
        except (CierraError, ValueError, KeyError, OSError) as e:
            return self.send(400, {"error": f"{type(e).__name__}: {e}"})
        return self.send(404, {"error": "not found"})


def serve(host="127.0.0.1", port=8790):
    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args()
    httpd = serve(a.host, a.port)
    print(f"Cierra playground on http://{a.host}:{httpd.server_address[1]}/", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
