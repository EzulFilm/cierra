"""The web playground's server side (tools/playground_server.py): compile replies, the exact display_encode shader,
the output LUT layout and frame upload. The browser side is measured by tools/playground_parity.py (headless Chromium
with WebGPU; not in CI)."""

import base64
import json
import struct
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.compile.wgsl_run import adapter, run_wgsl
from conftest import small

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import playground_server as PS  # noqa: E402

needs_wgpu = pytest.mark.skipif(adapter()[0] is None, reason=f"WGSL harness: {adapter()[1]}")


def test_compile_reply_ok_and_refusals():
    r = PS.compile_program("cierra 0.1\noutput aces2(display: srgb)\nnode a = exposure(stops: 0.5)\ngraph a\n")
    assert r["ok"] and "fn main" in r["wgsl"] and r["output"] == "aces2(display: srgb)"
    bad = PS.compile_program("cierra 0.1\nnode a = exposure(stops: 12)\ngraph a\n")
    assert not bad["ok"] and "outside" in " ".join(bad["errors"])
    spatial = PS.compile_program("cierra 0.1\nnode g = glow(threshold: 0.6, radius_px: 10, strength: 0.2)\ngraph g\n")
    assert not spatial["ok"] and "spatial" in " ".join(spatial["errors"])


def test_lut_layout_is_red_fastest():
    n = PS.LUT_N
    t = np.frombuffer(PS.lut_bytes("acescct()"), np.float32).reshape(n, n, n, 3)  # [b, g, r, channel]
    v = np.linspace(0, 1, n, dtype=np.float32)
    assert np.allclose(t[3, 5, 7], [v[7], v[5], v[3]])
    assert PS.output_shader("aces2(display: srgb)") is None
    assert "dst[p] = src[p]" in PS.output_shader("acescct()")  # no transform: an exact identity shader


@needs_wgpu
@pytest.mark.parametrize("out", ["display_encode(eotf: srgb)", "display_encode(eotf: gamma24)",
                                 "display_encode(eotf: srgb, clip: hard)"])
def test_display_encode_shader_matches_colour(chart, out):
    code = C.to_working(small(chart)) * 1.15 - 0.03  # push some channels past the knee and below zero
    got = run_wgsl(PS.output_shader(out), code)
    o = load(f"cierra 0.1\noutput {out}\n").output
    ref = C.display_encode(code, o.args.get("eotf", "srgb"), o.args.get("clip", "soft"))
    assert np.abs(got - ref).max() < 2e-4


def test_frame_upload_png_and_builtin():
    from PIL import Image
    a = (np.linspace(0, 255, 64 * 32 * 3) % 256).astype(np.uint8).reshape(32, 64, 3)
    buf = BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    body = PS.frame_payload({"program": "cierra 0.1\n", "name": "x.png", "data": base64.b64encode(buf.getvalue()).decode()})
    n = struct.unpack("<I", body[:4])[0]
    head = json.loads(body[4:4 + n])
    assert head["width"] == 64 and head["height"] == 32 and head["input"] == "srgb_photo"
    assert len(body) - 4 - n == 64 * 32 * 3 * 4
    body = PS.frame_payload({"program": "cierra 0.1\n", "builtin": "chart"})
    head = json.loads(body[4:4 + struct.unpack("<I", body[:4])[0]])
    assert head["input"] == "linear_rec709" and not head["display_referred"]
