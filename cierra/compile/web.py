"""The playground's compile side, shared by the local server (tools/playground_server.py) and the browser-only build
(tools/build_playground_static.py runs this module in Pyodide): numpy and the standard library only, no
OpenColorIO, so it runs in WebAssembly. What needs OpenColorIO (the ACES 2.0 output LUT, camera-log decodes, a
display-referred photo's photo_graph) is precomputed or done by the server."""

import numpy as np

from .. import colour as C
from ..checker import load
from ..errors import CierraError

LUT_N = 97  # the output LUT for aces2 (65 left 0.56 ΔE00 at p99.9 on the chart; 97: 0.23)
MAX_WIDTH = 1920


def compile_program(src, frame=0):
    from ..compile.wgsl import WGSLError, compile_wgsl
    try:
        prog = load(src)
    except CierraError as e:
        return {"ok": False, "errors": str(e).splitlines()}
    try:
        parts = compile_wgsl(prog, frame=int(frame))
    except (WGSLError, CierraError, ValueError) as e:
        return {"ok": False, "errors": ["the WGSL target cannot hold this program: " + str(e)]}
    warnings = []
    try:
        from ..checker import lint
        warnings = [str(w) for w in lint(prog, {})]
    except Exception:  # lint is advisory
        pass
    return {"ok": True, "wgsl": parts[0]["wgsl"], "warnings": warnings, "output": output_text(prog),
            "input": prog.input, "photo_graph": prog.photo_graph is not None}


def output_text(prog):
    o = prog.output
    args = ", ".join(f"{k}: {v}" for k, v in sorted(o.args.items()))
    return f"{o.name}({args})"


def _f(x):
    return f"{float(x):.9g}f"


IDENTITY_SHADER = """
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> size : vec2<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= size.x || gid.y >= size.y) { return; }
  let p = 3u * (gid.y * size.x + gid.x);
  dst[p] = src[p]; dst[p + 1u] = src[p + 1u]; dst[p + 2u] = src[p + 2u];
}
"""


def output_shader(output):
    """display_encode(...) as a WGSL compute shader with the shader buffer layout (cierra.colour.display_encode, line
    for line), or None for outputs that go through the LUT (aces2: OpenColorIO; acescct: no transform)."""
    o = load(f"cierra 0.1\noutput {output}\n").output
    if o.name == "acescct":  # no transform: the shader's ACEScct shown as it is (exact, no LUT)
        return IDENTITY_SHADER
    if o.name != "display_encode":
        return None
    eotf, clip = o.args.get("eotf", "srgb"), o.args.get("clip", "soft")
    m = C.MAP1_709
    mat = "mat3x3<f32>(" + ", ".join(_f(m[r][c]) for c in range(3) for r in range(3)) + ")"  # column-major
    wy = ", ".join(_f(v) for v in C.REC709_Y)
    if clip == "hard":
        clip_code = "let o = clamp(lin, vec3<f32>(0.0f), vec3<f32>(1.0f));"
    else:
        clip_code = """let y = clamp(dot(lin, W), 0.0f, 1.0f);
  let peak = max(lin.x, max(lin.y, lin.z));
  var rgb = lin;
  if (peak > KNEE) {
    let tgt = KNEE + (1.0f - KNEE) * tanh((peak - KNEE) / (1.0f - KNEE));
    let t = clamp((tgt - y) / max(peak - y, 1e-9f), 0.0f, 1.0f);
    rgb = y + (lin - y) * t;
  }
  let low = min(rgb.x, min(rgb.y, rgb.z));
  if (low < 0.0f) {
    let t2 = clamp(y / max(y - low, 1e-9f), 0.0f, 1.0f);
    rgb = y + (rgb - y) * t2;
  }
  let o = clamp(rgb, vec3<f32>(0.0f), vec3<f32>(1.0f));"""
    if eotf == "gamma24":
        enc = "pow(max(v, 0.0f), 1.0f / 2.4f)"
    else:
        enc = "select(1.055f * pow(max(v, 0.0f), 1.0f / 2.4f) - 0.055f, 12.92f * v, v <= 0.0031308f)"
    return f"""
@group(0) @binding(0) var<storage, read> src : array<f32>;
@group(0) @binding(1) var<storage, read_write> dst : array<f32>;
@group(0) @binding(2) var<uniform> size : vec2<u32>;
const M : mat3x3<f32> = {mat};
const W : vec3<f32> = vec3<f32>({wy});
const KNEE : f32 = 0.9f;
fn dec(y : f32) -> f32 {{ return select(pow(2.0f, y * 17.52f - 9.72f), (y - {_f(C.CCT_B)}) / {_f(C.CCT_A)}, y <= {_f(C.CCT_CUT_CODE)}); }}
fn enc(v : f32) -> f32 {{ return {enc}; }}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {{
  if (gid.x >= size.x || gid.y >= size.y) {{ return; }}
  let p = 3u * (gid.y * size.x + gid.x);
  let lin = M * vec3<f32>(dec(src[p]), dec(src[p + 1u]), dec(src[p + 2u]));
  {clip_code}
  dst[p] = enc(o.x); dst[p + 1u] = enc(o.y); dst[p + 2u] = enc(o.z);
}}
"""


def downscale(img):
    """Box-downsample to at most MAX_WIDTH wide (an integer step, deterministic)."""
    s = -(-img.shape[-1] // MAX_WIDTH)
    if s <= 1:
        return img
    h, w = img.shape[1] // s * s, img.shape[2] // s * s
    return img[:, :h, :w].reshape(3, h // s, s, w // s, s).mean(axis=(2, 4))


def working_frame(prog, img, input_space):
    """(3, H, W) image in input_space -> ACEScct float32, with the program's photo_graph for display-referred input."""
    code, display_referred = C.input_to_working(np.asarray(img, dtype=np.float64), input_space)
    if display_referred and prog.photo_graph is not None:
        from ..engine import make_ctx, run_graph
        ctx = make_ctx(prog, code.shape[-1])
        ctx["prog"], ctx["display_referred"] = prog, True
        code = run_graph(code, prog.photo_graph, prog, ctx)
    return np.asarray(code, dtype=np.float32), bool(display_referred)




def frame_bytes(code, input_space, display_referred):
    """ACEScct (3, H, W) -> the playground's frame payload: 4-byte length, JSON header, float32 r, g, b interleaved."""
    import json
    import struct
    code = np.asarray(code, dtype=np.float32)
    head = json.dumps({"width": code.shape[2], "height": code.shape[1], "input": input_space,
                       "display_referred": bool(display_referred)}).encode()
    return struct.pack("<I", len(head)) + head + np.ascontiguousarray(code.transpose(1, 2, 0)).tobytes()


def decode_rgba8(data, width, height, src):
    """A dropped PNG / JPEG decoded by the browser (RGBA, 8 bits) -> the frame payload, read as an sRGB photo. For the
    browser-only build: a program with a photo_graph (photo_to_scene needs OpenColorIO) is refused with the reason."""
    prog = load(src or "cierra 0.1\n")
    if prog.photo_graph is not None:
        raise CierraError("photo_graph (photo_to_scene) needs OpenColorIO: run the local server "
                          "(tools/playground_server.py) for this program")
    a = np.frombuffer(bytes(data), dtype=np.uint8).reshape(int(height), int(width), 4)[..., :3]
    img = downscale(a.transpose(2, 0, 1).astype(np.float64) / 255.0)
    code, disp = C.input_to_working(img, "srgb_photo")
    return frame_bytes(code, "srgb_photo", disp)
