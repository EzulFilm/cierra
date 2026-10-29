"""Cierra -> WGSL (the WebGPU shading language): one compute shader that grades an ACEScct float32 RGB buffer.

Same coverage and maths as the DCTL target (cierra/compile/dctl.py): the op / mask / graph code generator is shared
through its dialect hooks, so the two cannot drift; only the syntax (declarations, casts, select, function heads), the
prelude and the entry point differ here. Every pointwise and position op and mask; keyframes evaluated at one frame;
spatial nodes, blurred mattes and depth / segment masks refused with reasons (cierra.targets.wgsl_node_reasons), or
with split=True a top-level serial graph is cut into several shaders around them; a curve whose points or tangents do
not survive float32 is refused.

Interface (fixed, so a host binds it without reading the text):
    @group(0) @binding(0) var<storage, read>       cr_src  : array<f32>   ACEScct, interleaved r, g, b, row-major
    @group(0) @binding(1) var<storage, read_write> cr_dst  : array<f32>   same layout (in -> out; not in place)
    @group(0) @binding(2) var<uniform>             cr_size : vec2<u32>    width, height in pixels
    @compute @workgroup_size(8, 8) fn main(...)                           dispatch ceil(w / 8) x ceil(h / 8)
Everything is f32, constants are baked in; the program's own `output` and `photo_graph` are not part of the shader
(as for DCTL: ACEScct in, ACEScct out).
"""

from .dctl import DCTLError, _Gen, compile_parts, f
from ..targets import wgsl_node_reasons

WORKGROUP = (8, 8)


class WGSLError(ValueError):
    pass


# DCTL built-ins the shared code generator calls, as one-line WGSL wrappers (the compiler inlines them), then the
# helpers of dctl.PRELUDE written in WGSL. Kept line for line with the C versions.
PRELUDE = r"""
fn make_float3(a: f32, b: f32, c: f32) -> vec3<f32> { return vec3<f32>(a, b, c); }
fn _clampf(x: f32, lo: f32, hi: f32) -> f32 { return clamp(x, lo, hi); }
fn _fmaxf(a: f32, b: f32) -> f32 { return max(a, b); }
fn _fminf(a: f32, b: f32) -> f32 { return min(a, b); }
fn _fabs(x: f32) -> f32 { return abs(x); }
fn _powf(x: f32, y: f32) -> f32 { return pow(x, y); }
fn _expf(x: f32) -> f32 { return exp(x); }
fn _sqrtf(x: f32) -> f32 { return sqrt(x); }
fn _cosf(x: f32) -> f32 { return cos(x); }
fn _sinf(x: f32) -> f32 { return sin(x); }
fn cr_luma(r: f32, g: f32, b: f32) -> f32 { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }
fn cr_smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / max(e1 - e0, 1e-9f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
fn cr_band(x: f32, lo: f32, hi: f32, s: f32) -> f32 {
    return cr_smoothstep(lo - s, lo + s, x) * (1.0f - cr_smoothstep(hi - s, hi + s, x));
}
fn cr_mod360(a: f32) -> f32 { return a - 360.0f * floor(a / 360.0f); }
fn cr_hue(cb: f32, cr: f32) -> f32 {
    if (cb == 0.0f && cr == 0.0f) { return 0.0f; }  // atan2(0, 0) is undefined in WGSL; C and NumPy give 0
    return cr_mod360(atan2(cr, cb) * 57.29577951308232f);
}
fn cr_hue_w(hue: f32, h0: f32, width: f32) -> f32 {
    let d = cr_mod360(hue - h0 + 180.0f) - 180.0f;
    return 0.5f + 0.5f * cos(3.14159265358979f * clamp(d / width, -1.0f, 1.0f));
}
fn cr_herm(x: f32, x0: f32, x1: f32, y0: f32, y1: f32, m0: f32, m1: f32) -> f32 {
    let h = x1 - x0; let t = (x - x0) / h; let t2 = t * t; let t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * y0 + (t3 - 2.0f * t2 + t) * h * m0 + (-2.0f * t3 + 3.0f * t2) * y1
           + (t3 - t2) * h * m1;
}
fn cr_cct_dec(y: f32) -> f32 {
    if (y <= 0.155251141552511f) { return (y - 0.0729055341958355f) / 10.5402377416545f; }
    return exp2(y * 17.52f - 9.72f);
}
fn cr_cct_enc(x: f32) -> f32 {
    if (x <= 0.0078125f) { return 10.5402377416545f * x + 0.0729055341958355f; }
    return (log2(max(x, 1e-12f)) + 9.72f) / 17.52f;
}
"""

DISPLAY_HELPERS = r"""
fn cr_tanh(x: f32) -> f32 { let e = exp(2.0f * clamp(x, -15.0f, 15.0f)); return (e - 1.0f) / (e + 1.0f); }
fn cr_srgb_enc(v: f32) -> f32 {
    if (v <= 0.0031308f) { return 12.92f * v; }
    return 1.055f * pow(max(v, 0.0f), 1.0f / 2.4f) - 0.055f;
}
fn cr_srgb_dec(v: f32) -> f32 {
    if (v <= 0.04045f) { return v / 12.92f; }
    return pow((v + 0.055f) / 1.055f, 2.4f);
}
fn cr_display_encode(r: f32, g: f32, b: f32) -> vec3<f32> {
    let lr = cr_cct_dec(r); let lg = cr_cct_dec(g); let lb = cr_cct_dec(b);
    var x0 = %(A0)s * lr + %(A1)s * lg + %(A2)s * lb;
    var x1 = %(A3)s * lr + %(A4)s * lg + %(A5)s * lb;
    var x2 = %(A6)s * lr + %(A7)s * lg + %(A8)s * lb;
    let y = clamp(0.2126f * x0 + 0.7152f * x1 + 0.0722f * x2, 0.0f, 1.0f);
    let peak = max(x0, max(x1, x2));
    var tgt = peak;
    if (peak > 0.9f) { tgt = 0.9f + 0.1f * cr_tanh((peak - 0.9f) / 0.1f); }
    let t = clamp((tgt - y) / max(peak - y, 1e-9f), 0.0f, 1.0f);
    if (peak > 0.9f) { x0 = y + (x0 - y) * t; x1 = y + (x1 - y) * t; x2 = y + (x2 - y) * t; }
    let low = min(x0, min(x1, x2));
    let t2 = clamp(y / max(y - low, 1e-9f), 0.0f, 1.0f);
    if (low < 0.0f) { x0 = y + (x0 - y) * t2; x1 = y + (x1 - y) * t2; x2 = y + (x2 - y) * t2; }
    return vec3<f32>(cr_srgb_enc(clamp(x0, 0.0f, 1.0f)), cr_srgb_enc(clamp(x1, 0.0f, 1.0f)),
                     cr_srgb_enc(clamp(x2, 0.0f, 1.0f)));
}
fn cr_from_display(r: f32, g: f32, b: f32) -> vec3<f32> {
    let lr = cr_srgb_dec(r); let lg = cr_srgb_dec(g); let lb = cr_srgb_dec(b);
    let x0 = %(B0)s * lr + %(B1)s * lg + %(B2)s * lb;
    let x1 = %(B3)s * lr + %(B4)s * lg + %(B5)s * lb;
    let x2 = %(B6)s * lr + %(B7)s * lg + %(B8)s * lb;
    return vec3<f32>(cr_cct_enc(max(x0, 0.0f)), cr_cct_enc(max(x1, 0.0f)), cr_cct_enc(max(x2, 0.0f)));
}
"""

# u32 arithmetic wraps in WGSL as in C; bitcast<u32> of a negative cell index is C's (unsigned int) cast.
GRAIN_HELPERS = r"""
fn cr_hash(h0: u32, k: u32) -> u32 {
    var h = h0 ^ k; h = h ^ (h >> 16u); h = h * 0x7FEB352Du; h = h ^ (h >> 15u); h = h * 0x846CA68Bu; h = h ^ (h >> 16u);
    return h;
}
fn cr_cell_noise(ix: i32, iy: i32, frame: i32, seed: i32, ch: i32) -> f32 {
    var h = 0x9E3779B9u;
    h = cr_hash(h, bitcast<u32>(ix)); h = cr_hash(h, bitcast<u32>(iy)); h = cr_hash(h, bitcast<u32>(frame));
    h = cr_hash(h, bitcast<u32>(seed)); h = cr_hash(h, bitcast<u32>(ch));
    var s = 0.0f;
    for (var j = 0u; j < 4u; j++) {
        h = cr_hash(0x9E3779B9u, h); h = cr_hash(h, j);
        s += f32(h >> 8u) / 16777216.0f;
    }
    return (s - 2.0f) * 1.7320508075688772f;
}
fn cr_grain(gx: f32, gy: f32, frame: i32, seed: i32, ch: i32) -> f32 {
    let x0 = floor(gx); let y0 = floor(gy);
    var fx = gx - x0; var fy = gy - y0;
    fx = fx * fx * (3.0f - 2.0f * fx); fy = fy * fy * (3.0f - 2.0f * fy);
    let ix = i32(x0); let iy = i32(y0);
    let n00 = cr_cell_noise(ix, iy, frame, seed, ch); let n10 = cr_cell_noise(ix + 1, iy, frame, seed, ch);
    let n01 = cr_cell_noise(ix, iy + 1, frame, seed, ch); let n11 = cr_cell_noise(ix + 1, iy + 1, frame, seed, ch);
    return (n00 * (1.0f - fx) + n10 * fx) * (1.0f - fy) + (n01 * (1.0f - fx) + n11 * fx) * fy;
}
"""

TYPES = {"float": "f32", "float3": "vec3<f32>", "int": "i32"}


class _WGen(_Gen):
    target, short, key, Error = "WGSL", "WGSL", "wgsl", WGSLError
    texts = (PRELUDE, DISPLAY_HELPERS, GRAIN_HELPERS)

    @staticmethod
    def reasons(node):
        return wgsl_node_reasons(node)

    def decl(self, *pairs, ty="float"):
        return " ".join(f"var {n}: {TYPES[ty]}" + ("" if e is None else f" = {e}") + ";" for n, e in pairs)

    def flt(self, x):
        return f"f32({x[1:-1] if x.startswith('(') else x})"

    def sel(self, cond, a, b):
        return f"select({b}, {a}, {cond})"

    def if_(self, cond, stmt):
        return f"if ({cond}) {{ {stmt} }}"

    def fn_head(self, ret, name, params):
        return f"fn {name}({', '.join(f'{n}: {TYPES[t]}' for t, n in params)}) -> {TYPES[ret]} {{"

    def header(self, title):
        return super().header(title)[:2] + [
            "// WGSL compute shader. In: ACEScct (AP1) code values. Out: ACEScct code values (apply the program's",
            "// output transform after it). Keyframed values are evaluated at frame " + str(int(self.frame)) + ".",
            "// Bindings: 0 = input array<f32> (r, g, b interleaved, row-major), 1 = output (same layout),",
            f"// 2 = uniform vec2<u32> (width, height). Dispatch ceil(width / {WORKGROUP[0]}) x ceil(height / "
            f"{WORKGROUP[1]}) workgroups."]

    def entry(self, body, out, rw):
        wx, wy = WORKGROUP
        return ("@group(0) @binding(0) var<storage, read> cr_src: array<f32>;\n"
                "@group(0) @binding(1) var<storage, read_write> cr_dst: array<f32>;\n"
                "@group(0) @binding(2) var<uniform> cr_size: vec2<u32>;\n\n"
                f"@compute @workgroup_size({wx}, {wy})\n"
                "fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
                "    if (gid.x >= cr_size.x || gid.y >= cr_size.y) { return; }\n"
                "    let p_Width = i32(cr_size.x); let p_Height = i32(cr_size.y);\n"
                "    let p_X = i32(gid.x); let p_Y = i32(gid.y);\n"
                "    let i = 3u * (gid.y * cr_size.x + gid.x);\n"
                f"    let p_scale = f32(p_Width) / {f(rw)};\n"
                "    let c0 = vec3<f32>(cr_src[i], cr_src[i + 1u], cr_src[i + 2u]);\n"
                + "".join("    " + line + "\n" for line in body)
                + f"    let res = {out};\n"
                "    cr_dst[i] = res.x; cr_dst[i + 1u] = res.y; cr_dst[i + 2u] = res.z;\n}")


def compile_graph(prog, graph, frame=0, title=None, reference_width=None):
    return _WGen(prog, frame).program(graph, title, reference_width)


def compile_wgsl(prog, frame=0, split=False):
    """Checked Program -> list of parts {"name", "wgsl" (text or None), "graph", "nodes", "note"}, as compile_dctl:
    one part unless split=True cut a top-level serial graph around nodes WGSL cannot express (those come back with
    wgsl None: run them in Cierra's engine or the host's own tools)."""
    try:
        return compile_parts(_WGen, prog, frame, split)
    except DCTLError as e:  # a non-finite constant from the shared literal writer
        raise WGSLError(str(e)) from None
