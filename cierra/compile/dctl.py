"""Cierra -> DCTL (the colour transform language of a widely used grading application; plain C-like code with a
`transform` entry point per pixel).

The DCTL works on ACEScct code values in and out: put it on a node of an ACEScct timeline (an ACES colour-managed
project) and let the project's ACES 2.0 output transform render it; the program's own `output` and `photo_graph`
are not part of the DCTL. Everything is float32, constants are baked in, keyframed values are evaluated at one frame
(DCTL has no time input).

v0 coverage: every pointwise op and mask, plus position ops (vignette, grain) and window masks. Spatial ops (glow,
halation, diffusion, blur, sharpen, local_contrast, highlights_shadows), blurred mattes, depth / segment masks and
photo_to_scene are refused; with split=True a top-level serial graph is cut into several DCTLs around them.
"""

import math

from .. import ast as A
from ..engine import NEUTRAL_C, curve_tangents, hue_span, resolve, resolve_args
from ..format import program_hash
from ..ops import GREY, MASKS, OPS
from ..parser import CONTROL
from ..targets import dctl_node_reasons


class DCTLError(ValueError):
    pass


def one_line(text):
    """Text for a // comment: control characters and line breaks become spaces. The checker already refuses them in
    strings; this also holds for programs built in code."""
    return " ".join(CONTROL.sub(" ", str(text)).split())


def f(v):
    """C float literal."""
    v = float(v)
    if not math.isfinite(v):
        raise DCTLError(f"non-finite constant {v}")
    s = f"{v:.9g}"  # 9 significant digits round-trip every float32
    if "e" not in s and "." not in s:
        s += ".0"
    return s + "f"


PRELUDE = r"""
__DEVICE__ float cr_luma(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }
__DEVICE__ float cr_smoothstep(float e0, float e1, float x) {
    float t = _clampf((x - e0) / _fmaxf(e1 - e0, 1e-9f), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
__DEVICE__ float cr_band(float x, float lo, float hi, float s) {
    return cr_smoothstep(lo - s, lo + s, x) * (1.0f - cr_smoothstep(hi - s, hi + s, x));
}
__DEVICE__ float cr_mod360(float a) { return a - 360.0f * _floorf(a / 360.0f); }
__DEVICE__ float cr_hue(float cb, float cr) { return cr_mod360(_atan2f(cr, cb) * 57.29577951308232f); }
__DEVICE__ float cr_hue_w(float hue, float h0, float width) {
    float d = cr_mod360(hue - h0 + 180.0f) - 180.0f;
    return 0.5f + 0.5f * _cosf(3.14159265358979f * _clampf(d / width, -1.0f, 1.0f));
}
__DEVICE__ float cr_herm(float x, float x0, float x1, float y0, float y1, float m0, float m1) {
    float h = x1 - x0, t = (x - x0) / h, t2 = t * t, t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * y0 + (t3 - 2.0f * t2 + t) * h * m0 + (-2.0f * t3 + 3.0f * t2) * y1
           + (t3 - t2) * h * m1;
}
__DEVICE__ float cr_cct_dec(float y) {
    return y <= 0.155251141552511f ? (y - 0.0729055341958355f) / 10.5402377416545f : _exp2f(y * 17.52f - 9.72f);
}
__DEVICE__ float cr_cct_enc(float x) {
    return x <= 0.0078125f ? 10.5402377416545f * x + 0.0729055341958355f : (_log2f(_fmaxf(x, 1e-12f)) + 9.72f) / 17.52f;
}
"""

DISPLAY_HELPERS = r"""
__DEVICE__ float cr_tanh(float x) { float e = _expf(2.0f * _clampf(x, -15.0f, 15.0f)); return (e - 1.0f) / (e + 1.0f); }
__DEVICE__ float cr_srgb_enc(float v) { return v <= 0.0031308f ? 12.92f * v : 1.055f * _powf(_fmaxf(v, 0.0f), 1.0f / 2.4f) - 0.055f; }
__DEVICE__ float cr_srgb_dec(float v) { return v <= 0.04045f ? v / 12.92f : _powf((v + 0.055f) / 1.055f, 2.4f); }
__DEVICE__ float3 cr_display_encode(float r, float g, float b) {
    float lr = cr_cct_dec(r), lg = cr_cct_dec(g), lb = cr_cct_dec(b);
    float x0 = %(A0)s * lr + %(A1)s * lg + %(A2)s * lb;
    float x1 = %(A3)s * lr + %(A4)s * lg + %(A5)s * lb;
    float x2 = %(A6)s * lr + %(A7)s * lg + %(A8)s * lb;
    float y = _clampf(0.2126f * x0 + 0.7152f * x1 + 0.0722f * x2, 0.0f, 1.0f);
    float peak = _fmaxf(x0, _fmaxf(x1, x2));
    float target = peak > 0.9f ? 0.9f + 0.1f * cr_tanh((peak - 0.9f) / 0.1f) : peak;
    float t = _clampf((target - y) / _fmaxf(peak - y, 1e-9f), 0.0f, 1.0f);
    if (peak > 0.9f) { x0 = y + (x0 - y) * t; x1 = y + (x1 - y) * t; x2 = y + (x2 - y) * t; }
    float low = _fminf(x0, _fminf(x1, x2));
    float t2 = _clampf(y / _fmaxf(y - low, 1e-9f), 0.0f, 1.0f);
    if (low < 0.0f) { x0 = y + (x0 - y) * t2; x1 = y + (x1 - y) * t2; x2 = y + (x2 - y) * t2; }
    return make_float3(cr_srgb_enc(_clampf(x0, 0.0f, 1.0f)), cr_srgb_enc(_clampf(x1, 0.0f, 1.0f)),
                       cr_srgb_enc(_clampf(x2, 0.0f, 1.0f)));
}
__DEVICE__ float3 cr_from_display(float r, float g, float b) {
    float lr = cr_srgb_dec(r), lg = cr_srgb_dec(g), lb = cr_srgb_dec(b);
    float x0 = %(B0)s * lr + %(B1)s * lg + %(B2)s * lb;
    float x1 = %(B3)s * lr + %(B4)s * lg + %(B5)s * lb;
    float x2 = %(B6)s * lr + %(B7)s * lg + %(B8)s * lb;
    return make_float3(cr_cct_enc(_fmaxf(x0, 0.0f)), cr_cct_enc(_fmaxf(x1, 0.0f)), cr_cct_enc(_fmaxf(x2, 0.0f)));
}
"""

GRAIN_HELPERS = r"""
__DEVICE__ unsigned int cr_hash(unsigned int h, unsigned int k) {
    h = h ^ k; h = h ^ (h >> 16); h = h * 0x7FEB352Du; h = h ^ (h >> 15); h = h * 0x846CA68Bu; h = h ^ (h >> 16);
    return h;
}
__DEVICE__ float cr_cell_noise(int ix, int iy, int frame, int seed, int ch) {
    unsigned int h = 0x9E3779B9u;
    h = cr_hash(h, (unsigned int)ix); h = cr_hash(h, (unsigned int)iy); h = cr_hash(h, (unsigned int)frame);
    h = cr_hash(h, (unsigned int)seed); h = cr_hash(h, (unsigned int)ch);
    float s = 0.0f;
    for (int j = 0; j < 4; j++) {
        h = cr_hash(0x9E3779B9u, h); h = cr_hash(h, (unsigned int)j);
        s += (float)(h >> 8) / 16777216.0f;
    }
    return (s - 2.0f) * 1.7320508075688772f;
}
__DEVICE__ float cr_grain(float gx, float gy, int frame, int seed, int ch) {
    float x0 = _floorf(gx), y0 = _floorf(gy);
    float fx = gx - x0, fy = gy - y0;
    fx = fx * fx * (3.0f - 2.0f * fx); fy = fy * fy * (3.0f - 2.0f * fy);
    int ix = (int)x0, iy = (int)y0;
    float n00 = cr_cell_noise(ix, iy, frame, seed, ch), n10 = cr_cell_noise(ix + 1, iy, frame, seed, ch);
    float n01 = cr_cell_noise(ix, iy + 1, frame, seed, ch), n11 = cr_cell_noise(ix + 1, iy + 1, frame, seed, ch);
    return (n00 * (1.0f - fx) + n10 * fx) * (1.0f - fy) + (n01 * (1.0f - fx) + n11 * fx) * fy;
}
"""


class _Gen:
    """Shared by every C-like target: the op / mask / graph code is written once against the hooks below (C / DCTL
    syntax here; cierra/compile/wgsl.py overrides them). Built-ins keep their DCTL names (_fmaxf, make_float3, ...);
    a target without them defines them in its prelude."""
    target, short, key, Error = "DCTL v0", "DCTL", "dctl", DCTLError
    texts = (PRELUDE, DISPLAY_HELPERS, GRAIN_HELPERS)

    def __init__(self, prog, frame):
        self.prog, self.frame = prog, frame
        self.funcs, self.need_display, self.need_grain = [], False, False

    # ---------------------------------------------------------- dialect hooks (C)
    @staticmethod
    def reasons(node):
        return dctl_node_reasons(node)

    def decl(self, *pairs, ty="float"):
        """One declaration statement: pairs of (name, initial expression or None)."""
        return f"{ty} " + ", ".join(n if e is None else f"{n} = {e}" for n, e in pairs) + ";"

    def flt(self, x):
        """int -> float; x is a name or a parenthesised expression."""
        return f"(float){x}"

    def sel(self, cond, a, b):
        return f"{cond} ? {a} : {b}"

    def if_(self, cond, stmt):
        return f"if ({cond}) {stmt}"

    def fn_head(self, ret, name, params):
        return f"__DEVICE__ {ret} {name}({', '.join(f'{t} {n}' for t, n in params)}) {{"

    def unit(self, u, n):
        """Pixel position u (x / y) scaled to 0..1 over the frame's width / height n (W / H)."""
        return f"{self.flt('p_' + u.upper())} / {self.flt(f'(p_{n} - 1)')}"

    # ---------------------------------------------------------- curves
    def curve_fn(self, name, points):
        xs, ys, ms = curve_tangents(points)
        n = len(xs)
        lines = [self.fn_head("float", name, [("float", "x")]),
                 "    " + self.if_(f"x < {f(xs[0])}", f"return {f(ys[0])} + (x - {f(xs[0])}) * {f(ms[0])};"),
                 "    " + self.if_(f"x > {f(xs[-1])}", f"return {f(ys[-1])} + (x - {f(xs[-1])}) * {f(ms[-1])};")]
        for k in range(n - 1):
            seg = (f"cr_herm(x, {f(xs[k])}, {f(xs[k + 1])}, {f(ys[k])}, {f(ys[k + 1])}, {f(ms[k])}, "
                   f"{f(ms[k + 1])})")
            ret = f"return {seg};"
            lines.append("    " + (self.if_(f"x <= {f(xs[k + 1])}", ret) if k < n - 2 else ret))
        lines.append("}")
        self.funcs.append("\n".join(lines))
        return name

    def hue_curve_expr(self, var, points, neutral, out):
        code = [self.decl((f"{out}_s", "0.0f"), (f"{out}_w", "0.0f"), (f"{out}_k", None))]
        for p in points:
            width = p[2] if len(p) > 2 else 60.0
            code.append(f"{out}_k = cr_hue_w({var}, {f(p[0])}, {f(width)}); {out}_s += {out}_k * {f(p[1] - neutral)}; "
                        f"{out}_w += {out}_k;")
        code.append(self.decl((out, f"{f(neutral)} + {out}_s / _fmaxf({out}_w, 1.0f)")))
        return code

    # ---------------------------------------------------------- ops (in: r, g, b; out: o0, o1, o2)
    def op(self, nid, name, a):
        opp = [self.decl(("y", "cr_luma(r, g, b)")), self.decl(("cb", "(b - y) / 1.8556f"), ("cr", "(r - y) / 1.5748f"))]
        back = ["o0 = y2 + 1.5748f * cr2; o2 = y2 + 1.8556f * cb2; o1 = (y2 - 0.2126f * o0 - 0.0722f * o2) / 0.7152f;"]
        if name in ("exposure", "white_balance"):  # a linear-light gain (engine._linear_gain)
            gains = ([a["stops"]] * 3 if name == "exposure"
                     else [a["temperature"], -a["tint"], -a["temperature"]])
            if not any(gains):
                return ["o0 = r; o1 = g; o2 = b;"]
            return [f"o{i} = cr_cct_enc(cr_cct_dec({c}) * {f(2.0 ** k)});" for i, (c, k) in enumerate(zip("rgb", gains))]
        if name == "printer_lights":
            s = 0.025 / (17.52 * math.log10(2))
            return [f"o0 = r + {f(a['red'] * s)}; o1 = g + {f(a['green'] * s)}; o2 = b + {f(a['blue'] * s)};"]
        if name == "wheels":
            code = [self.decl(("y", "_clampf(cr_luma(r, g, b), 0.0f, 1.0f)")),
                    self.decl(("sh", f"1.0f - cr_smoothstep(0.1f, {f(GREY + 0.05)}, y)")),
                    self.decl(("d", f"(y - {f(GREY)}) / 0.18f")) + " " + self.decl(("mi", "_expf(-d * d)")),
                    self.decl(("hi", f"cr_smoothstep({f(GREY - 0.05)}, 0.75f, y)"))]
            for i, ch in enumerate("rgb"):
                code.append(f"o{i} = {ch} + {f(a['lift'][i] + a['master_lift'])} * sh + "
                            f"{f(a['gamma'][i] + a['master_gamma'])} * mi + {f(a['gain'][i] + a['master_gain'])} * hi + "
                            f"{f(a['offset'][i] + a['master_offset'])};")
            return code
        if name == "cdl":
            code = []
            for i, ch in enumerate("rgb"):
                code.append(self.decl((f"v{i}", f"{ch} * {f(a['slope'][i])} + {f(a['offset'][i])}")) + " "
                            + self.if_(f"v{i} >= 0.0f", f"v{i} = _powf(v{i}, {f(a['power'][i])});"))
            code.append(self.decl(("y", "cr_luma(v0, v1, v2)")))
            code += [f"o{i} = y + {f(a['sat'])} * (v{i} - y);" for i in range(3)]
            return code
        if name == "contrast":
            return [f"o{i} = ({ch} - {f(a['pivot'])}) * {f(a['amount'])} + {f(a['pivot'])};" for i, ch in enumerate("rgb")]
        if name == "saturation":
            code = opp + [self.decl(("k", f(a['amount'])))]
            if a["protect_skin"]:
                code += [self.decl(("hue", "cr_hue(cb, cr)")),
                         self.decl(("skin", "1.0f - cr_smoothstep(15.0f, 40.0f, _fabs(cr_mod360(hue - 141.0f + 180.0f) "
                                            "- 180.0f))")),
                         f"k = k + (1.0f - k) * skin * {f(a['protect_skin'])};"]
            return code + [self._scaled("k")] + back
        if name == "curve":
            fn = self.curve_fn(f"curve_{nid}", a["points"])
            if a["space"] == "display":
                self.need_display = True
                pre = [self.decl(("dd", "cr_display_encode(r, g, b)"), ty="float3"),
                       self.decl(*((f"d{i}", f"_clampf(dd.{c}, 0.0f, 1.0f)") for i, c in enumerate("xyz")))]
                body = self._curve_body(fn, a["channel"], "d0", "d1", "d2", "e")
                return pre + body + [self.decl(("ww", "cr_from_display(_clampf(e0, 0.0f, 1.0f), _clampf(e1, 0.0f, 1.0f), "
                                                      "_clampf(e2, 0.0f, 1.0f))"), ty="float3"),
                                     "o0 = ww.x; o1 = ww.y; o2 = ww.z;"]
            return self._curve_body(fn, a["channel"], "r", "g", "b", "o", declare=False)
        if name in ("hue_vs_hue", "hue_vs_sat"):
            neutral = 0.0 if name == "hue_vs_hue" else 1.0
            code = opp + [self.decl(("hue", "cr_hue(cb, cr)"))] + self.hue_curve_expr("hue", a["points"], neutral, "hv")
            if name == "hue_vs_hue":
                code += [self.decl(("sa", "hv * 0.017453292519943295f"), ("cs", "_cosf(sa)"), ("sn", "_sinf(sa)")),
                         self.decl(("y2", "y"), ("cb2", "cb * cs - cr * sn"), ("cr2", "cb * sn + cr * cs"))]
            else:
                code += [self._scaled("hv")]
            return code + back
        if name == "hue_vs_lum":
            return (opp + [self.decl(("hue", "cr_hue(cb, cr)"), ("c", "_sqrtf(cb * cb + cr * cr)"))]
                    + self.hue_curve_expr("hue", a["points"], 0.0, "hv")
                    + [self.decl(("y2", "y + hv * cr_smoothstep(0.0f, 0.06f, c)"), ("cb2", "cb"), ("cr2", "cr"))] + back)
        if name == "lum_vs_sat":
            fn = self.curve_fn(f"curve_{nid}", a["points"])
            return opp + [self.decl(("k", f"{fn}(y)")), self._scaled("k")] + back
        if name == "sat_vs_sat":
            fn = self.curve_fn(f"curve_{nid}", a["points"])
            return opp + [self.decl(("c", "_sqrtf(cb * cb + cr * cr)")), self.decl(("k", f"{fn}(c) / _fmaxf(c, 1e-6f)")),
                          self._scaled("k")] + back
        if name == "sat_vs_lum":
            fn = self.curve_fn(f"curve_{nid}", a["points"])
            return opp + [self.decl(("c", "_sqrtf(cb * cb + cr * cr)")),
                          self.decl(("y2", f"y + {fn}(c)"), ("cb2", "cb"), ("cr2", "cr"))] + back
        if name == "rgb_mixer":
            code = [f"o{i} = {f(m[0])} * r + {f(m[1])} * g + {f(m[2])} * b;"
                    for i, m in enumerate((a["red"], a["green"], a["blue"]))]
            if a["preserve_luma"]:
                code += [self.decl(("dy", "cr_luma(r, g, b) - cr_luma(o0, o1, o2)")), "o0 += dy; o1 += dy; o2 += dy;"]
            return code
        if name == "density":
            return opp + [self.decl(("c", "_sqrtf(cb * cb + cr * cr)")),
                          self.decl(("y2", f"y - {f(a['amount'])} * c"), ("cb2", "cb"), ("cr2", "cr"))] + back
        if name == "hue_rotate":
            t = math.radians(a["degrees"])
            cs, sn = f(math.cos(t)), f(math.sin(t))
            return opp + [self.decl(("y2", "y"), ("cb2", f"cb * {cs} - cr * {sn}"), ("cr2", f"cb * {sn} + cr * {cs}"))] + back
        if name == "vibrance":
            return opp + [self.decl(("c", "_sqrtf(cb * cb + cr * cr)")),
                          self.decl(("k", f"1.0f + {f(a['amount'])} * (1.0f - cr_smoothstep(0.0f, 0.12f, c))")),
                          self._scaled("k")] + back
        if name == "split_tone":
            pivot = f(GREY + a["balance"])
            hs, hh = math.radians(a["shadow_hue"]), math.radians(a["highlight_hue"])
            return opp + [self.decl(("sh", f"(1.0f - cr_smoothstep(0.15f, {pivot}, y)) * {f(a['shadow_amount'])}"),
                                    ("hi", f"cr_smoothstep({pivot}, 0.75f, y) * {f(a['highlight_amount'])}")),
                          self.decl(("y2", "y"), ("cb2", f"cb + sh * {f(math.cos(hs))} + hi * {f(math.cos(hh))}"),
                                    ("cr2", f"cr + sh * {f(math.sin(hs))} + hi * {f(math.sin(hh))}"))] + back
        if name == "gamut_compress":
            code = [self.decl(("l0", "cr_cct_dec(r)"), ("l1", "cr_cct_dec(g)"), ("l2", "cr_cct_dec(b)")),
                    self.decl(("ach", "_fmaxf(l0, _fmaxf(l1, l2))")), self.decl(("aa", "_fabs(ach)"))]
            p = a["power"]
            for i in range(3):
                t, lim = a["threshold"][i], a["limit"][i]
                s = (lim - t) / (((1 - t) / (lim - t)) ** -p - 1) ** (1 / p)
                code += [self.decl((f"d{i}", self.sel("aa > 0.0f", f"(ach - l{i}) / aa", "0.0f"))),
                         self.decl((f"x{i}", f"_fmaxf((d{i} - {f(t)}) / {f(s)}, 0.0f)")),
                         self.decl((f"q{i}", self.sel(f"d{i} < {f(t)}", f"d{i}", f"{f(t)} + {f(s)} * x{i} / "
                                                      f"_powf(1.0f + _powf(x{i}, {f(p)}), {f(1 / p)})"))),
                         f"o{i} = cr_cct_enc(ach - q{i} * aa);"]
            return code
        if name == "vignette":
            e = 1.0 - a["roundness"]
            return [self.decl((u + u, f"({self.sel(f'p_{n} > 1', '2.0f * ' + self.unit(u, n), '0.0f')}) - 1.0f"))
                    for u, n in (("x", "W"), ("y", "H"))] + [
                    self.decl(("ax", f"xx * _powf({self.flt('p_W')} / {self.flt('p_H')}, {f(e)})")),
                    self.decl(("rr", "_sqrtf(ax * ax + yy * yy) / 1.4142135623730951f")),
                    self.decl(("dv", f"{f(a['amount'] * 4 / 17.52)} * cr_smoothstep({f(1 - a['softness'])}, 1.2f, rr)")),
                    "o0 = r - dv; o1 = g - dv; o2 = b - dv;"]
        if name == "grain":
            self.need_grain = True
            code = [self.decl(("gs", f"_fmaxf({f(a['size_px'])} * p_scale, 0.001f)")),  # engine: max(size, 1e-3)
                    self.decl(("gx", f"{self.flt('p_X')} / gs"), ("gy", f"{self.flt('p_Y')} / gs")),
                    self.decl(("mono", f"cr_grain(gx, gy, {int(self.frame)}, {int(a['seed'])}, 3)")),
                    self.decl(("yl", "_clampf(cr_luma(r, g, b), 0.0f, 1.0f)"), ("resp", "4.0f * yl * (1.0f - yl)"))]
            for i, ch in enumerate("rgb"):
                code.append(f"o{i} = {ch} + {f(a['amount'] * 0.02)} * (mono + {f(a['chroma'])} * "
                            f"(cr_grain(gx, gy, {int(self.frame)}, {int(a['seed'])}, {i}) - mono)) * resp;")
            return code
        raise self.Error(f"{name}() cannot be compiled to {self.target}")

    def _scaled(self, k):
        return self.decl(("y2", "y"), ("cb2", f"cb * {k}"), ("cr2", f"cr * {k}"))

    def _curve_body(self, fn, channel, r, g, b, out, declare=True):
        def put(k, e):
            return self.decl((f"{out}{k}", e)) if declare else f"{out}{k} = {e};"
        if channel == "luma":
            return [self.decl(("yc", f"cr_luma({r}, {g}, {b})")) + " " + self.decl(("dy", f"{fn}(yc) - yc")),
                    " ".join(put(k, f"{c} + dy") for k, c in enumerate((r, g, b)))]
        if channel == "rgb":
            return [" ".join(put(k, f"{fn}({c})") for k, c in enumerate((r, g, b)))]
        i = {"red": 0, "green": 1, "blue": 2}[channel]
        src = [r, g, b]
        return [put(k, fn + "(" + src[k] + ")" if k == i else src[k]) for k in range(3)]

    # ---------------------------------------------------------- masks (expression in r, g, b)
    def mask(self, m, nid, counter):
        if isinstance(m, A.MaskNot):
            e, code = self.mask(m.m, nid, counter)
            return f"(1.0f - {e})", code
        if isinstance(m, A.MaskBin):
            ea, ca = self.mask(m.a, nid, counter)
            eb, cb = self.mask(m.b, nid, counter)
            e = {"and": f"({ea} * {eb})", "+": f"_fminf({ea} + {eb}, 1.0f)", "-": f"_fmaxf({ea} - {eb}, 0.0f)"}
            return e.get(m.op, f"_fmaxf({ea}, {eb})"), ca + cb
        if isinstance(m, A.MaskRef):  # a carried node key is the kv_ parameter; a key statement is made here
            e, code = (f"kv_{m.name}", []) if m.carried else self.mask(m.target, nid, counter)
            k = resolve_args({"gain": m.gain, "offset": m.offset}, self.frame)
            if k["gain"] == 1.0 and k["offset"] == 0.0:
                return e, code
            return f"_clampf({f(k['gain'])} * {e} + {f(k['offset'])}, 0.0f, 1.0f)", code
        v, code = self._mask_call(m, nid, counter)
        a = resolve_args(m.call.args, self.frame)
        lo, hi = a.get("clip_black", 0.0), a.get("clip_white", 1.0)
        if lo or hi != 1.0:  # matte clip: black..white stretched to 0..1
            code = code + [f"{v} = _clampf(({v} - {f(lo)}) / {f(max(hi - lo, 1e-6))}, 0.0f, 1.0f);"]
        return v, code

    def _mask_call(self, m, nid, counter):
        a = resolve_args(m.call.args, self.frame)
        kind = m.call.name
        spec = MASKS[kind]
        if not spec.dctl or not spec.pointwise and not spec.position:
            raise self.Error(f"node {nid}: mask {kind}() cannot be compiled to {self.target}")
        if a.get("blur_px") or a.get("feather_px"):
            raise self.Error(f"node {nid}: blurred or feathered mattes (blur_px / feather_px) are spatial; "
                             f"{self.target} cannot blur")
        counter[0] += 1
        v = f"m{counter[0]}"
        if kind == "skin":
            half = a["width"] / 2
            a = {"hue": A.Range(141 - half, 141 + half), "sat": A.Range(0.006, 0.09), "lum": A.Range(0.15, 0.7),
                 "soft": a["soft"]}
            kind = "qualifier"
        if kind == "qualifier":
            code = [self.decl((v, "1.0f")), "{",
                    "  " + self.decl(("qy", "cr_luma(r, g, b)"), ("qcb", "(b - qy) / 1.8556f"),
                                     ("qcr", "(r - qy) / 1.5748f")),
                    "  " + self.decl(("qc", "_sqrtf(qcb * qcb + qcr * qcr)")),
                    "  " + self.decl(("qh", self.sel(f"qc < {f(NEUTRAL_C)}", "0.0f", "cr_hue(qcb, qcr)")))]
            s = a["soft"]
            span = hue_span(a.get("hue"))
            if span is not None and span < 360:
                lo = a["hue"].lo
                centre, half = (lo + span / 2) % 360, span / 2
                code.append(f"  {v} *= 1.0f - cr_smoothstep({f(half)}, {f(half + s * 360)}, "
                            f"_fabs(cr_mod360(qh - {f(centre)} + 180.0f) - 180.0f));")
            if a.get("sat") is not None:
                code.append(f"  {v} *= cr_band(qc, {f(a['sat'].lo)}, {f(a['sat'].hi)}, {f(s * 0.2)});")
            if a.get("lum") is not None:
                code.append(f"  {v} *= cr_band(qy, {f(a['lum'].lo)}, {f(a['lum'].hi)}, {f(s)});")
            return v, code + ["}"]
        if kind == "luma":
            band = f"cr_band(cr_luma(r, g, b), {f(a['range'].lo)}, {f(a['range'].hi)}, {f(a['soft'])})"
            return v, [self.decl((v, band))]
        if kind == "rgb_key":
            code = [self.decl((v, "1.0f"))]
            for key, ch in (("red", "r"), ("green", "g"), ("blue", "b")):
                if a.get(key) is not None:
                    code.append(f"{v} *= cr_band({ch}, {f(a[key].lo)}, {f(a[key].hi)}, {f(a['soft'])});")
            return v, code
        if kind == "polygon":
            wk = self.sel("p_W > 1 && p_H > 1", f"{self.flt('(p_H - 1)')} / {self.flt('(p_W - 1)')}", "1.0f")
            code = [self.decl((v, None)), "{"] + [
                "  " + self.decl((f"w{u}", self.sel(f"p_{n} > 1", self.unit(u, n), "0.0f")))
                for u, n in (("x", "W"), ("y", "H"))] + [
                "  " + self.decl(("wk", wk)), "  " + self.decl(("py", "wy * wk"), ("ins", "0.0f"), ("d2", "1e30f"))]
            pts = a["points"]
            for i in range(len(pts)):
                (xi, yi), (xj, yj) = pts[i][:2], pts[i - 1][:2]
                ex, ey = xj - xi, yj - yi
                code += ["  " + self.decl((f"yi{i}", f"{f(yi)} * wk"), (f"ey{i}", f"{f(ey)} * wk")),
                         "  " + self.if_(f"(yi{i} > py) != ({f(yj)} * wk > py)",
                                         self.if_(f"wx < {f(ex)} * (py - yi{i}) / ey{i} + {f(xi)}", "ins = 1.0f - ins;")),
                         "  " + self.decl((f"t{i}", f"_clampf(((wx - {f(xi)}) * {f(ex)} + (py - yi{i}) * ey{i}) / "
                                                   f"_fmaxf({f(ex)} * {f(ex)} + ey{i} * ey{i}, 1e-12f), 0.0f, 1.0f)"),
                                          (f"dx{i}", f"wx - {f(xi)} - t{i} * {f(ex)}"),
                                          (f"dy{i}", f"py - yi{i} - t{i} * ey{i}")),
                         f"  d2 = _fminf(d2, dx{i} * dx{i} + dy{i} * dy{i});"]
            s = a["soft"]
            sign = self.sel("ins > 0.5f", "1.0f", "-1.0f")
            code.append(f"  {v} = cr_smoothstep({f(-s)}, {f(s)}, _sqrtf(d2) * ({sign}));")
            return v, code + ["}"]
        if kind in ("graduated", "window", "box") and (kind in ("graduated", "box") or a["angle"]):
            # rotated frame (cierra.engine._rotated): offsets in frame widths, y up, rotated by the angle
            t = math.radians(a["angle"])
            (cx, cy) = a["centre"]
            code = [self.decl((v, None)), "{"] + [
                "  " + self.decl((f"w{u}", self.sel(f"p_{n} > 1", self.unit(u, n), "0.0f")))
                for u, n in (("x", "W"), ("y", "H"))] + [
                "  " + self.decl(("wk", self.sel("p_W > 1 && p_H > 1",
                                                 f"{self.flt('(p_H - 1)')} / {self.flt('(p_W - 1)')}",
                                                 "1.0f"))),
                "  " + self.decl(("wdx", f"wx - {f(cx)}"), ("wup", f"({f(cy)} - wy) * wk")),
                "  " + self.decl(("wu", f"wdx * {f(math.cos(t))} + wup * {f(math.sin(t))}"),
                                 ("wv", f"wup * {f(math.cos(t))} - wdx * {f(math.sin(t))}"))]
            if kind == "graduated":
                fb, fa = a["falloff"]
                code.append(f"  {v} = cr_smoothstep({f(-fb)}, {f(fa)}, wu);")
            elif kind == "box":
                (sx, sy), s = a["size"], a["soft"]
                n = 2 + 18 * (1 - a["roundness"])
                code += ["  " + self.decl(("bx", f"_fabs(wu / {f(sx)})"), ("by", f"_fabs(wv / (wk * {f(sy)}))")),
                         f"  {v} = 1.0f - cr_smoothstep({f(1 - s)}, {f(1 + s)}, "
                         f"_powf(_powf(bx, {f(n)}) + _powf(by, {f(n)}), {f(1 / n)}));"]
            else:
                (rx, ry), s = a["radius"], a["soft"]
                code += ["  " + self.decl(("ex", f"wu / {f(rx)}"), ("ey", f"wv / (wk * {f(ry)})")),
                         f"  {v} = 1.0f - cr_smoothstep({f(1 - s)}, {f(1 + s)}, _sqrtf(ex * ex + ey * ey));"]
            return v, code + ["}"]
        if kind == "window":
            (cx, cy), (rx, ry) = a["centre"], a["radius"]
            return v, [self.decl((v, None)), "{"] + [
                       "  " + self.decl((f"w{u}", self.sel(f"p_{n} > 1", self.unit(u, n), "0.0f")))
                       for u, n in (("x", "W"), ("y", "H"))] + [
                       "  " + self.decl(("ex", f"(wx - {f(cx)}) / {f(rx)}"), ("ey", f"(wy - {f(cy)}) / {f(ry)}")),
                       f"  {v} = 1.0f - cr_smoothstep({f(1 - a['soft'])}, {f(1 + a['soft'])}, _sqrtf(ex * ex + ey * ey));",
                       "}"]
        raise self.Error(f"node {nid}: mask {kind}() cannot be compiled to {self.target}")

    def node_fn(self, node):
        reasons = self.reasons(node)
        if reasons:
            raise self.Error(f"cannot compile to {self.target}: " + "; ".join(reasons))
        a = resolve_args(node.call.args, self.frame)
        body = self.op(node.id, node.call.name, a)
        mix = resolve(node.mix, self.frame)
        lines = [self.fn_head("float3", f"n_{node.id}", self.params(node))]
        if node.label:
            lines.insert(0, "// " + one_line(node.label))
        if not node.enabled:
            lines += ["    return make_float3(r, g, b);", "}"]
            self.funcs.append("\n".join(lines))
            return
        lines.append("    " + self.decl(("o0", None), ("o1", None), ("o2", None)))
        lines += ["    " + x for x in body]
        w = f(mix)
        if node.mask is not None:
            e, code = self.mask(node.mask, node.id, [0])
            lines += ["    " + x for x in code]
            w = f"({f(mix)} * {e})"
        lines += ["    " + self.decl(("w", w)),
                  "    return make_float3(r + (o0 - r) * w, g + (o1 - g) * w, b + (o2 - b) * w);", "}"]
        self.funcs.append("\n".join(lines))

    def params(self, node):
        """A node function's parameters: the pixel, its position, and the carried keys its mask reads."""
        from ..walk import carried_keys
        return ([("float", "r"), ("float", "g"), ("float", "b"), ("int", "p_X"), ("int", "p_Y"), ("int", "p_W"),
                 ("int", "p_H"), ("float", "p_scale")] + [("float", f"kv_{k}") for k in carried_keys(node.mask)])

    def key_fn(self, node):
        """k_<node>: the node's key on its input, for the nodes downstream that read it (key(node))."""
        e, code = self.mask(node.mask, node.id, [0])
        self.funcs.append("\n".join([self.fn_head("float", f"k_{node.id}", self.params(node))]
                                     + ["    " + x for x in code] + [f"    return {e};", "}"]))

    def call_args(self, node, vin):
        from ..walk import carried_keys
        extra = carried_keys(node.mask)
        for k in extra:
            if k not in self.kv_done:
                raise self.Error(f"node {node.id} reads key({k}), but node {k} is not in this {self.target} part: "
                                 "a carried key cannot cross a split")
        return ", ".join([f"{vin}.x, {vin}.y, {vin}.z, p_X, p_Y, p_Width, p_Height, p_scale"]
                         + [f"kv_{k}" for k in extra])

    # ---------------------------------------------------------- graph
    def graph(self, g, vin, body, counter):
        if g is None or isinstance(g, A.Pass):  # omitted graph and `pass`: identity
            return vin
        if isinstance(g, A.Ref):
            node = self.prog.nodes[g.name]
            if g.name in self.carry:  # its key, made on its input, before it runs
                body.append(self.decl((f"kv_{g.name}", f"k_{g.name}({self.call_args(node, vin)})")))
                self.kv_done.add(g.name)
            counter[0] += 1
            v = f"c{counter[0]}"
            body.append(self.decl((v, f"n_{g.name}({self.call_args(node, vin)})"), ty="float3"))
            return v
        if isinstance(g, A.Serial):
            for item in g.items:
                vin = self.graph(item, vin, body, counter)
            return vin
        if isinstance(g, A.Parallel):
            outs = [self.graph(b, vin, body, counter) for b in g.branches]
            counter[0] += 1
            v = f"c{counter[0]}"
            n = f(len(outs))
            comps = [" + ".join(f"{o}.{c}" for o in outs) for c in "xyz"]
            body.append(self.decl((v, f"make_float3(({comps[0]}) / {n}, ({comps[1]}) / {n}, ({comps[2]}) / {n})"),
                                  ty="float3"))
            return v
        if isinstance(g, A.Layer):
            b = self.graph(g.bottom, vin, body, counter)
            t = self.graph(g.top, vin, body, counter)
            counter[0] += 1
            v = f"c{counter[0]}"
            op = f(resolve(g.opacity, self.frame))
            comps = []
            for c in "xyz":
                if g.mode == "normal":
                    bl = f"{t}.{c}"
                elif g.mode == "lighten":
                    bl = f"_fmaxf({b}.{c}, {t}.{c})"
                elif g.mode == "darken":
                    bl = f"_fminf({b}.{c}, {t}.{c})"
                elif g.mode == "screen":
                    bl = f"1.0f - (1.0f - {b}.{c}) * (1.0f - {t}.{c})"
                elif g.mode == "overlay":
                    bl = "(" + self.sel(f"{b}.{c} < 0.5f", f"2.0f * {b}.{c} * {t}.{c}",
                                  f"1.0f - 2.0f * (1.0f - {b}.{c}) * (1.0f - {t}.{c})") + ")"
                elif g.mode == "subtract":
                    bl = f"cr_cct_enc(_fmaxf(cr_cct_dec({b}.{c}) - cr_cct_dec({t}.{c}), 0.0f))"
                elif g.mode == "difference":
                    bl = f"cr_cct_enc(_fabs(cr_cct_dec({b}.{c}) - cr_cct_dec({t}.{c})))"
                elif g.mode == "add":
                    bl = f"cr_cct_enc(_fmaxf(cr_cct_dec({b}.{c}) + cr_cct_dec({t}.{c}), 0.0f))"
                else:
                    bl = f"cr_cct_enc(_fmaxf(cr_cct_dec({b}.{c}) * cr_cct_dec({t}.{c}) / 0.18f, 0.0f))"
                comps.append(f"{b}.{c} + {op} * ({bl} - {b}.{c})")
            body.append(self.decl((v, f"make_float3({', '.join(comps)})"), ty="float3"))
            return v
        raise AssertionError(g)

    # ---------------------------------------------------------- whole program (the DCTL `transform` entry point)
    def header(self, title):
        return [f"// {one_line(title or self.prog.name or 'Cierra program')}",
                f"// Generated by Cierra 0.1 from program {program_hash(self.prog)}; edit the .cierra source, not this "
                "file.",
                "// In: ACEScct (AP1) code values. Out: ACEScct code values. Use on an ACEScct timeline with an ACES 2.0",
                "// output transform. Keyframed values are evaluated at frame " + str(int(self.frame)) + "."]

    def helpers(self):
        """The prelude, plus the display-curve and grain helpers when an op needs them (texts: the target's own)."""
        from ..colour import M709_AP1, MAP1_709
        prelude, display, grain = self.texts
        parts = [prelude.strip()]
        if self.need_display:
            consts = {f"A{i}": f(v) for i, v in enumerate(MAP1_709.flatten())}
            consts.update({f"B{i}": f(v) for i, v in enumerate(M709_AP1.flatten())})
            parts.append(display.strip() % consts)
        if self.need_grain:
            parts.append(grain.strip())
        return parts

    def program(self, graph, title=None, reference_width=None):
        from ..walk import carried_keys, graph_nodes
        self.carry = {k for n in self.prog.nodes.values() for k in carried_keys(n.mask)}
        self.kv_done = set()
        for node in graph_nodes(self.prog, graph):
            self.node_fn(node)
            if node.id in self.carry:
                self.key_fn(node)
        body = []
        out = self.graph(graph, "c0", body, [0])
        rw = float(reference_width or self.prog.reference_width)
        parts = ["\n".join(self.header(title))] + self.helpers() + self.funcs + [self.entry(body, out, rw)]
        return "\n\n".join(parts) + "\n"

    def entry(self, body, out, rw):
        return ("__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, "
                "float p_B) {\n"
                f"    float p_scale = (float)p_Width / {f(rw)};\n"
                "    float3 c0 = make_float3(p_R, p_G, p_B);\n"
                + "".join("    " + line + "\n" for line in body)
                + f"    return {out};\n}}")


def compile_graph(prog, graph, frame=0, title=None, reference_width=None):
    return _Gen(prog, frame).program(graph, title, reference_width)


def compile_parts(gen, prog, frame=0, split=False):
    """Checked Program -> list of parts {"name", gen.key (text or None), "graph", "nodes", "note"} for the target of
    generator class `gen` (shared by DCTL and WGSL). One part unless split=True cut a top-level serial graph around
    nodes the target cannot express; those come back as parts with text None."""
    from ..walk import graph_nodes

    def why(node):
        return "; ".join(gen.reasons(node))

    def compiled(g):
        return gen(prog, frame).program(g)

    key = gen.key
    bad = [n for n in graph_nodes(prog, prog.graph) if gen.reasons(n)]
    if not bad:
        return [{"name": "main", key: compiled(prog.graph), "graph": prog.graph,
                 "nodes": [n.id for n in graph_nodes(prog, prog.graph)], "note": "whole graph"}]
    if not split:
        raise gen.Error(f"cannot compile to {gen.target}: " + "; ".join(why(n) for n in bad)
                        + " (compile with --split / split=True to cut the graph around them)")
    items = prog.graph.items if isinstance(prog.graph, A.Serial) else [prog.graph]
    out, run = [], []

    def flush():
        if run:
            g = run[0] if len(run) == 1 else A.Serial(list(run))
            ids = [n.id for n in graph_nodes(prog, g)]
            out.append({"name": f"part{len(out) + 1}", key: compiled(g), "graph": g,
                        "nodes": ids, "note": "nodes " + ", ".join(ids)})
            run.clear()

    for item in items:
        nodes = graph_nodes(prog, item)
        if not any(gen.reasons(n) for n in nodes):
            run.append(item)
            continue
        if not isinstance(item, A.Ref):
            raise gen.Error(f"a parallel / layer branch holds a node {gen.target} cannot express: "
                            + "; ".join(why(n) for n in nodes if gen.reasons(n)))
        flush()
        out.append({"name": f"part{len(out) + 1}", key: None, "graph": item, "nodes": [item.name],
                    "note": f"not in {gen.short}: " + why(prog.nodes[item.name])})
    flush()
    return out


def compile_dctl(prog, frame=0, split=False):
    """Checked Program -> list of parts {"name", "dctl" (text or None), "graph", "nodes", "note"}. One part unless
    split=True cut a top-level serial graph around nodes DCTL v0 cannot express; those come back as parts with
    dctl None (do them with the host's own tools, or in Cierra's engine)."""
    return compile_parts(_Gen, prog, frame, split)
