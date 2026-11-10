"""DCTL wrapper (G5): a grade for a normal Rec.709 (YRGB) project, not only an ACEScct timeline.

`input` decodes the node's input to ACEScct before the graph, as the engine's `render --input` does: Rec.1886 /
sRGB display video (the engine's rec1886_photo / srgb_photo) or a camera log (the curves below; the gamut matrices are
least-squares fits to OpenColorIO's studio config, grey exact to 2e-5 cv, colours within 6e-4 cv, measured
2026-11-02). `output` encodes the result for a display through a 3D LUT written next to the DCTL and applied with
DEFINE_LUT / APPLY_LUT (the application interpolates it as the project sets, tetrahedral or trilinear): the program's
own output transform, or the same kind of output for another SDR Rec.709 display. At 65 points the LUT is 0.01-0.02
dE00 mean from the exact transform on the test frames (33 points: 0.04-0.06); inputs above 1.0 cv (linear above
222) clip at the LUT's edge. photo_graph is not part of the DCTL, as before.
"""

import math

from .. import colour as C
from ..ast import Call

L2_10 = math.log2(10.0)


def _c(v):
    from .dctl import f
    return f(v)


def _curves():
    """input kind -> C expression of x (one channel's code value) giving camera / display linear."""
    slog3 = (f"x >= {_c(171.2102946929 / 1023)} ? _exp2f((x * 1023.0f - 420.0f) / 261.5f * {_c(L2_10)}) * 0.19f - 0.01f"
             f" : (x * 1023.0f - 95.0f) * {_c(0.01125 / (171.2102946929 - 95))}")
    cut, a, b, c, d, e, f_ = 0.010591, 5.555556, 0.052272, 0.247190, 0.385537, 5.367655, 0.092809
    logc3 = (f"x > {_c(e * cut + f_)} ? (_exp2f((x - {_c(d)}) / {_c(c)} * {_c(L2_10)}) - {_c(b)}) / {_c(a)}"
             f" : (x - {_c(f_)}) / {_c(e)}")
    a4, b4, c4 = (2 ** 18 - 16) / 117.45, (1023 - 95) / 1023, 95 / 1023
    s4 = (7 * math.log(2) * 2 ** (7 - 14 * c4 / b4)) / (a4 * b4)
    t4 = (2 ** (14 * (-c4 / b4) + 6) - 64) / a4
    logc4 = (f"x >= 0.0f ? (_exp2f(14.0f * (x - {_c(c4)}) / {_c(b4)} + 6.0f) - 64.0f) / {_c(a4)}"
             f" : x * {_c(s4)} + {_c(t4)}")
    vlog = f"x < 0.181f ? (x - 0.125f) / 5.6f : _exp2f((x - 0.598206f) / 0.241514f * {_c(L2_10)}) - 0.00873f"
    r0, rt, ca, beta, gamma, delta = -0.05641088, 0.01, 47.28711236, 0.00964052, 0.08550479, 0.69336945
    apple = (f"x >= {_c(ca * (rt - r0) ** 2)} ? _exp2f((x - {_c(delta)}) / {_c(gamma)}) - {_c(beta)}"
             f" : (x > 0.0f ? _sqrtf(x / {_c(ca)}) + {_c(r0)} : {_c(r0)})")
    dlog = f"x <= 0.14f ? (x - 0.0929f) / 6.025f : (_exp2f((3.89616f * x - 2.27752f) * {_c(L2_10)}) - 0.0108f) / 0.9892f"
    return {"rec1886_photo": "_powf(_clampf(x, 0.0f, 1.0f), 2.4f)",
            "srgb_photo": "cr_srgb_dec(_clampf(x, 0.0f, 1.0f))",
            "sony_slog3_sgamut3": slog3, "sony_slog3_sgamut3cine": slog3, "arri_logc3": logc3, "arri_logc4": logc4,
            "panasonic_vlog": vlog, "apple_log": apple, "dji_dlog": dlog}


# camera linear -> AP1 linear (rows sum to 1 within 5e-6)
MATRICES = {
    "sony_slog3_sgamut3": [[1.0897985042, -0.0327650282, -0.0570331545], [-0.0311718471, 1.1829363066, -0.1517646702],
                           [-0.0032637043, -0.0015699555, 1.0048342916]],
    "sony_slog3_sgamut3cine": [[0.9345180682, 0.1436416137, -0.0781588219],
                               [-0.0505268081, 1.2616104987, -0.2110830009],
                               [-0.0245031531, -0.0306711117, 1.0551746837]],
    "arri_logc3": [[0.9666390115, 0.1155420898, -0.0821761363], [0.0481902168, 1.1849432393, -0.2331295257],
                   [0.0071929507, -0.0665944899, 1.0594060942]],
    "arri_logc4": [[1.0898816797, -0.0284555626, -0.061426158], [-0.0564717935, 1.1739580397, -0.1174877337],
                   [0.0057412041, -0.0057280422, 0.9999867471]],
    "panasonic_vlog": [[1.048662942, 0.0095532924, -0.0582161741], [-0.0293914001, 1.1458053014, -0.1164139869],
                       [-0.0033171553, -0.0056077733, 1.0089254614]],
    "apple_log": [[0.9748958374, 0.0195991892, 0.0055061538], [0.0021795991, 0.9955363381, 0.0022850974],
                  [0.0047974525, 0.0245321825, 0.9706714955]],
    "dji_dlog": [[0.991404723, 0.0885363614, -0.0799413953], [0.0266969475, 1.181194932, -0.2078912313],
                 [-0.0119096684, -0.0815217581, 1.0934311976]],
}
INPUTS = ("acescct", "rec1886_photo", "srgb_photo") + tuple(MATRICES)
OUTPUTS = ("acescct", "program") + C.SDR_709
LUT_NAME = "cr_out"


class WrapError(ValueError):
    pass


def input_fn(kind):
    """C source of cr_in(r, g, b) -> ACEScct for an input kind; None for acescct."""
    if kind == "acescct":
        return None
    if kind not in INPUTS:
        raise WrapError(f"input {kind!r} has no DCTL decode (inputs: {', '.join(INPUTS)}); put the application's "
                        "own colour space transform to ACEScct before this node and use input acescct")
    display = kind in ("rec1886_photo", "srgb_photo")
    m = C.M709_AP1 if display else MATRICES[kind]
    lines = [f"// input: {kind} -> ACEScct",
             "__DEVICE__ float cr_in_curve(float x) { return " + _curves()[kind] + "; }",
             "__DEVICE__ float cr_cct_enc_signed(float x) {",
             "    return x <= 0.0078125f ? 10.5402377416545f * x + 0.0729055341958355f : (_log2f(x) + 9.72f) / 17.52f;",
             "}",
             "__DEVICE__ float3 cr_in(float r, float g, float b) {",
             "    float l0 = cr_in_curve(r), l1 = cr_in_curve(g), l2 = cr_in_curve(b);"]
    for i in range(3):
        e = f"{_c(m[i][0])} * l0 + {_c(m[i][1])} * l1 + {_c(m[i][2])} * l2"
        # display inputs clip negative AP1 at 0 (the engine's to_working); camera inputs keep them (OpenColorIO)
        lines.append(f"    float a{i} = " + (f"_fmaxf({e}, 0.0f)" if display else e) + ";")
    lines += ["    return make_float3(cr_cct_enc_signed(a0), cr_cct_enc_signed(a1), cr_cct_enc_signed(a2));", "}"]
    return "\n".join(lines)


def output_call(prog, output):
    """The output transform the wrapper encodes with, or None for acescct: the program's own ("program"; an ACEScct
    program gets the standard ACES 2.0 sRGB output), or the program's kind of output for another SDR Rec.709
    display ("srgb" / "rec1886")."""
    if output not in OUTPUTS:
        raise WrapError(f"output {output!r} unknown (outputs: {', '.join(OUTPUTS)})")
    if output == "acescct":
        return None
    oc = prog.output if prog.output.name != "acescct" else Call("aces2", {"display": "srgb"})
    if output == "program":
        if oc.name == "aces2" and oc.args.get("display", "srgb") not in C.SDR_709:
            raise WrapError(f"output display {oc.args['display']} is not an SDR Rec.709 display; wrap with "
                            f"--out {' or '.join(C.SDR_709)}, or use an ACEScct timeline")
        return oc
    args = dict(oc.args)
    if oc.name == "aces2":
        args["display"] = output
    else:
        args["eotf"] = "srgb" if output == "srgb" else "gamma24"
    return Call(oc.name, args)


def describe(out_call):
    if out_call.name == "aces2":
        return f"ACES 2.0 SDR 100 nits, {out_call.args.get('display', 'srgb')}"
    a = out_call.args
    return f"display_encode {a.get('eotf', 'srgb')}, clip {a.get('clip', 'soft')} (no tone scale)"


def output_fn(out_call):
    """C source of cr_out(r, g, b) -> display code values for a display_encode output (exact, no LUT: through a LUT
    the steep display gamma near black costs 0.08 dE00 mean); None for aces2 (OpenColorIO only: the LUT)."""
    if out_call.name != "display_encode":
        return None
    from .dctl import f
    eotf, clip = out_call.args.get("eotf", "srgb"), out_call.args.get("clip", "soft")
    m = C.MAP1_709
    lines = ["__DEVICE__ float cr_out_enc(float v) {",
             "    v = _clampf(v, 0.0f, 1.0f);",
             "    return " + ("_powf(v, 1.0f / 2.4f)" if eotf == "gamma24" else
                              "v <= 0.0031308f ? 12.92f * v : 1.055f * _powf(v, 1.0f / 2.4f) - 0.055f") + ";",
             "}",
             "__DEVICE__ float3 cr_out(float r, float g, float b) {",
             "    float lr = cr_cct_dec(r), lg = cr_cct_dec(g), lb = cr_cct_dec(b);"]
    for i in range(3):
        lines.append(f"    float x{i} = {f(m[i][0])} * lr + {f(m[i][1])} * lg + {f(m[i][2])} * lb;")
    if clip != "hard":  # colour.soft_clip, knee 0.9
        lines += ["    float y = _clampf(0.2126f * x0 + 0.7152f * x1 + 0.0722f * x2, 0.0f, 1.0f);",
                  "    float peak = _fmaxf(x0, _fmaxf(x1, x2));",
                  "    float e2 = _expf(2.0f * _clampf((peak - 0.9f) / 0.1f, -15.0f, 15.0f));",
                  "    float target = peak > 0.9f ? 0.9f + 0.1f * (e2 - 1.0f) / (e2 + 1.0f) : peak;",
                  "    float t = _clampf((target - y) / _fmaxf(peak - y, 1e-9f), 0.0f, 1.0f);",
                  "    if (peak > 0.9f) { x0 = y + (x0 - y) * t; x1 = y + (x1 - y) * t; x2 = y + (x2 - y) * t; }",
                  "    float low = _fminf(x0, _fminf(x1, x2));",
                  "    float t2 = _clampf(y / _fmaxf(y - low, 1e-9f), 0.0f, 1.0f);",
                  "    if (low < 0.0f) { x0 = y + (x0 - y) * t2; x1 = y + (x1 - y) * t2; x2 = y + (x2 - y) * t2; }"]
    lines += ["    return make_float3(cr_out_enc(x0), cr_out_enc(x1), cr_out_enc(x2));", "}"]
    return "\n".join(lines)


OUTPUT_LUT_SIZE = 97  # 65 sampled left up to 0.83 ΔE00 on the ACES 2.0 film looks (2026-11-10); 97 fitted: 0.48


def output_lattice(out_call, size=OUTPUT_LUT_SIZE, fit_iters=20):
    """(3, size, size, size) [channel, b, g, r]: ACEScct 0..1 -> display code values through out_call, fitted by
    least squares for tetrahedral interpolation (cierra.compile.lut._fit; fit_iters 0 = sampled at the nodes).
    Measured on the worst film look (teal_orange, S-Log3 in, 6 CC0 frames): 65 sampled max 0.83 / p99.9 0.47,
    97 fitted max 0.48 / p99.9 0.26 ΔE00 (tetrahedral); trilinear 0.59 / 0.34 (the host's interpolation is not
    checked yet)."""
    from .lut import _fit, lattice

    def fn(code):
        return C.output_from_working(code, out_call)

    lat = fn(lattice(size).reshape(3, size, size * size)).reshape(3, size, size, size)
    return _fit(lat, fn, size, fit_iters) if fit_iters else lat
