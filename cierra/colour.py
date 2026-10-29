"""Colour maths shared by the reference engine and the compilers: ACEScct, gamut matrices, display encodings, the
ACES 2.0 output transform (OpenColorIO). Images are float64 arrays shaped (3, ...) (channels first).

Provenance: acescct / matrix construction / soft_clip / display follow the reference implementation the
looks were first built on (2026).
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .backend import const, is_gpu, to_device, to_host, xp

CCT_A, CCT_B = 10.5402377416545, 0.0729055341958355
CCT_CUT_LIN, CCT_CUT_CODE = 0.0078125, 0.155251141552511
REC709_Y = np.array([0.2126, 0.7152, 0.0722])

D65 = (0.3127, 0.3290)
ACES_WHITE = (0.32168, 0.33767)
GAMUTS = {
    "rec709": (((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), D65),
    "ap1": (((0.713, 0.293), (0.165, 0.830), (0.128, 0.044)), ACES_WHITE),
    "ap0": (((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770)), ACES_WHITE),
}
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def acescct_encode(x):
    m = xp(x)
    hi = (m.log2(m.maximum(x, 1e-12)) + 9.72) / 17.52
    return m.where(x <= CCT_CUT_LIN, CCT_A * x + CCT_B, hi)


def acescct_decode(y):
    m = xp(y)
    return m.where(y <= CCT_CUT_CODE, (y - CCT_B) / CCT_A, m.power(2.0, y * 17.52 - 9.72))


def _xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz(name):
    prim, white = GAMUTS[name]
    p = np.stack([_xyz(xy) for xy in prim], axis=1)
    return p * np.linalg.solve(p, _xyz(white))


def _adapt(src, dst):
    if src == dst:
        return np.eye(3)
    a, b = _BRADFORD @ _xyz(src), _BRADFORD @ _xyz(dst)
    return np.linalg.inv(_BRADFORD) @ np.diag(b / a) @ _BRADFORD


def matrix(src, dst):
    """Linear RGB src -> dst (Bradford adaptation when the white points differ)."""
    return np.linalg.inv(rgb_to_xyz(dst)) @ _adapt(GAMUTS[src][1], GAMUTS[dst][1]) @ rgb_to_xyz(src)


def apply_matrix(m, rgb):
    return xp(rgb).einsum("ij,j...->i...", const(m, rgb), rgb)


M709_AP1 = matrix("rec709", "ap1")
MAP1_709 = matrix("ap1", "rec709")
MAP0_AP1 = matrix("ap0", "ap1")


def to_working(lin709):
    """Scene-linear Rec.709 -> ACEScct (negative AP1 values clipped to 0)."""
    return acescct_encode(xp(lin709).maximum(apply_matrix(M709_AP1, lin709), 0.0))


def from_working(code):
    return apply_matrix(MAP1_709, acescct_decode(code))


def srgb_decode(v):
    m = xp(v)
    return m.where(v <= 0.04045, v / 12.92, m.power((v + 0.055) / 1.055, 2.4))


def srgb_encode(v):
    m = xp(v)
    return m.where(v <= 0.0031308, 12.92 * v, 1.055 * m.power(m.maximum(v, 0.0), 1 / 2.4) - 0.055)


def soft_clip(lin, knee=0.9):
    """Saturated colours near the display limit pulled toward their own luminance (hue kept) so channels land
    smoothly below 1; negative channels pulled in the same way. Neutral highlights and pixels below the knee
    unchanged."""
    m = xp(lin)
    w = const(REC709_Y, lin).reshape(3, *([1] * (lin.ndim - 1)))
    y = m.clip((lin * w).sum(0, keepdims=True), 0, 1)
    peak = lin.max(0, keepdims=True)
    target = m.where(peak > knee, knee + (1 - knee) * m.tanh((peak - knee) / (1 - knee)), peak)
    t = m.clip((target - y) / m.maximum(peak - y, 1e-9), 0, 1)
    rgb = m.where(peak > knee, y + (lin - y) * t, lin)
    low = rgb.min(0, keepdims=True)
    t2 = m.clip(y / m.maximum(y - low, 1e-9), 0, 1)
    return m.clip(m.where(low < 0, y + (rgb - y) * t2, rgb), 0, 1)


def display_encode(code, eotf="srgb"):
    lin = soft_clip(from_working(code))
    if eotf == "gamma24":
        return xp(lin).power(lin, 1 / 2.4)
    return srgb_encode(lin)


def from_display(disp):
    return to_working(srgb_decode(disp))


# ---------------------------------------------------------------- ACES 2.0 output transform (OpenColorIO)

ACES2_CONFIG = "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
ACES2_VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
_P3 = "ACES 2.0 - SDR 100 nits (P3 D65)"
# display key -> (studio-config display, view)
ACES2_DISPLAYS = {"srgb": ("sRGB - Display", ACES2_VIEW), "rec1886": ("Rec.1886 Rec.709 - Display", ACES2_VIEW),
                  "display_p3": ("Display P3 - Display", _P3), "p3d65": ("P3-D65 - Display", _P3),
                  "pq_p3_1000": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"),
                  "pq_rec2020_1000": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 nits (Rec.2020)"),
                  "pq_p3_2000": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 2000 nits (P3 D65)"),
                  "pq_p3_4000": ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 4000 nits (P3 D65)"),
                  "hlg_p3_1000": ("Rec.2100-HLG - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)")}
SDR_709 = ("srgb", "rec1886")
# camera inputs -> studio-config colour space (scene-referred), converted to ACEScct by OCIO
CAMERA_INPUTS = {
    "arri_logc3": "ARRI LogC3 (EI800)",
    "arri_logc4": "ARRI LogC4",
    "sony_slog3_sgamut3": "S-Log3 S-Gamut3",
    "sony_slog3_sgamut3cine": "S-Log3 S-Gamut3.Cine",
    "panasonic_vlog": "V-Log V-Gamut",
    "canon_log2": "CanonLog2 CinemaGamut D55",
    "canon_log3": "CanonLog3 CinemaGamut D55",
    "red_log3g10": "Log3G10 REDWideGamutRGB",
    "bmd_film_gen5": "BMDFilm WideGamut Gen5",
    "apple_log": "Apple Log",
    "dji_dlog": "D-Log D-Gamut",
    "linear_rec2020": "Linear Rec.2020",
    "linear_p3d65": "Linear P3-D65",
}
_PROC = {}


def aces2_processor(display="srgb", inverse=False):
    key = (display, inverse)
    if key not in _PROC:
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromBuiltinConfig(ACES2_CONFIG)
        disp, view = ACES2_DISPLAYS[display]
        t = ocio.DisplayViewTransform(src="ACEScct", display=disp, view=view,
                                      direction=ocio.TRANSFORM_DIR_INVERSE if inverse else ocio.TRANSFORM_DIR_FORWARD)
        _PROC[key] = cfg.getProcessor(t).getDefaultCPUProcessor()
    return _PROC[key]


def _ocio_apply(proc, img):
    """OpenColorIO on the CPU; a GPU array goes to the host and the result comes back to the GPU. Large images are
    split across CIERRA_OCIO_THREADS threads (default 4): OCIO works per pixel, so the result is bit-identical, and
    its CPU processor releases Python's lock (1080p ACES 2.0 output: 1.3 s on one thread, 0.33 s on four)."""
    shape = img.shape
    a = np.ascontiguousarray(to_host(img).reshape(3, -1).T, dtype=np.float32)
    n = max(1, int(os.environ.get("CIERRA_OCIO_THREADS", "4")))
    if n > 1 and len(a) >= 65536:
        with ThreadPoolExecutor(n) as ex:
            list(ex.map(proc.applyRGB, np.array_split(a, n)))  # in place, on views of a
    else:
        proc.applyRGB(a)
    out = a.T.reshape(shape).astype(np.float64)
    return to_device(out, "gpu") if is_gpu(img) else out


def aces2_display(code, display="srgb"):
    """ACEScct -> display code values through the ACES 2.0 SDR 100-nit Rec.709 output (exact OCIO per pixel).
    OCIO 2.6's ACES 2.0 output has rare singular points where the tone-mapped J lands a float32 step above the limit:
    the gamut compression then takes powf of a slightly negative base (estimate_line_and_boundary_intersection_M),
    gets NaN, and the config's range clamp makes it exact black instead of white. It hits very bright values, saturated
    blues as well as warm highlights (e.g. ACEScct [1.025, 0.905, 0.681] and [0.126, 0.103, 1.111]); found
    by the live grading-app test (two spark pixels; the application shows them white). Such a pixel (exact black out,
    a channel above 0.5 cv in; genuine black stops at the 0.073 floor, the singular points start at 0.96) is
    re-evaluated at a 1e-4 cv nudge, the nearest value that is not singular."""
    proc = aces2_processor(display)
    code = to_host(code)  # display values come back on the CPU
    out = _ocio_apply(proc, code)
    # "black" is <= 1e-5: on PQ the clamped NaN comes out as a tiny code value, not 0 (never genuine for > 0.5 cv in)
    bad = (out.reshape(3, -1) <= 1e-5).all(0) & (np.asarray(code).reshape(3, -1).max(0) > 0.5)
    if bad.any():
        flat = out.reshape(3, -1)
        src = np.asarray(code, dtype=np.float64).reshape(3, -1)[:, bad]
        fixed = np.zeros_like(src)
        todo = np.ones(src.shape[1], dtype=bool)
        for d in (1e-4, -1e-4, 2e-4, -2e-4, 4e-4):
            if not todo.any():
                break
            trial = _ocio_apply(proc, np.ascontiguousarray((src[:, todo] + d)[:, :, None]))[:, :, 0]
            ok = trial.max(0) > 1e-5
            idx = np.where(todo)[0]
            fixed[:, idx[ok]] = trial[:, ok]
            todo[idx[ok]] = False
        flat[:, bad] = fixed
    return np.clip(out, 0, 1)


def luma_cv(code):
    """Rec.709-weighted luma of working values (as cierra.engine.luma)."""
    c = np.asarray(code)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def aces2_inverse(disp, display="srgb"):
    return _ocio_apply(aces2_processor(display, inverse=True), disp)


# ---------------------------------------------------------------- input transforms

def input_to_working(img, kind):
    """Input image (3, H, W) of the declared kind -> (ACEScct working values, display_referred flag)."""
    m = xp(img)
    if kind == "linear_rec709":
        return to_working(img), False
    if kind == "aces2065_1":
        return acescct_encode(m.maximum(apply_matrix(MAP0_AP1, img), 0.0)), False
    if kind == "acescg":
        return acescct_encode(m.maximum(img, 0.0)), False
    if kind == "acescct":
        return m.array(img, dtype=m.float64), False
    if kind == "srgb_photo":
        return to_working(srgb_decode(m.clip(img, 0, 1))), True
    if kind == "linear_photo":
        return to_working(m.maximum(img, 0.0)), True
    if kind in CAMERA_INPUTS:
        return _ocio_apply(camera_processor(kind), m.ascontiguousarray(img, dtype=m.float64)), False
    raise ValueError(kind)


_CAM = {}


def camera_processor(kind):
    if kind not in _CAM:
        import PyOpenColorIO as ocio
        cfg = ocio.Config.CreateFromBuiltinConfig(ACES2_CONFIG)
        _CAM[kind] = cfg.getProcessor(ocio.ColorSpaceTransform(src=CAMERA_INPUTS[kind], dst="ACEScct"))             .getDefaultCPUProcessor()
    return _CAM[kind]


def output_from_working(code, out_call):
    name, a = out_call.name, out_call.args
    if name == "aces2":
        return aces2_display(code, a.get("display", "srgb"))
    if name == "display_encode":
        return display_encode(code, a.get("eotf", "srgb"))
    if name == "acescct":
        return code
    raise ValueError(name)


# ---------------------------------------------------------------- CIE Lab / CIEDE2000 (display sRGB in)

M709_XYZ = rgb_to_xyz("rec709")


def display_decode(disp, eotf="srgb"):
    """Display code values -> display-linear Rec.709 with the display's EOTF: srgb, or gamma 2.4 (rec1886 /
    gamma24)."""
    d = np.clip(disp, 0, 1)
    return srgb_decode(d) if eotf == "srgb" else np.power(d, 2.4)


def output_eotf(out_call):
    """The EOTF a program's output is meant for, or None for an ACEScct (LMT) output."""
    if out_call.name == "acescct":
        return None
    v = out_call.args.get("display") or out_call.args.get("eotf") or "srgb"
    if out_call.name == "aces2" and v not in SDR_709:
        return None  # P3 / HDR: not measured as an sRGB-like display
    return "srgb" if v == "srgb" else "gamma24"


def measurable(out_call):
    """Can the grading loop measure this output (an SDR Rec.709 display, or ACEScct)?"""
    return out_call.name != "aces2" or out_call.args.get("display", "srgb") in SDR_709


def srgb_to_lab(disp, eotf="srgb"):
    """Display code values (3, ...) -> CIE L*a*b* (D65), decoded with `eotf`."""
    xyz = apply_matrix(M709_XYZ, display_decode(disp, eotf))
    wn = _xyz(D65)
    f = xyz / wn.reshape(3, *([1] * (xyz.ndim - 1)))
    d = 6 / 29
    f = np.where(f > d ** 3, np.cbrt(f), f / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def delta_e00(lab1, lab2):
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005), per pixel."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    cm = (c1 + c2) / 2
    g = 0.5 * (1 - np.sqrt(cm ** 7 / (cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(c1p * c2p == 0, 0.0, dh)
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    Lm, cmp_ = (L1 + L2) / 2, (c1p + c2p) / 2
    hs = h1p + h2p
    hm = np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2)
    hm = np.where(c1p * c2p == 0, hs, hm)
    t = (1 - 0.17 * np.cos(np.radians(hm - 30)) + 0.24 * np.cos(np.radians(2 * hm))
         + 0.32 * np.cos(np.radians(3 * hm + 6)) - 0.20 * np.cos(np.radians(4 * hm - 63)))
    dtheta = 30 * np.exp(-(((hm - 275) / 25) ** 2))
    rc = 2 * np.sqrt(cmp_ ** 7 / (cmp_ ** 7 + 25.0 ** 7))
    sl = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    sc = 1 + 0.045 * cmp_
    sh = 1 + 0.015 * cmp_ * t
    rt = -np.sin(np.radians(2 * dtheta)) * rc
    return np.sqrt((dL / sl) ** 2 + (dC / sc) ** 2 + (dH / sh) ** 2 + rt * (dC / sc) * (dH / sh))


def delta_e00_display(d1, d2, eotf="srgb"):
    """CIEDE2000 between two display images, both decoded with the display's EOTF (srgb or gamma24)."""
    return delta_e00(srgb_to_lab(d1, eotf), srgb_to_lab(d2, eotf))
