"""Run a generated DCTL on the CPU (gcc, float32) for parity tests: a small shim maps the DCTL built-ins to C99
<math.h>. This is a test harness, not a production path; inside the grading application the same text runs on the
GPU.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

TIMEOUT = 600  # seconds: a compiler or test harness that hangs fails instead of holding the run

SHIM = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define __DEVICE__ static inline
#define __CONSTANT__ static const
typedef struct { float x, y, z; } float3;
typedef struct { float x, y, z, w; } float4;
static inline float3 make_float3(float a, float b, float c) { float3 r = {a, b, c}; return r; }
static inline float4 make_float4(float a, float b, float c, float d) { float4 r = {a, b, c, d}; return r; }
#define _fabs fabsf
#define _powf powf
#define _logf logf
#define _log2f log2f
#define _exp2f exp2f
#define _expf expf
#define _sqrtf sqrtf
#define _cosf cosf
#define _sinf sinf
#define _atan2f atan2f
#define _floorf floorf
#define _fmaxf fmaxf
#define _fminf fminf
static inline float _clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
/* texture form: one plane of the input, read at integer pixel positions (clamped to the frame) */
typedef struct { const float *d; int w, h, c, n; } cr_tex;
#define __TEXTURE__ cr_tex
static inline float _tex2D(cr_tex t, int x, int y) {
    x = x < 0 ? 0 : (x >= t.w ? t.w - 1 : x); y = y < 0 ? 0 : (y >= t.h ? t.h - 1 : y);
    return t.d[t.n * ((size_t)y * t.w + x) + t.c];
}
"""

MAIN = r"""
int main(int argc, char **argv) {
    int w = atoi(argv[1]), h = atoi(argv[2]);
    size_t n = (size_t)w * h;
    float *buf = malloc(n * 3 * sizeof(float));
    if (fread(buf, sizeof(float), n * 3, stdin) != n * 3) return 2;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float *p = buf + 3 * ((size_t)y * w + x);
        float3 o = transform(w, h, x, y, p[0], p[1], p[2]);
        p[0] = o.x; p[1] = o.y; p[2] = o.z;
    }
    fwrite(buf, sizeof(float), n * 3, stdout);
    return 0;
}
"""


LUT_SHIM = r"""
/* DEFINE_LUT / APPLY_LUT as the application applies a 3D .cube: lattice red fastest, domain 0..1, inputs clamped */
static float3 cr_apply_lut(float r, float g, float b, const float *d, int n) {
    float x[3] = {_clampf(r, 0.0f, 1.0f) * (n - 1), _clampf(g, 0.0f, 1.0f) * (n - 1), _clampf(b, 0.0f, 1.0f) * (n - 1)};
    int i[3]; float t[3];
    for (int k = 0; k < 3; k++) { i[k] = (int)floorf(x[k]); if (i[k] > n - 2) i[k] = n - 2; t[k] = x[k] - i[k]; }
#define CR_AT(dr, dg, db, c) d[3 * (((size_t)(i[2] + (db)) * n + (i[1] + (dg))) * n + (i[0] + (dr))) + (c)]
    float o[3];
#ifdef CR_TRILINEAR
    for (int c = 0; c < 3; c++) {
        float v = 0.0f;
        for (int db = 0; db < 2; db++) for (int dg = 0; dg < 2; dg++) for (int dr = 0; dr < 2; dr++)
            v += CR_AT(dr, dg, db, c) * (dr ? t[0] : 1 - t[0]) * (dg ? t[1] : 1 - t[1]) * (db ? t[2] : 1 - t[2]);
        o[c] = v;
    }
#else
    int a = 0, m = 1, z = 2, s;  /* axes by fraction, largest first */
    if (t[a] < t[m]) { s = a; a = m; m = s; }
    if (t[m] < t[z]) { s = m; m = z; z = s; }
    if (t[a] < t[m]) { s = a; a = m; m = s; }
    int p1[3] = {0, 0, 0}, p2[3]; p1[a] = 1; p2[0] = p1[0]; p2[1] = p1[1]; p2[2] = p1[2]; p2[m] = 1;
    for (int c = 0; c < 3; c++)
        o[c] = (1 - t[a]) * CR_AT(0, 0, 0, c) + (t[a] - t[m]) * CR_AT(p1[0], p1[1], p1[2], c)
               + (t[m] - t[z]) * CR_AT(p2[0], p2[1], p2[2], c) + t[z] * CR_AT(1, 1, 1, c);
#endif
#undef CR_AT
    return make_float3(o[0], o[1], o[2]);
}
#define APPLY_LUT(r, g, b, name) cr_apply_lut((r), (g), (b), cr_lutd_##name, cr_lutn_##name)
"""
DEFINE_LUT = re.compile(r'^DEFINE_LUT\((\w+), "([^"]*)"\);$', re.M)

MAIN_TEX = r"""
int main(int argc, char **argv) {
    int w = atoi(argv[1]), h = atoi(argv[2]);
    size_t n = (size_t)w * h;
    float *buf = malloc(n * __IN__ * sizeof(float)), *dst = malloc(n * 3 * sizeof(float));
    if (fread(buf, sizeof(float), n * __IN__, stdin) != n * __IN__) return 2;
    cr_tex tr = {buf, w, h, 0, __IN__}, tg = {buf, w, h, 1, __IN__}, tb = {buf, w, h, 2, __IN__};
    cr_tex ta = {buf, w, h, 3, __IN__};
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) {
        float *p = dst + 3 * ((size_t)y * w + x);
        float3 o = transform(w, h, x, y, __TEXARGS__);
        p[0] = o.x; p[1] = o.y; p[2] = o.z;
    }
    fwrite(dst, sizeof(float), n * 3, stdout);
    return 0;
}
"""

MAIN_TEX_OUT4 = MAIN_TEX.replace("n * 3 * sizeof(float));", "n * 4 * sizeof(float));").replace(
    "float *p = dst + 3 *", "float *p = dst + 4 *").replace(
    "float3 o = transform", "float4 o = transform").replace(
    "p[0] = o.x; p[1] = o.y; p[2] = o.z;", "p[0] = o.x; p[1] = o.y; p[2] = o.z; p[3] = o.w;").replace(
    "fwrite(dst, sizeof(float), n * 3, stdout)", "fwrite(dst, sizeof(float), n * 4, stdout)")


def run_dctl(dctl_text, code, workdir=None, luts=None, interp="tetrahedral"):
    """Apply DCTL source to ACEScct (C, H, W), C=3 or 4 for a two-pass intermediate; returns float64. luts:
    {path as in DEFINE_LUT: (3, n, n, n) lattice [channel, b, g, r]}; interp "tetrahedral" or "trilinear"."""
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc not found")
    channels, h, w = code.shape
    luts = luts or {}
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        defs, loads = [], []
        for name, path in DEFINE_LUT.findall(dctl_text):
            lat = np.asarray(luts[path], dtype=np.float32)
            n = lat.shape[1]
            f = Path(tmp) / f"lut_{name}.bin"
            f.write_bytes(np.ascontiguousarray(lat.transpose(1, 2, 3, 0)).tobytes())  # (b, g, r, 3): red fastest
            defs.append(f"static float *cr_lutd_{name}; static const int cr_lutn_{name} = {n};")
            loads.append(f'    {{ FILE *fp = fopen("{f.as_posix()}", "rb"); size_t m = (size_t){n * n * n * 3}; '
                         f"cr_lutd_{name} = malloc(m * sizeof(float)); "
                         f"if (!fp || fread(cr_lutd_{name}, sizeof(float), m, fp) != m) return 3; fclose(fp); }}")
        text = DEFINE_LUT.sub("", dctl_text)
        out_ch = 4 if "__DEVICE__ float4 transform" in dctl_text else 3
        base = (MAIN_TEX_OUT4 if out_ch == 4 else MAIN_TEX) if "__TEXTURE__ p_TexR" in dctl_text else MAIN
        args = "tr, tg, tb" + (", ta" if "__TEXTURE__ p_TexA" in dctl_text else "")
        base = base.replace("__IN__", str(channels)).replace("__TEXARGS__", args)
        main = base.replace("    int w = atoi", "\n".join(loads) + "\n    int w = atoi") if loads else base
        shim = SHIM + ("#define CR_TRILINEAR\n" if interp == "trilinear" else "") + LUT_SHIM + "\n".join(defs)
        src = Path(tmp) / "prog.c"
        src.write_text(shim + "\n" + text + "\n" + main)
        exe = Path(tmp) / "prog"
        r = subprocess.run(["gcc", "-std=c99", "-O1", "-ffp-contract=off", "-o", str(exe), str(src), "-lm"],
                           capture_output=True, text=True,
                           timeout=TIMEOUT)
        if r.returncode:
            raise RuntimeError("DCTL did not compile as C:\n" + r.stderr[-4000:])
        data = np.ascontiguousarray(code.transpose(1, 2, 0), dtype=np.float32).tobytes()
        r = subprocess.run([str(exe), str(w), str(h)], input=data, capture_output=True, timeout=TIMEOUT)
        if r.returncode:
            raise RuntimeError(f"DCTL harness failed ({r.returncode})")
        out = np.frombuffer(r.stdout, dtype=np.float32).reshape(h, w, out_ch)
    return out.transpose(2, 0, 1).astype(np.float64)
