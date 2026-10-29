"""Run a generated DCTL on the CPU (gcc, float32) for parity tests: a small shim maps the DCTL built-ins to C99
<math.h>. This is a test harness, not a production path; inside the grading application the same text runs on the
GPU.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SHIM = r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define __DEVICE__ static inline
#define __CONSTANT__ static const
typedef struct { float x, y, z; } float3;
static inline float3 make_float3(float a, float b, float c) { float3 r = {a, b, c}; return r; }
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


def run_dctl(dctl_text, code, workdir=None):
    """Apply DCTL source to ACEScct (3, H, W); returns float64 (3, H, W)."""
    if shutil.which("gcc") is None:
        raise RuntimeError("gcc not found")
    _, h, w = code.shape
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src = Path(tmp) / "prog.c"
        src.write_text(SHIM + "\n" + dctl_text + "\n" + MAIN)
        exe = Path(tmp) / "prog"
        r = subprocess.run(["gcc", "-std=c99", "-O1", "-ffp-contract=off", "-o", str(exe), str(src), "-lm"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("DCTL did not compile as C:\n" + r.stderr[-4000:])
        data = np.ascontiguousarray(code.transpose(1, 2, 0), dtype=np.float32).tobytes()
        r = subprocess.run([str(exe), str(w), str(h)], input=data, capture_output=True)
        if r.returncode:
            raise RuntimeError(f"DCTL harness failed ({r.returncode})")
        out = np.frombuffer(r.stdout, dtype=np.float32).reshape(h, w, 3)
    return out.transpose(2, 0, 1).astype(np.float64)
