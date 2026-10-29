"""Run a generated CUDA kernel on the CPU (g++, float32) for parity tests: a small shim header defines the CUDA
qualifiers away, provides float3 / make_float3 and the thread indices, and a main launches the kernel over every
block and thread of the grid in turn. No nvcc or GPU is needed. This is a test harness, not a production path; on a
GPU the same text runs unchanged (cierra/compile/cuda_gpu.py, when nvcc is present).
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SHIM = r"""
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#define __global__
#define __device__
#define __forceinline__ inline
struct float3 { float x, y, z; };
static inline float3 make_float3(float a, float b, float c) { float3 r = {a, b, c}; return r; }
struct dim3 { unsigned int x, y, z; };
static dim3 threadIdx, blockIdx, blockDim, gridDim;
"""

MAIN = r"""
int main(int argc, char **argv) {
    int w = atoi(argv[1]), h = atoi(argv[2]);
    float scale = argc > 3 ? strtof(argv[3], NULL) : (float)w / CIERRA_REFERENCE_WIDTH;
    size_t n = (size_t)w * h * 3;
    float *src = (float *)malloc(n * sizeof(float)), *dst = (float *)calloc(n, sizeof(float));
    if (fread(src, sizeof(float), n, stdin) != n) return 2;
    blockDim.x = CIERRA_BLOCK_X; blockDim.y = CIERRA_BLOCK_Y; blockDim.z = 1;
    gridDim.x = (w + CIERRA_BLOCK_X - 1) / CIERRA_BLOCK_X; gridDim.y = (h + CIERRA_BLOCK_Y - 1) / CIERRA_BLOCK_Y;
    gridDim.z = 1;
    for (blockIdx.y = 0; blockIdx.y < gridDim.y; blockIdx.y++) for (blockIdx.x = 0; blockIdx.x < gridDim.x; blockIdx.x++)
        for (threadIdx.y = 0; threadIdx.y < blockDim.y; threadIdx.y++)
            for (threadIdx.x = 0; threadIdx.x < blockDim.x; threadIdx.x++)
                cierra_grade(src, dst, w, h, scale);
    fwrite(dst, sizeof(float), n, stdout);
    return 0;
}
"""


def have_gxx():
    return shutil.which("g++") is not None


def scale_arg(px_scale):
    """The px scale as the float32 the kernel receives, written so strtof reads it back exactly."""
    return f"{float(np.float32(px_scale)):.9g}"


def run_cuda(cuda_text, code, px_scale=None, workdir=None):
    """Apply CUDA source from compile_cuda to ACEScct (3, H, W); returns float64 (3, H, W). px_scale None: width /
    the program's reference width, as the DCTL computes it."""
    if not have_gxx():
        raise RuntimeError("g++ not found")
    _, h, w = code.shape
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src = Path(tmp) / "prog.cpp"
        src.write_text(SHIM + "\n" + cuda_text + "\n" + MAIN)
        exe = Path(tmp) / "prog"
        r = subprocess.run(["g++", "-std=c++17", "-O1", "-ffp-contract=off", "-o", str(exe), str(src), "-lm"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("CUDA source did not compile as C++:\n" + r.stderr[-4000:])
        data = np.ascontiguousarray(code.transpose(1, 2, 0), dtype=np.float32).tobytes()
        args = [str(exe), str(w), str(h)] + ([] if px_scale is None else [scale_arg(px_scale)])
        r = subprocess.run(args, input=data, capture_output=True)
        if r.returncode:
            raise RuntimeError(f"CUDA CPU harness failed ({r.returncode})")
        out = np.frombuffer(r.stdout, dtype=np.float32).reshape(h, w, 3)
    return out.transpose(2, 0, 1).astype(np.float64)
