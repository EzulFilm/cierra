"""Run a generated CUDA kernel on a real NVIDIA GPU through nvcc, for parity tests on a machine that has the CUDA
toolkit. Optional: nothing else in Cierra needs it, and the tests that use it skip unless nvcc is on the PATH and
CIERRA_CUDA_GPU=1 is set (a GPU may be shared; running on it is an explicit choice). A test harness, not a
production path: a host links the same kernel source into its own CUDA (or NVRTC) code.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .cuda_cpu import scale_arg

HOST_MAIN = r"""
int main(int argc, char **argv) {
    int w = atoi(argv[1]), h = atoi(argv[2]);
    float scale = argc > 3 ? strtof(argv[3], NULL) : (float)w / CIERRA_REFERENCE_WIDTH;
    size_t n = (size_t)w * h * 3;
    float *buf = (float *)malloc(n * sizeof(float));
    if (fread(buf, sizeof(float), n, stdin) != n) return 2;
    float *d_src = NULL, *d_dst = NULL;
    if (cudaMalloc(&d_src, n * sizeof(float)) != cudaSuccess || cudaMalloc(&d_dst, n * sizeof(float)) != cudaSuccess)
        return 3;
    cudaMemcpy(d_src, buf, n * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(CIERRA_BLOCK_X, CIERRA_BLOCK_Y);
    dim3 grid((w + CIERRA_BLOCK_X - 1) / CIERRA_BLOCK_X, (h + CIERRA_BLOCK_Y - 1) / CIERRA_BLOCK_Y);
    cierra_grade<<<grid, block>>>(d_src, d_dst, w, h, scale);
    if (cudaDeviceSynchronize() != cudaSuccess || cudaGetLastError() != cudaSuccess) return 4;
    cudaMemcpy(buf, d_dst, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src); cudaFree(d_dst);
    fwrite(buf, sizeof(float), n, stdout);
    return 0;
}
"""


def unavailable():
    """None when the GPU runner may run here, else why not."""
    if shutil.which("nvcc") is None:
        return "nvcc not found (CUDA toolkit not installed)"
    if os.environ.get("CIERRA_CUDA_GPU") != "1":
        return "set CIERRA_CUDA_GPU=1 to run CUDA kernels on the GPU"
    return None


def run_cuda_gpu(cuda_text, code, px_scale=None, workdir=None):
    """Apply CUDA source from compile_cuda to ACEScct (3, H, W) on the GPU; returns float64 (3, H, W)."""
    why = unavailable()
    if why:
        raise RuntimeError(why)
    _, h, w = code.shape
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        src = Path(tmp) / "prog.cu"
        src.write_text("#include <stdio.h>\n#include <stdlib.h>\n" + cuda_text + "\n" + HOST_MAIN)
        exe = Path(tmp) / "prog"
        r = subprocess.run(["nvcc", "-O2", "--fmad=false", "-o", str(exe), str(src)], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("CUDA source did not compile with nvcc:\n" + r.stderr[-4000:])
        data = np.ascontiguousarray(code.transpose(1, 2, 0), dtype=np.float32).tobytes()
        args = [str(exe), str(w), str(h)] + ([] if px_scale is None else [scale_arg(px_scale)])
        r = subprocess.run(args, input=data, capture_output=True)
        if r.returncode:
            raise RuntimeError(f"CUDA GPU harness failed ({r.returncode})")
        out = np.frombuffer(r.stdout, dtype=np.float32).reshape(h, w, 3)
    return out.transpose(2, 0, 1).astype(np.float64)
