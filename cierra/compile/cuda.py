"""Cierra -> CUDA C++: one self-contained kernel source that grades an ACEScct float32 RGB buffer, one thread per pixel.

Same coverage and maths as the DCTL target (cierra/compile/dctl.py): the op / mask / graph code generator is shared,
and since DCTL is already CUDA-flavoured C, the helper texts are the DCTL ones with `__DEVICE__` spelled
`__device__ __forceinline__`, after one-line wrappers for the DCTL built-ins (_fmaxf, _powf, ...). Every pointwise and
position op and mask; keyframes evaluated at one frame; spatial nodes, blurred mattes and depth / segment masks refused
with reasons (cierra.targets.cuda_node_reasons), or with split=True a top-level serial graph is cut into several
kernels around them; a curve whose points or tangents do not survive float32 is refused.

Interface (fixed, so a host binds it without reading the text):
    extern "C" __global__ void cierra_grade(const float* src, float* dst, int width, int height, float px_scale)
    src, dst   ACEScct code values, r, g, b interleaved, row-major, 3 * width * height floats (not in place)
    px_scale   width / CIERRA_REFERENCE_WIDTH (the program's reference_width, #defined in the source): the scale of
               pixel-sized parameters such as grain size_px
    launch     a 2D grid of CIERRA_BLOCK_X x CIERRA_BLOCK_Y blocks covering width x height; threads outside return
No headers are needed (nvcc or NVRTC provide float3 / make_float3 and the float maths). Compile without fast maths
and with FMA contraction off (nvcc --fmad=false) to stay closest to the reference; the CPU harness
(cierra/compile/cuda_cpu.py) builds the same text with g++. The program's own `output` and `photo_graph` are not
part of the kernel (as for DCTL: ACEScct in, ACEScct out).
"""

from .dctl import DCTLError, _Gen, compile_parts, f
from .dctl import DISPLAY_HELPERS as _DCTL_DISPLAY, GRAIN_HELPERS as _DCTL_GRAIN, PRELUDE as _DCTL_PRELUDE
from ..targets import cuda_node_reasons

BLOCK = (16, 16)
KERNEL = "cierra_grade"
DEVICE = "__device__ __forceinline__"


class CUDAError(ValueError):
    pass


# The DCTL built-ins the shared generator and helpers call, as CUDA device functions.
BUILTINS = r"""
__DEVICE__ float _fabs(float x) { return fabsf(x); }
__DEVICE__ float _powf(float x, float y) { return powf(x, y); }
__DEVICE__ float _logf(float x) { return logf(x); }
__DEVICE__ float _log2f(float x) { return log2f(x); }
__DEVICE__ float _exp2f(float x) { return exp2f(x); }
__DEVICE__ float _expf(float x) { return expf(x); }
__DEVICE__ float _sqrtf(float x) { return sqrtf(x); }
__DEVICE__ float _cosf(float x) { return cosf(x); }
__DEVICE__ float _sinf(float x) { return sinf(x); }
__DEVICE__ float _atan2f(float y, float x) { return atan2f(y, x); }
__DEVICE__ float _floorf(float x) { return floorf(x); }
__DEVICE__ float _fmaxf(float a, float b) { return fmaxf(a, b); }
__DEVICE__ float _fminf(float a, float b) { return fminf(a, b); }
__DEVICE__ float _clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
"""


def _cuda(text):
    return text.replace("__DEVICE__", DEVICE)


PRELUDE = _cuda(BUILTINS.strip() + "\n" + _DCTL_PRELUDE)
DISPLAY_HELPERS = _cuda(_DCTL_DISPLAY)
GRAIN_HELPERS = _cuda(_DCTL_GRAIN)


class _CGen(_Gen):
    target, short, key, Error = "CUDA", "CUDA", "cuda", CUDAError
    texts = (PRELUDE, DISPLAY_HELPERS, GRAIN_HELPERS)

    @staticmethod
    def reasons(node):
        return cuda_node_reasons(node)

    def fn_head(self, ret, name, params):
        return f"{DEVICE} {ret} {name}({', '.join(f'{t} {n}' for t, n in params)}) {{"

    def header(self, title):
        return super().header(title)[:2] + [
            "// CUDA C++ kernel source (nvcc or NVRTC; no headers needed). In: ACEScct (AP1) code values. Out: ACEScct",
            "// code values (apply the program's output transform after it). Keyframed values are evaluated at frame "
            + str(int(self.frame)) + ".",
            f'// Entry: extern "C" __global__ void {KERNEL}(const float* src, float* dst, int width, int height,',
            "//     float px_scale): r, g, b interleaved, row-major, not in place; px_scale = width /",
            f"//     CIERRA_REFERENCE_WIDTH. Launch {BLOCK[0]} x {BLOCK[1]} thread blocks over a grid covering the "
            "frame.",
            "// Build with --fmad=false and without --use_fast_math to stay closest to the reference engine."]

    def entry(self, body, out, rw):
        bx, by = BLOCK
        return (f"#define CIERRA_REFERENCE_WIDTH {f(rw)}\n"
                f"#define CIERRA_BLOCK_X {bx}\n"
                f"#define CIERRA_BLOCK_Y {by}\n\n"
                f'extern "C" __global__ void {KERNEL}(const float* __restrict__ src, float* __restrict__ dst, '
                "int p_Width, int p_Height, float p_scale) {\n"
                "    int p_X = (int)(blockIdx.x * blockDim.x + threadIdx.x);\n"
                "    int p_Y = (int)(blockIdx.y * blockDim.y + threadIdx.y);\n"
                "    if (p_X >= p_Width || p_Y >= p_Height) return;\n"
                "    long long i = 3LL * ((long long)p_Y * p_Width + p_X);\n"  # no size_t: NVRTC has no headers
                "    float3 c0 = make_float3(src[i], src[i + 1], src[i + 2]);\n"
                + "".join("    " + line + "\n" for line in body)
                + f"    float3 res = {out};\n"
                "    dst[i] = res.x; dst[i + 1] = res.y; dst[i + 2] = res.z;\n}")


def compile_graph(prog, graph, frame=0, title=None, reference_width=None):
    return _CGen(prog, frame).program(graph, title, reference_width)


def compile_cuda(prog, frame=0, split=False):
    """Checked Program -> list of parts {"name", "cuda" (text or None), "graph", "nodes", "note"}, as compile_dctl:
    one part unless split=True cut a top-level serial graph around nodes CUDA cannot express (those come back with
    cuda None: run them in Cierra's engine or the host's own tools)."""
    try:
        return compile_parts(_CGen, prog, frame, split)
    except DCTLError as e:  # a non-finite constant from the shared literal writer
        raise CUDAError(str(e)) from None
