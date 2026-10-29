"""The CUDA target against the reference engine, run on the CPU by compiling the generated kernel source as C++ with
g++ through a small shim (cierra/compile/cuda_cpu.py). Same programs, frames and thresholds as the DCTL and WGSL tests
(spec §11: ΔE00 <= 0.05 mean and <= 0.5 max after the program's output transform, on every test frame). Skipped when
g++ is missing. A real-GPU run (nvcc) is optional and skipped unless nvcc is present and CIERRA_CUDA_GPU=1."""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.compile.cuda import CUDAError, KERNEL, compile_cuda
from cierra.compile.cuda_cpu import have_gxx, run_cuda
from cierra.compile.cuda_gpu import run_cuda_gpu, unavailable
from cierra.compile.dctl import compile_dctl
from cierra.engine import grade_working, make_ctx, run_graph
from cierra.targets import cuda_reasons, dctl_reasons
from conftest import LOOKS, small
from regression_cases import OP_CASES, op_input, op_source
from test_targets import KEYS_PROGRAM, KITCHEN_SINK, PASS_PROGRAM, _de, _inputs

needs_gxx = pytest.mark.skipif(not have_gxx(), reason="g++ needed for the CUDA CPU harness")
needs_gpu = pytest.mark.skipif(unavailable() is not None, reason=f"CUDA GPU runner: {unavailable()}")


def _run_parts(parts, code, prog, run=run_cuda):
    """A split program: CUDA parts through the harness, the refused nodes in the reference engine between them."""
    ctx = make_ctx(prog, code.shape[-1])
    out = code
    for p in parts:
        out = run(p["cuda"], out) if p["cuda"] else run_graph(out, p["graph"], prog, ctx)
    return out


@needs_gxx
def test_cuda_kitchen_sink_matches_reference(frames):
    prog = load(KITCHEN_SINK)
    [part] = compile_cuda(prog, frame=5)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, frame=5, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, run_cuda(part["cuda"], code), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


@needs_gxx
@pytest.mark.parametrize("src", [PASS_PROGRAM, KEYS_PROGRAM], ids=["pass", "keys"])
def test_cuda_pass(chart, src):
    prog = load(src)
    code = C.to_working(small(chart))
    ref = grade_working(code, prog)
    assert np.abs(run_cuda(compile_cuda(prog)[0]["cuda"], code) - ref).max() < 1e-4


@needs_gxx
@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern", "teal_night"])
def test_cuda_looks(frames, look):
    prog = load_file(LOOKS / f"{look}.cierra")
    parts = compile_cuda(prog, split=True)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, _run_parts(parts, code, prog), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


@needs_gxx
@pytest.mark.parametrize("name", [n for n in OP_CASES if not cuda_reasons(load(op_source(n)))])
def test_cuda_every_op_numerically(name):
    """Each op and mask on its own, in ACEScct: float32 rounding only (the bound the WGSL test uses)."""
    prog = load(op_source(name))
    code = op_input()
    out = run_cuda(compile_cuda(prog, frame=7)[0]["cuda"], code)
    ref = grade_working(code, prog, frame=7, px_scale=code.shape[-1] / prog.reference_width)
    assert np.abs(out - ref).max() < 1e-4, (name, np.abs(out - ref).max())


@needs_gxx
@pytest.mark.parametrize("frame", [0, 3, 7, 12])
def test_cuda_every_keyframable_kind(chart, frame):
    from test_analytic import KEYED_ALL
    prog = load(KEYED_ALL)
    code = C.to_working(np.ascontiguousarray(chart[:, ::4, ::4]))
    out = run_cuda(compile_cuda(prog, frame=frame)[0]["cuda"], code)
    ref = grade_working(code, prog, frame=frame, px_scale=code.shape[-1] / prog.reference_width)
    assert np.abs(out - ref).max() < 1e-5


@needs_gxx
@pytest.mark.parametrize("shape", [(1, 1), (17, 33), (3, 1)])
def test_cuda_frames_not_a_multiple_of_the_block(shape):
    """Threads outside the frame return without writing; every pixel inside is graded once (a position op and a
    grain on a frame smaller or not a multiple of the 16 x 16 block, rotated masks on one-pixel frames)."""
    prog = load("cierra 0.1\noutput acescct()\nnode s = saturation(amount: 0)\n  mask window(angle: 25)\n"
                "node v = vignette(amount: 0.3)\nnode gr = grain(amount: 0.5, size_px: 1, seed: 3)\n"
                "node g = exposure(stops: 1)\n  mask graduated(angle: 30)\ngraph s -> v -> gr -> g\n")
    rng = np.random.default_rng(1)
    code = rng.uniform(0.2, 0.6, size=(3, *shape))
    out = run_cuda(compile_cuda(prog)[0]["cuda"], code)
    ref = grade_working(code, prog, px_scale=shape[1] / prog.reference_width)
    assert np.isfinite(out).all() and np.abs(out - ref).max() < 1e-4


@needs_gxx
def test_cuda_px_scale_is_a_kernel_argument(chart):
    """The kernel takes the px scale; the harness default is width / CIERRA_REFERENCE_WIDTH, as the DCTL computes
    it, and another scale (a proxy graded at a different reference size) reaches size_px."""
    prog = load("cierra 0.1\noutput acescct()\nnode gr = grain(amount: 0.6, size_px: 3, seed: 5)\ngraph gr\n")
    code = C.to_working(small(chart, 4))
    text = compile_cuda(prog)[0]["cuda"]
    assert f"#define CIERRA_REFERENCE_WIDTH {float(prog.reference_width)}f" in text
    w = code.shape[-1]
    assert np.array_equal(run_cuda(text, code), run_cuda(text, code, px_scale=w / prog.reference_width))
    out = run_cuda(text, code, px_scale=2.0)
    assert np.abs(out - grade_working(code, prog, px_scale=2.0)).max() < 1e-4


@needs_gxx
def test_cuda_grain_hash_is_bit_exact_with_reference():
    """Spec §12: same seed + frame selects the same random cells on every target, also for negative cell indices."""
    from cierra.compile.cuda import GRAIN_HELPERS, PRELUDE
    from cierra.engine import _cell_noise, hash_u32
    frame, seed = 7, 123456789
    src = PRELUDE + GRAIN_HELPERS + r"""
#define CIERRA_REFERENCE_WIDTH 1920.0f
#define CIERRA_BLOCK_X 16
#define CIERRA_BLOCK_Y 16
extern "C" __global__ void cierra_grade(const float* src, float* dst, int w, int h, float s) {
    int x = (int)(blockIdx.x * blockDim.x + threadIdx.x), y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= w || y >= h) return;
    int ix = x - 40, iy = y - 3;
    unsigned int hh = 0x9E3779B9u;
    hh = cr_hash(hh, (unsigned int)ix); hh = cr_hash(hh, (unsigned int)iy); hh = cr_hash(hh, (unsigned int)%d);
    hh = cr_hash(hh, (unsigned int)%d); hh = cr_hash(hh, (unsigned int)2);
    long long i = 3LL * ((long long)y * w + x);
    dst[i] = (float)(hh >> 8); dst[i + 1] = (float)(hh & 255u); dst[i + 2] = cr_cell_noise(ix, iy, %d, %d, 2);
}
""" % (frame, seed, frame, seed)
    w, hgt = 96, 8
    out = run_cuda(src, np.zeros((3, hgt, w)))
    ix = (np.arange(w)[None, :] - 40) * np.ones((hgt, 1), dtype=np.int64)
    iy = (np.arange(hgt)[:, None] - 3) * np.ones((1, w), dtype=np.int64)
    ref = hash_u32(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)
    assert np.array_equal(out[0].astype(np.uint64), (ref >> np.uint32(8)).astype(np.uint64))
    assert np.array_equal(out[1].astype(np.uint64), (ref & np.uint32(255)).astype(np.uint64))
    assert np.abs(out[2] - _cell_noise(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)).max() < 1e-5


def test_cuda_source_is_self_contained():
    """One extern "C" kernel with the fixed interface, device functions only, no #include."""
    text = compile_cuda(load(KITCHEN_SINK), frame=5)[0]["cuda"]
    assert "#include" not in text and "__DEVICE__" not in text
    assert sum(ln.startswith("__global__") or ln.startswith('extern "C" __global__') for ln in text.splitlines()) == 1
    assert (f'extern "C" __global__ void {KERNEL}(const float* __restrict__ src, float* __restrict__ dst, '
            "int p_Width, int p_Height, float p_scale)") in text
    assert "__device__ __forceinline__ float3 n_exp(" in text


def test_cuda_refuses_what_dctl_refuses():
    """Same coverage as DCTL, with the target named in the reasons; split cuts around spatial nodes."""
    with pytest.raises(CUDAError, match="spatial"):
        compile_cuda(load("cierra 0.1\nnode a = glow()\ngraph a\n"))
    parts = compile_cuda(load("cierra 0.1\nnode a = exposure()\nnode b = glow()\nnode c = contrast()\n"
                              "graph a -> b -> c\n"), split=True)
    assert [p["cuda"] is None for p in parts] == [False, True, False]
    assert parts[1]["note"].startswith("not in CUDA: ")
    for mask, why in (("depth(range: 1..5)", "host plane CUDA does not have"),
                      ("luma(range: 0.2..0.5, blur_px: 4)", "CUDA cannot blur")):
        prog = load(f"cierra 0.1\nrequires depth\nnode a = exposure()\n  mask {mask}\ngraph a\n")
        with pytest.raises(CUDAError, match=why):
            compile_cuda(prog)
    with pytest.raises(CUDAError, match="parallel / layer branch"):
        compile_cuda(load("cierra 0.1\nnode a = exposure()\nnode b = glow()\ngraph parallel(a, b)\n"), split=True)
    for p in sorted(LOOKS.glob("*.cierra")):  # one coverage rule for every pixel-shader target
        prog = load_file(p)
        assert len(cuda_reasons(prog)) == len(dctl_reasons(prog))
        assert [x["nodes"] for x in compile_cuda(prog, split=True)] == [x["nodes"] for x in
                                                                         compile_dctl(prog, split=True)]


def test_cuda_refuses_float32_collapse():
    """A curve whose x points are distinct in float64 but not in float32 (the compiler refuses it on its own)."""
    from cierra import ast as A
    prog = load("cierra 0.1\nnode c = curve(points: [[0.5, 0.4], [0.6, 0.6], [1, 1]], channel: red)\ngraph c\n")
    node = prog.nodes["c"]
    prog.nodes["c"] = A.Node("c", A.Call("curve", dict(node.call.args, points=((1.0, 0.0), (1.00000001, 1.0),
                                                                                (1.5, 1.5)))))
    with pytest.raises(CUDAError, match="not distinct in float32"):
        compile_cuda(prog)


def test_cli_cuda(tmp_path, capsys):
    """`cierra cuda`: .cu files per part with --split; without it a one-line refusal and exit code 1."""
    from cierra.__main__ import main
    look = str(LOOKS / "teal_night.cierra")
    assert not main(["cuda", look, str(tmp_path / "c"), "--split"])  # None or 0: exit code 0
    assert "NOT IN CUDA" in capsys.readouterr().out
    assert len(list((tmp_path / "c").glob("*.cu"))) == 2
    assert main(["cuda", look, str(tmp_path / "d"), "--frame", "3"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("NOT IN CUDA:") and err.count("\n") == 1
    assert not (tmp_path / "d").exists()


@needs_gpu
def test_cuda_on_gpu_matches_cpu_harness(frames):
    """Optional: the kitchen sink and a split look on a real GPU (nvcc), within the same thresholds."""
    prog = load(KITCHEN_SINK)
    [part] = compile_cuda(prog, frame=5)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, frame=5, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, run_cuda_gpu(part["cuda"], code), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())
    look = load_file(LOOKS / "teal_night.cierra")
    parts = compile_cuda(look, split=True)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, look, px_scale=code.shape[-1] / look.reference_width)
        de = _de(look, _run_parts(parts, code, look, run=run_cuda_gpu), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())
