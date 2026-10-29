"""The WGSL target against the reference engine, run through wgpu-py (cierra/compile/wgsl_run.py). Same programs,
frames and thresholds as the DCTL tests in test_targets.py (spec §11: ΔE00 <= 0.05 mean and <= 0.5 max after the
program's output transform, on every test frame). Skipped when wgpu or a wgpu adapter is missing (on a machine without a
GPU, wgpu finds Mesa's llvmpipe, a CPU Vulkan driver, so these run there too)."""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.compile.dctl import compile_dctl
from cierra.compile.wgsl import WGSLError, compile_wgsl
from cierra.compile.wgsl_run import adapter, run_wgsl
from cierra.engine import grade_working, make_ctx, run_graph
from cierra.targets import dctl_reasons, wgsl_reasons
from conftest import LOOKS, small
from regression_cases import OP_CASES, op_input, op_source
from test_targets import KEYS_PROGRAM, KITCHEN_SINK, PASS_PROGRAM, _de, _inputs

needs_wgpu = pytest.mark.skipif(adapter()[0] is None, reason=f"WGSL harness: {adapter()[1]}")


def _run_parts(parts, code, prog, frame=0):
    """A split program: WGSL parts on the adapter, the refused nodes in the reference engine between them."""
    ctx = make_ctx(prog, code.shape[-1])
    out = code
    for p in parts:
        out = run_wgsl(p["wgsl"], out) if p["wgsl"] else run_graph(out, p["graph"], prog, ctx)
    return out


@needs_wgpu
def test_wgsl_kitchen_sink_matches_reference(frames):
    prog = load(KITCHEN_SINK)
    [part] = compile_wgsl(prog, frame=5)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, frame=5, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, run_wgsl(part["wgsl"], code), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


@needs_wgpu
@pytest.mark.parametrize("src", [PASS_PROGRAM, KEYS_PROGRAM], ids=["pass", "keys"])
def test_wgsl_pass(chart, src):
    prog = load(src)
    code = C.to_working(small(chart))
    ref = grade_working(code, prog)
    assert np.abs(run_wgsl(compile_wgsl(prog)[0]["wgsl"], code) - ref).max() < 1e-4


@needs_wgpu
@pytest.mark.parametrize("look", ["cinematic_modern", "beauty_modern", "teal_night"])
def test_wgsl_looks(frames, look):
    prog = load_file(LOOKS / f"{look}.cierra")
    parts = compile_wgsl(prog, split=True)
    for name, code in _inputs(frames).items():
        ref = grade_working(code, prog, px_scale=code.shape[-1] / prog.reference_width)
        de = _de(prog, _run_parts(parts, code, prog), ref)
        assert de.mean() <= 0.05 and de.max() <= 0.5, (name, de.mean(), de.max())


@needs_wgpu
@pytest.mark.parametrize("name", [n for n in OP_CASES if not wgsl_reasons(load(op_source(n)))])
def test_wgsl_every_op_numerically(name):
    """Each op and mask on its own, in ACEScct: float32 rounding only (the same bound the DCTL keyframe test uses
    would be too tight for grain's hash-to-float and the display curve's pow; 1e-4 cv is ~0.02 % of a stop)."""
    prog = load(op_source(name))
    code = op_input()
    out = run_wgsl(compile_wgsl(prog, frame=7)[0]["wgsl"], code)
    ref = grade_working(code, prog, frame=7, px_scale=code.shape[-1] / prog.reference_width)
    assert np.abs(out - ref).max() < 1e-4, (name, np.abs(out - ref).max())


@needs_wgpu
@pytest.mark.parametrize("frame", [0, 3, 7, 12])
def test_wgsl_every_keyframable_kind(chart, frame):
    from test_analytic import KEYED_ALL
    prog = load(KEYED_ALL)
    code = C.to_working(np.ascontiguousarray(chart[:, ::4, ::4]))
    out = run_wgsl(compile_wgsl(prog, frame=frame)[0]["wgsl"], code)
    ref = grade_working(code, prog, frame=frame, px_scale=code.shape[-1] / prog.reference_width)
    assert np.abs(out - ref).max() < 1e-5


@needs_wgpu
def test_wgsl_grain_hash_is_bit_exact_with_reference():
    """Spec §12: same seed + frame selects the same random cells on every target, also for negative cell indices."""
    from cierra.compile.wgsl import GRAIN_HELPERS, PRELUDE
    from cierra.engine import _cell_noise, hash_u32
    frame, seed = 7, 123456789
    src = PRELUDE + GRAIN_HELPERS + r"""
@group(0) @binding(0) var<storage, read> cr_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cr_dst: array<f32>;
@group(0) @binding(2) var<uniform> cr_size: vec2<u32>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= cr_size.x || gid.y >= cr_size.y) { return; }
    let ix = i32(gid.x) - 40; let iy = i32(gid.y) - 3;
    var h = 0x9E3779B9u;
    h = cr_hash(h, bitcast<u32>(ix)); h = cr_hash(h, bitcast<u32>(iy)); h = cr_hash(h, %du);
    h = cr_hash(h, %du); h = cr_hash(h, 2u);
    let i = 3u * (gid.y * cr_size.x + gid.x);
    _ = cr_src[i];  // the harness binds all three buffers
    cr_dst[i] = f32(h >> 8u); cr_dst[i + 1u] = f32(h & 255u); cr_dst[i + 2u] = cr_cell_noise(ix, iy, %d, %d, 2);
}
""" % (frame, seed, frame, seed)
    w, hgt = 96, 8
    out = run_wgsl(src, np.zeros((3, hgt, w)))
    ix = (np.arange(w)[None, :] - 40) * np.ones((hgt, 1), dtype=np.int64)
    iy = (np.arange(hgt)[:, None] - 3) * np.ones((1, w), dtype=np.int64)
    ref = hash_u32(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)
    assert np.array_equal(out[0].astype(np.uint64), (ref >> np.uint32(8)).astype(np.uint64))
    assert np.array_equal(out[1].astype(np.uint64), (ref & np.uint32(255)).astype(np.uint64))
    assert np.abs(out[2] - _cell_noise(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)).max() < 1e-5


def test_wgsl_refuses_what_dctl_refuses():
    """Same coverage as DCTL, with the target named in the reasons; split cuts around spatial nodes."""
    with pytest.raises(WGSLError, match="spatial"):
        compile_wgsl(load("cierra 0.1\nnode a = glow()\ngraph a\n"))
    parts = compile_wgsl(load("cierra 0.1\nnode a = exposure()\nnode b = glow()\nnode c = contrast()\n"
                              "graph a -> b -> c\n"), split=True)
    assert [p["wgsl"] is None for p in parts] == [False, True, False]
    assert parts[1]["note"].startswith("not in WGSL: ")
    for mask, why in (("depth(range: 1..5)", "host plane WGSL does not have"),
                      ("luma(range: 0.2..0.5, blur_px: 4)", "WGSL cannot blur")):
        prog = load(f"cierra 0.1\nrequires depth\nnode a = exposure()\n  mask {mask}\ngraph a\n")
        with pytest.raises(WGSLError, match=why):
            compile_wgsl(prog)
    with pytest.raises(WGSLError, match="parallel / layer branch"):
        compile_wgsl(load("cierra 0.1\nnode a = exposure()\nnode b = glow()\ngraph parallel(a, b)\n"), split=True)
    for p in sorted(LOOKS.glob("*.cierra")):  # one coverage rule for both targets
        prog = load_file(p)
        assert len(wgsl_reasons(prog)) == len(dctl_reasons(prog))
        assert [x["nodes"] for x in compile_wgsl(prog, split=True)] == [x["nodes"] for x in
                                                                         compile_dctl(prog, split=True)]


def test_wgsl_refuses_float32_collapse():
    """A curve whose x points are distinct in float64 but not in float32 (the checker's spacing rule would stop it;
    the compiler refuses it on its own)."""
    from cierra import ast as A
    prog = load("cierra 0.1\nnode c = curve(points: [[0.5, 0.4], [0.6, 0.6], [1, 1]], channel: red)\ngraph c\n")
    node = prog.nodes["c"]
    prog.nodes["c"] = A.Node("c", A.Call("curve", dict(node.call.args, points=((1.0, 0.0), (1.00000001, 1.0),
                                                                                (1.5, 1.5)))))
    with pytest.raises(WGSLError, match="not distinct in float32"):
        compile_wgsl(prog)
