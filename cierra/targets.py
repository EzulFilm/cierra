"""What each compiled target can express: one set of rules for the compilers and the linter (and the next backends).

Each function returns a list of reasons (empty = expressible)."""

import numpy as np

from . import ast as A
from .ops import MASKS, OPS
from .walk import graph_keys, graph_nodes, has_keys, keys_in, mask_calls

CURVE_OPS = ("curve", "lum_vs_sat", "sat_vs_sat")


def _blur_values(mc):
    out = []
    for key in ("blur_px", "feather_px"):
        v = mc.call.args.get(key) or 0.0
        ks = keys_in(v)
        out += [x for k in ks for _, x in k.keys] if ks else [v]
    return out


def blurred(mc):
    """True when a mask call blurs or feathers its matte at any frame (both are spatial)."""
    return any(x > 0 for x in _blur_values(mc))


def float32_curve_reason(node):
    """A curve whose points or tangents do not survive float32 (adjacent x values collapse, or overflow)."""
    if node.call.name not in CURVE_OPS:
        return None
    from .engine import curve_tangents
    xs, ys, ms = curve_tangents(node.call.args["points"])
    with np.errstate(over="ignore"):
        xs32 = np.array(xs, dtype=np.float32)
        rest = np.array(list(ys) + list(ms), dtype=np.float64).astype(np.float32)
    if np.any(np.diff(xs32) <= 0):
        return f"node {node.id}: curve x points are not distinct in float32"
    if not np.all(np.isfinite(rest)):
        return f"node {node.id}: curve value or tangent is outside the float32 range"
    return None


def _pixel_shader_reasons(node, target):
    """The C-like per-pixel targets (DCTL, WGSL, CUDA) share one coverage: pointwise and position ops, masks that need no
    host plane and no blur, float32-safe curves (the `dctl` flag in ops.py marks what that code generator emits)."""
    spec = OPS[node.call.name]
    out = []
    if not spec.dctl:
        out.append(f"node {node.id}: {node.call.name}() is {'spatial' if spec.spatial else 'input-stage only'}")
    for mc in mask_calls(node.mask):
        ms = MASKS[mc.call.name]
        if not ms.dctl:
            out.append(f"node {node.id}: mask {mc.call.name}() needs a host plane {target} does not have")
        elif blurred(mc):
            out.append(f"node {node.id}: mask {mc.call.name}() blurs or feathers its matte (blur_px / feather_px); {target} cannot blur")
    r = float32_curve_reason(node)
    if r:
        out.append(r)
    return out


def dctl_node_reasons(node):
    return _pixel_shader_reasons(node, "DCTL v0")


def dctl_reasons(prog, graph=None):
    g = prog.graph if graph is None else graph
    return [r for n in graph_nodes(prog, g) for r in dctl_node_reasons(n)]


def wgsl_node_reasons(node):
    """WGSL compute shader (cierra/compile/wgsl.py): the same coverage as DCTL; it has no depth / segment planes
    bound and grades one pixel per invocation, so it cannot blur either."""
    return _pixel_shader_reasons(node, "WGSL")


def wgsl_reasons(prog, graph=None):
    g = prog.graph if graph is None else graph
    return [r for n in graph_nodes(prog, g) for r in wgsl_node_reasons(n)]


def cuda_node_reasons(node):
    """CUDA kernel (cierra/compile/cuda.py): the same coverage as DCTL; one thread per pixel with no depth / segment
    planes bound, so it cannot blur either."""
    return _pixel_shader_reasons(node, "CUDA")


def cuda_reasons(prog, graph=None):
    g = prog.graph if graph is None else graph
    return [r for n in graph_nodes(prog, g) for r in cuda_node_reasons(n)]


def lut_reasons(prog, graph=None):
    """Can the graph live in a 3D LUT (pointwise, nothing animated)?"""
    g = prog.graph if graph is None else graph
    out = []
    for n in graph_nodes(prog, g):
        spec = OPS[n.call.name]
        if not spec.pointwise:
            out.append(f"node {n.id}: {n.call.name}() is {'spatial' if spec.spatial else 'position-dependent'}")
        for mc in mask_calls(n.mask):
            ms = MASKS[mc.call.name]
            if not ms.pointwise:
                out.append(f"node {n.id}: mask {mc.call.name}() is not pointwise")
            elif blurred(mc):
                out.append(f"node {n.id}: mask {mc.call.name}() is blurred")
        if has_keys(n):
            out.append(f"node {n.id}: keyframed (a LUT holds one frame)")
    if graph_keys(g):
        out.append("graph: keyframed layer opacity (a LUT holds one frame)")
    return out


__all__ = ["A", "blurred", "cuda_node_reasons", "cuda_reasons", "dctl_node_reasons", "dctl_reasons", "float32_curve_reason", "lut_reasons",
           "wgsl_node_reasons", "wgsl_reasons"]
