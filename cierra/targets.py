"""What each compiled target can express: one set of rules for the compilers and the linter (and the next backends).

Each function returns a list of reasons (empty = expressible)."""

import numpy as np

from . import ast as A
from .ops import MASKS, OPS
from .walk import carried_keys, graph_keys, graph_nodes, has_keys, keys_in, mask_calls

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


def _kind(spec, other):
    """Why a per-pixel target cannot hold an op: it reads other frames, or neighbouring pixels, or `other`."""
    return "temporal (reads neighbouring frames)" if "frames" in spec.aux else "spatial" if spec.spatial else other


def _pixel_shader_reasons(node, target):
    """The C-like per-pixel targets (DCTL, WGSL, CUDA) share one coverage: pointwise and position ops, masks that need no
    host plane and no blur, float32-safe curves (the `dctl` flag in ops.py marks what that code generator emits)."""
    spec = OPS[node.call.name]
    out = []
    if not spec.dctl:
        out.append(f"node {node.id}: {node.call.name}() is {_kind(spec, 'input-stage only')}")
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


# Gaussian ops the DCTL samples directly from neighbouring pixels (G5): the kernel radius int(3 sigma) at the reference
# width may be at most DCTL_TAP_RADIUS px, so (2 r + 1)^2 <= 625 taps at the reference width (4x at twice the width).
DCTL_TAP_OPS = ("blur", "sharpen", "glow", "halation", "diffusion")
DCTL_TAP_RADIUS = 12
DCTL_LUMA_OPS = ("sharpen", "local_contrast", "highlights_shadows")


def dctl_tap_radius(node):
    """The largest kernel radius (px at the reference width) a DCTL tap node needs at any frame."""
    v = node.call.args.get("radius_px", OPS[node.call.name].params["radius_px"].default)
    ks = keys_in(v)
    sigma = max(x for k in ks for _, x in k.keys) if ks else float(v)
    return 0 if sigma < 0.3 else max(1, int(3 * sigma))


def dctl_tap_node(node):
    """True for a Gaussian op the DCTL can sample directly (DCTL_TAP_OPS within the radius budget)."""
    return node.call.name in DCTL_TAP_OPS and dctl_tap_radius(node) <= DCTL_TAP_RADIUS


def dctl_luma_node(node):
    """A luma Gaussian emitted as horizontal-alpha and vertical-combine DCTLs. Small sharpen keeps the direct taps."""
    return node.call.name in DCTL_LUMA_OPS and not dctl_tap_node(node)


def dctl_spatial_node(node):
    return dctl_tap_node(node) or dctl_luma_node(node)


def dctl_node_reasons(node):
    out = _pixel_shader_reasons(node, "DCTL v0")
    if dctl_spatial_node(node):
        out = [r for r in out if not r.startswith(f"node {node.id}: {node.call.name}() is spatial")]
    elif node.call.name in DCTL_TAP_OPS:
        out = [r + f" (wider than the DCTL's {DCTL_TAP_RADIUS} px kernel radius)" if " is spatial" in r else r
               for r in out]
    return out


def dctl_tap_reasons(prog, g):
    """DCTL structure: one neighbour-sampling node per DCTL, in the top-level serial chain (it re-runs the nodes before
    it on every sampled pixel), and no carried key (key(node)) read in this graph. Uses are counted, not nodes: the
    same node twice is two neighbour-sampling passes."""
    from .walk import graph_refs
    uses = [r for r in graph_refs(g) if dctl_spatial_node(prog.nodes[r])]
    if not uses:
        return []
    top = g.items if isinstance(g, A.Serial) else [g]
    out = []
    if len(uses) > 1:
        out.append("more than one neighbour-sampling node use (" + ", ".join(uses) + "): one per DCTL")
    on_top = sum(isinstance(x, A.Ref) and x.name in uses for x in top)
    if on_top < len(uses):
        out.append("neighbour-sampling node inside a parallel / layer branch: "
                   + ", ".join(sorted(set(uses))))
    if any(carried_keys(n.mask) for n in graph_nodes(prog, g)):
        out.append("a carried key (key(node)) with a neighbour-sampling node")
    return out


def dctl_reasons(prog, graph=None):
    g = prog.graph if graph is None else graph
    return [r for n in graph_nodes(prog, g) for r in dctl_node_reasons(n)] + dctl_tap_reasons(prog, g)


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
            out.append(f"node {n.id}: {n.call.name}() is {_kind(spec, 'position-dependent')}")
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
