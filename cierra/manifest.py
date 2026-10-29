"""Machine-readable operation manifest, built from the op table (ops.py) so documentation, checker and prompt cannot
drift. tools/make_authoring.py writes it to docs/ops_manifest.json."""

from .ops import BLEND_MODES, IDENTITY_DEFAULTS, INPUTS, INTERPS, LIMITS, MASKS, MODIFIERS, OPS, OUTPUTS

KEYFRAMABLE = ("num", "rgb", "pair")


def _param(p):
    d = {"kind": p.kind, "unit": p.unit or None, "doc": p.doc or None, "required": p.required,
         "keyframable": p.kind in KEYFRAMABLE}
    if p.lo is not None:
        d["range"] = [p.lo, p.hi]
    if p.choices:
        d["choices"] = list(p.choices)
    if p.kind == "points":
        d["columns"] = [{"label": c[2], "range": [c[0], c[1]]} for c in p.cols]
        d["optional_columns"] = [{"label": c[2], "range": [c[0], c[1]]} for c in p.opt_cols]
        d["length"] = [p.min_len, p.max_len]
        d["sorted"] = p.sorted
    if p.kind == "ints":
        d["length"] = [p.min_len, p.max_len]
    if not p.required:
        d["default"] = list(p.default) if isinstance(p.default, tuple) else p.default
    return d


def _spec(s, mask=False):
    kind = "spatial" if s.spatial else ("position" if s.position else "pointwise")
    if s.aux:
        kind = "aux"
    return {"doc": s.doc, "category": s.category, "kind": kind, "aux_planes": list(s.aux),
            "photo_graph_only": s.photo_only, "identity_with_defaults": s.name in IDENTITY_DEFAULTS,
            "targets": {"reference": True, "dctl": bool(s.dctl and kind in ("pointwise", "position")),
                        "lut": kind == "pointwise" and not s.photo_only},
            "params": {k: _param(p) for k, p in s.params.items()}}


def manifest():
    from . import LANGUAGE_VERSION
    return {
        "language": "cierra", "version": LANGUAGE_VERSION,
        "inputs": INPUTS,
        "outputs": {k: {"doc": v.doc, "params": {n: _param(p) for n, p in v.params.items()}} for k, v in OUTPUTS.items()},
        "ops": {k: _spec(v) for k, v in OPS.items()},
        "masks": {k: _spec(v, True) for k, v in MASKS.items()},
        "mask_combinators": ["and (product)", "or (max)", "not (1 - m)"],
        "node_modifiers": {"mix": _param(MODIFIERS["mix"]), "mask": "a mask expression, at most one per node",
                           "label": "string <= 400 characters", "off": "bypass the node (passes its input through)"},
        "graph": {"serial": "a -> b", "parallel": "parallel(g1, ..., gn): unweighted mean of the branch outputs in "
                  "ACEScct code values, every branch on the same input (n <= 8)",
                  "layer": {"modes": list(BLEND_MODES), "mode": _param(MODIFIERS["layer.mode"]),
                            "opacity": _param(MODIFIERS["layer.opacity"])},
                  "omitted": "no graph line = identity"},
        "keyframes": {"interpolations": list(INTERPS), "frames": [0, LIMITS["max_frame"]], "max_keys": LIMITS["keys"]},
        "limits": LIMITS,
        "notes": "targets.lut is per op; a mask with blur_px > 0, keyframes, or any non-pointwise node makes the "
                 "whole program not LUT-expressible. targets.dctl excludes blurred mattes and depth / segment masks.",
    }
