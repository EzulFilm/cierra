"""Cierra -> ASC CDL (.cc / .ccc / .cdl): one slope / offset / power / saturation that every on-set and grading tool
reads. A CDL is a small fixed formula, so only programs that are exactly one CDL fold into it:

  a serial graph (no parallel / layer) of printer_lights, contrast, cdl and saturation (protect_skin 0) nodes,
  without masks, mix < 1 or keyframes, in the CDL's own order: the affine ops (slope / offset per channel) first, at
  most one power (a cdl node), then saturation. exposure and white_balance are linear-light gains, which are an
  ACEScct offset only above the toe, so they are refused (printer_lights is the CDL offset).

Everything else is refused with the reasons (use the DCTL or a LUT instead). Values are ACEScct code values in and
out, so the file is an LMT for an ACEScct pipeline ("ACEScct CDL"); the program's output transform is not in it.
Cierra's cdl op is unclamped (CLF FwdNoClamp); ASC readers usually clamp to 0..1 after the SOP and after saturation,
which only matters above ACEScct 1.0 (~222 linear) or below 0.
"""

import math

import numpy as np

from ..format import program_hash
from ..walk import graph_refs

PRINTER_POINT = 0.025 / (17.52 * math.log10(2))  # cv per printer point, as cierra.engine.op_printer_lights
NS = "urn:ASC:CDL:v1.01"


class CDLError(ValueError):
    pass


def _is_keys(v):
    from .. import ast as A
    return isinstance(v, A.Keys) or (isinstance(v, tuple) and any(isinstance(x, A.Keys) for x in v))


def _affine(node):
    """(slope rgb, offset rgb) of an affine op, or None."""
    a, name = node.call.args, node.call.name
    get = lambda k, d: a.get(k, d)  # noqa: E731
    if name in ("exposure", "white_balance") and not any(get(k, 0.0) for k in ("stops", "temperature", "tint")):
        return np.ones(3), np.zeros(3)  # at their defaults: the identity
    if name == "printer_lights":
        return np.ones(3), np.array([get(c, 0.0) for c in ("red", "green", "blue")]) * PRINTER_POINT
    if name == "contrast":
        amt, piv = get("amount", 1.0), get("pivot", 0.4135884)
        return np.full(3, amt), np.full(3, piv * (1 - amt))
    return None


def fold(prog):
    """(slope, offset, power, saturation) of the whole graph, or CDLError listing every reason it cannot be one CDL."""
    from .. import ast as A
    reasons = []
    if prog.graph is None:
        return np.ones(3), np.zeros(3), np.ones(3), 1.0
    g = prog.graph

    def serial_only(x):
        return isinstance(x, (A.Ref, A.Pass)) or (isinstance(x, A.Serial) and all(serial_only(i) for i in x.items))
    if not serial_only(g):
        reasons.append("the graph uses parallel() or layer(): a CDL is one serial formula")
    slope, offset, power, sat = np.ones(3), np.zeros(3), np.ones(3), 1.0
    stage = 0  # 0 affine, 1 after a power, 2 after saturation
    for node in [prog.nodes[r] for r in graph_refs(g)]:  # every use, in order (a node may be used twice)
        if not node.enabled:
            continue
        nid, call = node.id, node.call
        if node.mask is not None:
            reasons.append(f"node {nid}: has a mask")
        if not _is_keys(node.mix) and node.mix != 1.0:
            reasons.append(f"node {nid}: mix is not 1")
        if any(_is_keys(v) for v in call.args.values()) or _is_keys(node.mix):
            reasons.append(f"node {nid}: keyframed")
            continue
        aff = _affine(node)
        if aff is not None:
            s, o = aff
            if stage and not (np.allclose(s, 1) and np.allclose(o, 0)):
                reasons.append(f"node {nid}: {call.name} after a power or saturation (a CDL applies slope and offset "
                               "first)")
            slope, offset = slope * s, offset * s + o
        elif call.name == "cdl":
            s = np.array(call.args.get("slope", (1.0, 1.0, 1.0)), dtype=float)
            o = np.array(call.args.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            p = np.array(call.args.get("power", (1.0, 1.0, 1.0)), dtype=float)
            k = float(call.args.get("sat", 1.0))
            if stage and not (np.allclose(s, 1) and np.allclose(o, 0)):
                reasons.append(f"node {nid}: slope / offset after a power or saturation")
            if stage == 2 and not np.allclose(p, 1):
                reasons.append(f"node {nid}: a power after saturation")
            if not np.allclose(p, 1) and not np.allclose(power, 1):
                reasons.append(f"node {nid}: a second power (a CDL has one)")
            slope, offset = slope * s, offset * s + o
            if not np.allclose(p, 1):
                power, stage = p, max(stage, 1)
            if k != 1.0:
                sat, stage = sat * k, 2
        elif call.name in ("exposure", "white_balance"):
            reasons.append(f"node {nid}: {call.name} is a linear-light gain, not an ACEScct offset below the toe (code "
                           "0.155); use printer_lights or cdl() for a CDL offset")
        elif call.name == "saturation":
            if call.args.get("protect_skin", 0.0):
                reasons.append(f"node {nid}: protect_skin is not a CDL saturation")
            sat, stage = sat * float(call.args.get("amount", 1.0)), 2
        else:
            reasons.append(f"node {nid}: {call.name} is not a CDL operation")
    if reasons:
        raise CDLError("not one CDL: " + "; ".join(reasons))
    return slope, offset, power, sat


def _cc(prog, values, cid):
    slope, offset, power, sat = values
    f = lambda v: " ".join(f"{x:.7g}" for x in v)  # noqa: E731
    desc = (prog.name or "Cierra grade").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'  <ColorCorrection id="{cid}">\n'
            f"    <Description>{desc} (Cierra 0.1, program {program_hash(prog)}; ACEScct in and out)</Description>\n"
            f"    <SOPNode>\n      <Slope>{f(slope)}</Slope>\n      <Offset>{f(offset)}</Offset>\n"
            f"      <Power>{f(power)}</Power>\n    </SOPNode>\n"
            f"    <SatNode>\n      <Saturation>{sat:.7g}</Saturation>\n    </SatNode>\n  </ColorCorrection>\n")


def export_cdl(prog, folder, stem):
    """Write <stem>.cc, .ccc and .cdl (the same correction three ways); return {file: description}."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    values = fold(prog)
    cid = f"cierra-{program_hash(prog)}"
    cc = _cc(prog, values, cid)
    head = '<?xml version="1.0" encoding="UTF-8"?>\n'
    files = {}
    (folder / f"{stem}.cc").write_text(head + cc.replace("  <ColorCorrection", f'<ColorCorrection xmlns="{NS}"', 1)
                                       .replace("\n  ", "\n"), encoding="utf-8")
    files[f"{stem}.cc"] = "ASC CDL ColorCorrection (ACEScct in/out)"
    (folder / f"{stem}.ccc").write_text(head + f'<ColorCorrectionCollection xmlns="{NS}">\n{cc}'
                                        "</ColorCorrectionCollection>\n", encoding="utf-8")
    files[f"{stem}.ccc"] = "ASC CDL ColorCorrectionCollection with one correction"
    (folder / f"{stem}.cdl").write_text(head + f'<ColorDecisionList xmlns="{NS}">\n<ColorDecision>\n{cc}'
                                        "</ColorDecision>\n</ColorDecisionList>\n", encoding="utf-8")
    files[f"{stem}.cdl"] = "ASC CDL ColorDecisionList with one decision"
    return files
