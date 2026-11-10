"""A simulated grading application for the checked route (cierra/assist.py): what the fixed apply script would
send back from an ACEScct colour-managed project, and the same with a fault. No application is involved: the package's
DCTL runs through the CPU harness (the same C source), then the output transform, then a 16-bit still. Used by the
tests and by benchmarks of the route; it says nothing about a real host."""

import json

import numpy as np

from .. import colour as C
from ..assist import run_chain
from ..checker import load
from .grading_app import VERIFY_LIMITS, VERIFY_P99

FAULTS = ("no_readback", "not_applied", "applied_twice", "wrong_dctl", "wrong_frame", "bad_pixel", "wrong_output",
          "half_res")


def still16(disp):
    return np.round(np.clip(disp, 0, 1) * 65535) / 65535


def host(folder, img, prog, fault=None, other_dctl=None):
    """(result JSON, [still]) as the apply script would report them."""
    rec = json.loads((folder / "record.json").read_text())
    dctls = [(folder / n).read_text() for n in rec["dctl_files"]]
    code = np.asarray(C.input_to_working(img, prog.input)[0], dtype=np.float32)
    out = code if fault == "not_applied" else run_chain([other_dctl] if fault == "wrong_dctl" else dctls, code)
    if fault == "applied_twice":
        out = run_chain(dctls, out)
    output = load("cierra 0.1\noutput display_encode(eotf: srgb)\n").output if fault == "wrong_output" else prog.output
    disp = C.output_from_working(out, output)
    if fault == "wrong_frame":
        disp = np.roll(disp, 6, axis=2)
    if fault == "bad_pixel":
        y, x = np.argwhere((disp.max(0) < 0.9) & (disp.min(0) > 0.1))[len(disp[0]) // 2]
        disp = disp.copy()
        disp[:, y, x] = np.clip(disp[:, y, x] + 0.08, 0, 1)
    if fault == "half_res":
        disp = disp[:, ::2, ::2]
    result = {"ok": True, "readback": None if fault == "no_readback" else f"Cierra/{rec['stem']}.dctl",
              "stills": [{"ok": True, "tc": tc} for tc, _ in rec["stills"]]}
    return result, [still16(disp)]


def observable(still, ref):
    """Would this picture fail the verification limits at all (the fault shows in pixels)?"""
    if still.shape != ref.shape:
        return True
    de = C.delta_e00_display(np.clip(still, 0, 1), np.clip(ref, 0, 1), "srgb")
    white = still.max(0) >= 0.998
    rest = de[~white]
    return bool(np.quantile(de, 0.99) > VERIFY_P99 or (de > 0.5).mean() > VERIFY_LIMITS["share_over_0.5"]
                or (rest.size and rest.max() > VERIFY_LIMITS["max_unclipped"]))
