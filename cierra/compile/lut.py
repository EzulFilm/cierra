"""Cierra -> 3D LUTs: a .cube and a CLF (Academy/ASC Common LUT Format v3) for any tool.

Two LUTs per program:
  <name>_ACEScct.cube / .clf          the graph alone (an LMT): ACEScct in -> ACEScct out
  <name>_ACEScct_to_<display>.cube    graph + the program's output transform: ACEScct in -> display code values out
A LUT holds one pointwise function of colour, so it can only express programs whose graph is pointwise with no
keyframes; spatial / position ops, blurred or position masks and aux masks are refused, and the reasons are written
into the export note. Lattice 65^3 by default, float values unclamped (CLF) / as written (cube). The lattice is
fitted for tetrahedral interpolation (bake: least squares at 8 points per cell), not just sampled at the nodes.
"""

import re

import numpy as np

from .. import colour as C
from ..engine import grade_working, is_lut_expressible
from ..format import program_hash


class LUTError(ValueError):
    pass


def lattice(size):
    """(3, size, size, size) ACEScct lattice indexed [channel, b, g, r] (red fastest when flattened)."""
    axis = np.linspace(0.0, 1.0, size)
    b, g, r = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([r, g, b])


def _tetra(size, pts):
    """Tetrahedral interpolation (as OpenColorIO and most grading tools apply a 3D LUT) of points pts (3, M) in
    0..1: the four lattice nodes (flat indices into (b, g, r) order, red fastest) and their weights, each (4, M)."""
    x = np.clip(pts, 0.0, 1.0) * (size - 1)
    i0 = np.minimum(np.floor(x).astype(np.int64), size - 2)
    f = x - i0
    order = np.argsort(-f, axis=0, kind="stable")  # largest fraction first
    fs = np.take_along_axis(f, order, axis=0)
    stride = np.array([1, size, size * size])  # r, g, b
    base = (i0 * stride[:, None]).sum(0)
    steps = stride[order]
    idx = np.stack([base, base + steps[0], base + steps[0] + steps[1], base + steps.sum(0)])
    return idx, np.stack([1 - fs[0], fs[0] - fs[1], fs[1] - fs[2], fs[2]])


def _fit(lat, fn, size, iters):
    """Least-squares lattice: adjust the node values so tetrahedral interpolation matches fn at 8 points per cell
    (offsets 1/4 and 3/4 of the cell on each axis), not only at the nodes. Preconditioned Landweber iterations with
    step 1 / (sum of each node's weights), which converge for interpolation weights that sum to 1; deterministic.
    Sharp, low-chroma features (a narrow skin qualifier) lose about a quarter of their worst error."""
    n = size - 1
    g = (np.arange(n)[:, None] + np.array([0.25, 0.75])[None, :]).ravel() / n
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([r.ravel(), gg.ravel(), b.ravel()])
    target = fn(pts.reshape(3, 2 * n, -1)).reshape(3, -1)
    idx, w = _tetra(size, pts)
    lat = lat.reshape(3, -1).copy()
    m = lat.shape[1]
    norm = np.maximum(sum(np.bincount(idx[k], weights=w[k], minlength=m) for k in range(4)), 1e-12)
    for _ in range(iters):
        res = target - sum(w[k] * lat[:, idx[k]] for k in range(4))
        for c in range(3):
            lat[c] += sum(np.bincount(idx[k], weights=w[k] * res[c], minlength=m) for k in range(4)) / norm
    return lat.reshape(3, size, size, size)


def bake(prog, size=65, with_output=False, fit_iters=20):
    """The lattice for a LUT. fit_iters > 0: least-squares fit for tetrahedral interpolation (_fit); 0: the function
    sampled at the nodes."""
    ok, reasons = is_lut_expressible(prog)
    if not ok:
        raise LUTError("not LUT-expressible: " + "; ".join(reasons))

    def fn(code):
        out = grade_working(code, prog, px_scale=1.0)
        return C.output_from_working(out, prog.output) if with_output else out

    lat = fn(lattice(size).reshape(3, size, size * size)).reshape(3, size, size, size)
    return _fit(lat, fn, size, fit_iters) if fit_iters else lat


def _rows(lut, order):
    """lut (3, B, G, R) -> rows of 'r g b'. order 'cube' = red fastest; 'clf' = blue fastest."""
    a = lut.transpose(1, 2, 3, 0)  # (B, G, R, 3)
    if order == "clf":
        a = a.transpose(2, 1, 0, 3)  # (R, G, B, 3): blue fastest
    return a.reshape(-1, 3)


def write_cube(path, lut, title):
    size = lut.shape[1]
    title = re.sub(r"[^ !#-~]", "?", " ".join(str(title).split()))  # one line of printable ASCII, no quote
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {size}", "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    lines += [f"{r:.7f} {g:.7f} {b:.7f}" for r, g, b in _rows(lut, "cube")]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_clf(path, lut, title, uid, in_desc, out_desc):
    size = lut.shape[1]
    rows = "\n".join(f"{r:.7f} {g:.7f} {b:.7f}" for r, g, b in _rows(lut, "clf"))
    esc = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    xml = (f'<?xml version="1.0" encoding="UTF-8"?>\n'
           f'<ProcessList xmlns="urn:AMPAS:CLF:v3.0" id="cierra-{uid}" name="{esc}" compCLFversion="3.0">\n'
           f"  <Description>{esc} (Cierra 0.1, program {uid})</Description>\n"
           f"  <InputDescriptor>{in_desc}</InputDescriptor>\n"
           f"  <OutputDescriptor>{out_desc}</OutputDescriptor>\n"
           f'  <LUT3D id="look" name="{esc}" inBitDepth="32f" outBitDepth="32f" interpolation="tetrahedral">\n'
           f'    <Array dim="{size} {size} {size} 3">\n{rows}\n    </Array>\n'
           f"  </LUT3D>\n</ProcessList>\n")
    path.write_text(xml, encoding="utf-8")


def export_luts(prog, folder, stem, size=65):
    """Write the LMT .cube + .clf and the with-output .cube; return {file: description}."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    uid = program_hash(prog)
    title = prog.name or stem
    lmt = bake(prog, size)
    files = {}
    p = folder / f"{stem}_ACEScct.cube"
    write_cube(p, lmt, f"{title} (ACEScct in/out, Cierra {uid})")
    files[p.name] = "LMT: ACEScct in -> ACEScct out"
    p = folder / f"{stem}_ACEScct.clf"
    write_clf(p, lmt, title, uid, "ACEScct (AP1)", "ACEScct (AP1)")
    files[p.name] = "LMT as CLF v3: ACEScct in -> ACEScct out"
    if prog.output.name != "acescct":
        disp = prog.output.args.get("display") or prog.output.args.get("eotf")
        p = folder / f"{stem}_ACEScct_to_{prog.output.name}_{disp}.cube"
        write_cube(p, bake(prog, size, with_output=True), f"{title} + {prog.output.name} {disp} (Cierra {uid})")
        files[p.name] = f"graph + output transform: ACEScct in -> display ({prog.output.name}, {disp}) out"
    return files


def apply_file_lut(path, code):
    """Apply a written .cube / .clf to ACEScct (3, ...) through OpenColorIO (tetrahedral): the parity check."""
    import PyOpenColorIO as ocio
    cfg = ocio.Config.CreateRaw()
    t = ocio.FileTransform(src=str(path), interpolation=ocio.INTERP_TETRAHEDRAL)
    proc = cfg.getProcessor(t).getDefaultCPUProcessor()
    return C._ocio_apply(proc, code)
