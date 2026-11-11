"""Look packs with proofs: each look as a pack: its source, a compiled DCTL and LUT per camera log, the measured
maximum error of every file against the reference engine, and a damage report on a standard frame set.

    python tools/look_packs.py [--looks-dir looks] [--out out/packs] [--sheets out/packs/sheets] [--looks a b ...]

Standard frame set: the CC0 test frames (crowd, pens, window_0/1/2: raw.pixls.us CC0 photos; chart: synthetic),
scene-linear Rec.709. For a camera log they are encoded into that log through OpenColorIO's ACES 2.0 studio config
(the inverse of the input transform the engine uses), so every file is measured on its own kind of input.

Per input kind, the files:
  DCTL  a Rec.709 project node: decodes the input, grades, encodes for the display with the look's own output
        (cierra.compile.dctl_wrap); for "acescct": an ACES project node (ACEScct in / out, the project's ACES output)
  LUT   65^3 .cube, input code values 0..1 -> display, tetrahedral, least-squares fitted; only for looks with no
        spatial or position node (a LUT cannot hold grain, glow, vignette...): refused with the reason otherwise
  LMT   for "acescct": the ACEScct in / out .cube and .clf
Error = ΔE00 of the file's result against the engine's render of the same frame (the program with `input <kind>`),
max / p99 / mean over all frames; the pack states the max. Damage = cierra.qc on each frame (scene-linear input),
with the look's own stated shadow lift allowed (cierra.qc.intended_black_lift). One sheet (PNG) per pack.
"""

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.format import format_program, program_hash  # noqa: E402

LOOKS = ROOT / "looks"  # --looks-dir changes it (set in each worker through the job tuple)
FRAMES = ["crowd", "pens", "window_0", "window_1", "window_2", "chart"]
DCTL_LOGS = ["sony_slog3_sgamut3", "sony_slog3_sgamut3cine", "arri_logc3", "arri_logc4", "panasonic_vlog",
             "apple_log", "dji_dlog"]
LUT_ONLY_LOGS = ["canon_log2", "canon_log3", "red_log3g10", "bmd_film_gen5"]
KINDS = ["acescct", "rec1886_photo", *DCTL_LOGS, *LUT_ONLY_LOGS]


def source(look, looks_dir=None):
    return Path(looks_dir or LOOKS) / f"{look}.cierra"


def encode(lin709, kind):
    """Scene-linear Rec.709 (3, H, W) -> code values of `kind` (the inverse of colour.input_to_working)."""
    return C.working_to_input(C.to_working(lin709), kind)


def with_input(src, kind):
    return re.sub(r"^input .*$", f"input {kind}", src, count=1, flags=re.M) if re.search(r"^input ", src, re.M) \
        else src.replace("cierra 0.1\n", f"cierra 0.1\ninput {kind}\n", 1)


def stats(de):
    return {"max": round(float(de.max()), 4), "p99": round(float(np.quantile(de, 0.99)), 4),
            "mean": round(float(de.mean()), 4)}


def lut_for(prog, kind, size=65):
    """input code 0..1 lattice -> display, fitted for tetrahedral interpolation (cierra.compile.lut._fit)."""
    from cierra.compile.lut import LUTError, _fit, lattice
    from cierra.engine import grade_working, is_lut_expressible
    ok, reasons = is_lut_expressible(prog)
    if not ok:
        raise LUTError("not LUT-expressible: " + "; ".join(reasons))

    def fn(code):
        w, disp = C.input_to_working(code, kind)
        out = C.output_from_working(grade_working(w, prog, px_scale=1.0, display_referred=disp), prog.output)
        # lattice corners far outside any picture (a log code of 1.0 decodes to hundreds of stops of light) can
        # overflow to inf / NaN in display_encode's soft clip; a display LUT holds 0..1 anyway, and one NaN node
        # would spread through the least-squares fit
        return np.clip(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    lat = fn(lattice(size).reshape(3, size, size * size)).reshape(3, size, size, size)
    return _fit(lat, fn, size, 20)


def one_kind(args):
    """All files of one look x input kind, measured on the frame set. Runs in a worker."""
    look, kind, outdir, looks_dir = args
    from cierra.compile.dctl import DCTLError, compile_dctl
    from cierra.compile.dctl_cpu import run_dctl
    from cierra.compile.lut import LUTError, apply_file_lut, write_cube
    from cierra.engine import render
    src = source(look, looks_dir).read_text()
    prog = load(with_input(src, kind))
    eotf = C.output_eotf(prog.output) or "srgb"
    d = Path(outdir) / look / kind
    d.mkdir(parents=True, exist_ok=True)
    imgs = {f: encode(np.load(ROOT / "tests" / "data" / "frames" / f"{f}.npy").astype(np.float64), kind)
            for f in FRAMES}
    refs = {f: render(prog, im, input=kind) for f, im in imgs.items()}
    res = {"look": look, "kind": kind}
    # DCTL
    if kind in ("acescct", "rec1886_photo", *DCTL_LOGS):
        try:
            if kind == "acescct":
                parts = compile_dctl(prog, split=False, stem=f"{look}_{kind}")
            else:
                parts = compile_dctl(prog, split=False, input=kind, output="program", stem=f"{look}_{kind}")
            names = [f"{look}_{kind}.dctl"] if len(parts) == 1 else [f"{look}_{kind}_node{i}.dctl"
                                                                    for i in range(1, len(parts) + 1)]
            luts = {}
            for n, p in zip(names, parts):
                (d / n).write_text(p["dctl"])
                for lname, lat in (p.get("luts") or {}).items():
                    write_cube(d / lname, np.asarray(lat), f"{look} output for {kind}")
                    luts[lname] = lat
            des = []
            for f, im in imgs.items():
                out = np.asarray(im, dtype=np.float32)
                for p in parts:
                    out = run_dctl(p["dctl"], np.asarray(out, dtype=np.float32), luts=p.get("luts"))
                out = out[:3]
                shown = C.output_from_working(out, prog.output) if kind == "acescct" else out
                des.append(C.delta_e00_display(np.clip(shown, 0, 1), np.clip(refs[f], 0, 1), eotf).ravel())
            res["dctl"] = {"files": names + sorted(luts), "nodes": len(parts), **stats(np.concatenate(des))}
        except DCTLError as e:
            res["dctl"] = {"refused": str(e)[:300]}
    # LUT (camera log / video in -> display out); for acescct: the LMT with the output transform baked in
    try:
        lut = lut_for(prog, kind)
        name = f"{look}_{kind}_to_{prog.output.name}.cube"
        write_cube(d / name, lut, f"{look}: {kind} in -> display out (Cierra {program_hash(prog)})")
        des = []
        for f, im in imgs.items():
            shown = apply_file_lut(d / name, np.clip(im, 0, 1))
            des.append(C.delta_e00_display(np.clip(shown, 0, 1), np.clip(refs[f], 0, 1), eotf).ravel())
        res["lut"] = {"files": [name], **stats(np.concatenate(des))}
    except LUTError as e:
        res["lut"] = {"refused": str(e)[:300]}
    if kind == "acescct":  # the LMT: ACEScct in / out .cube + .clf, the project's output transform after it
        from cierra.compile.lut import bake, write_clf
        try:
            lmt = bake(prog, 65)
            name = f"{look}_LMT_ACEScct.cube"
            write_cube(d / name, lmt, f"{look} LMT, ACEScct in / out (Cierra {program_hash(prog)})")
            write_clf(d / name.replace(".cube", ".clf"), lmt, look, program_hash(prog), "ACEScct (AP1)", "ACEScct (AP1)")
            des = []
            for f, im in imgs.items():
                shown = C.output_from_working(apply_file_lut(d / name, im), prog.output)
                des.append(C.delta_e00_display(np.clip(shown, 0, 1), np.clip(refs[f], 0, 1), eotf).ravel())
            res["lmt"] = {"files": [name, name.replace(".cube", ".clf")], **stats(np.concatenate(des))}
        except LUTError as e:
            res["lmt"] = {"refused": str(e)[:300]}
    return res


def damage(args):
    look, looks_dir = args
    from cierra.engine import render
    from cierra.qc import check_program
    from cierra.qc import intended_black_lift
    path = source(look, looks_dir)
    prog = load(path.read_text())
    lift = round(intended_black_lift(prog), 4)  # the look's own print black, stated in the pack
    rows = {}
    for f in FRAMES:
        img = np.load(ROOT / "tests" / "data" / "frames" / f"{f}.npy").astype(np.float64)
        q = check_program(prog, img, width=img.shape[-1], allow={"milky": lift})
        q0 = check_program(prog, img, width=img.shape[-1])
        rows[f] = {"flags": q["flags"], "watch": q["watch"], "flags_without_allowance": q0["flags"]}
    thumbs = {f: render(prog, np.load(ROOT / "tests" / "data" / "frames" / f"{f}.npy").astype(np.float64)[:, ::2, ::2],
                        px_scale=0.5) for f in ("window_0", "pens", "chart")}  # sheet pictures: no people
    plain = load("cierra 0.1\noutput aces2(display: srgb)\n")
    befores = {f: render(plain, np.load(ROOT / "tests" / "data" / "frames" / f"{f}.npy").astype(np.float64)[:, ::2, ::2])
               for f in thumbs}
    return {"look": look, "source": path.name, "output": str(prog.output.name),
            "program_hash": program_hash(prog), "intended_black_lift": lift, "frames": rows,
            "flagged_frames": sum(bool(r["flags"]) for r in rows.values())}, thumbs, befores


def sheet(look, dmg, kinds, thumbs, befores, png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10.5), dpi=120)
    fig.text(0.5, 0.975, f"Cierra look pack · {look} · {dmg['program_hash']} · output {dmg['output']}",
             ha="center", fontsize=13, weight="bold")
    for i, f in enumerate(thumbs):
        for j, (img, lab) in enumerate(((befores[f], "ACES 2.0, no look"), (thumbs[f], look))):
            ax = fig.add_axes([0.02 + i * 0.33, 0.70 - j * 0.0, 0.155, 0.24]) if j == 0 else \
                fig.add_axes([0.18 + i * 0.33, 0.70, 0.155, 0.24])
            ax.imshow(np.clip(np.asarray(img).transpose(1, 2, 0), 0, 1))
            ax.set_title(f"{f}: {lab}", fontsize=8)
            ax.axis("off")
    rows = []
    for k in kinds:
        dc, lu = k.get("dctl"), k["lut"]
        cell = (lambda x: "refused" if x is None else ("refused: " + x["refused"].split(":")[-1].strip()[:38]
                                                         if "refused" in x else f"{x['max']:.4f} / {x['p99']:.4f}"))
        rows.append([k["kind"], "-" if dc is None else cell(dc), cell(lu)])
    ax = fig.add_axes([0.02, 0.22, 0.6, 0.45])
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=["input", "DCTL ΔE00 max / p99", "LUT ΔE00 max / p99"], loc="upper center",
                 cellLoc="left", colWidths=[0.3, 0.33, 0.37])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.25)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_facecolor("#2b3a55")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
    ax2 = fig.add_axes([0.64, 0.22, 0.34, 0.45])
    ax2.axis("off")
    import textwrap
    drows = [[f, "\n".join(textwrap.wrap("; ".join(r["flags"]), 52)[:3]) or "clean"] for f, r in dmg["frames"].items()]
    t2 = ax2.table(cellText=drows, colLabels=["frame", "damage gate"], loc="upper center", cellLoc="left",
                   colWidths=[0.22, 0.78])
    t2.auto_set_font_size(False)
    t2.set_fontsize(8)
    t2.scale(1, 2.6)
    for (r, c), cell in t2.get_celld().items():
        cell.set_edgecolor("#bbb")
        if r == 0:
            cell.set_facecolor("#2b3a55")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif c == 1:
            cell.get_text().set_color("#1a7a1a" if cell.get_text().get_text() == "clean" else "#a33")
    dmax = max((k["dctl"]["max"] for k in kinds if k.get("dctl") and "max" in k["dctl"]), default=None)
    lmax = max((k["lut"]["max"] for k in kinds if "max" in k["lut"]), default=None)
    lm = next((k.get("lmt") for k in kinds if k["kind"] == "acescct"), None) or {}
    fig.text(0.02, 0.195, "LMT (ACEScct in / out .cube + .clf, the ACES output after it): " + (
        f"ΔE00 max {lm['max']:.4f} / p99 {lm['p99']:.4f}" if "max" in lm else "refused (" + lm.get("refused", "-")
        .split(":")[-1].strip()[:60] + ")"), fontsize=10)
    if all(k.get("dctl") is None or "refused" in k["dctl"] for k in kinds):
        why = next((k["dctl"]["refused"] for k in kinds if k.get("dctl") and "refused" in k["dctl"]), "")
        node = re.search(r"node (\w+): (\w+)\(\)", why)
        fig.text(0.02, 0.22, "DCTL refuses" + (f" ({node.group(2)}() in node {node.group(1)} is wider than a DCTL can hold)"
                 if node else "") + ": it needs a host plugin to run in a grading application.", fontsize=11, color="#a33",
                 weight="bold")
    fig.text(0.02, 0.145, f"Stated black lift (the look's own shadow toe on a neutral ramp, display luma): "
             f"{dmg['intended_black_lift']:.3f}; the damage gate allows it on top of the usual +0.04 for milky blacks.",
             fontsize=9, color="#444")
    fig.text(0.02, 0.17, f"Stated max error over 6 CC0 frames x {len(kinds)} inputs: DCTL {dmax if dmax is not None else 'n/a'} "
             f"ΔE00, LUT {lmax if lmax is not None else 'n/a (refused)'} ΔE00. Damage gate: "
             f"{dmg['flagged_frames']} of {len(dmg['frames'])} frames flagged.", fontsize=10)
    fig.text(0.02, 0.12, "Inputs encoded from scene-linear CC0 frames through OpenColorIO's ACES 2.0 config; DCTL = Rec.709"
             " project node (decode, grade, encode), acescct = ACES project node; LUT = 65^3 input -> display, tetrahedral.",
             fontsize=8, color="#444")
    fig.text(0.02, 0.10, "Measured against Cierra's reference engine.", fontsize=8, color="#444")
    fig.savefig(png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--looks-dir", default=str(LOOKS), help="a folder of .cierra looks (default: looks/)")
    ap.add_argument("--out", default=str(ROOT / "out" / "packs"))
    ap.add_argument("--sheets", default=None, help="where the sheets go (default: <out>/sheets)")
    ap.add_argument("--looks", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sheets-only", action="store_true", help="redraw the sheets from the packs' PACK.json")
    a = ap.parse_args()
    if a.sheets_only:
        out, sheets = Path(a.out), Path(a.sheets or Path(a.out) / "sheets")
        looks = a.looks or sorted(p.parent.name for p in out.glob("*/PACK.json"))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            dmgs = list(ex.map(damage, [(lk, a.looks_dir) for lk in looks]))
        for look, (dmg, thumbs, befores) in zip(looks, dmgs):
            rec = json.loads((out / look / "PACK.json").read_text())
            sheet(look, dmg, rec["inputs"], thumbs, befores, sheets / f"{look}.png")
        return
    looks = a.looks or sorted(p.stem for p in Path(a.looks_dir).glob("*.cierra"))
    out, sheets = Path(a.out), Path(a.sheets or Path(a.out) / "sheets")
    sheets.mkdir(parents=True, exist_ok=True)
    jobs = [(lk, k, str(out), a.looks_dir) for lk in looks for k in KINDS]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(one_kind, jobs))
        dmgs = list(ex.map(damage, [(lk, a.looks_dir) for lk in looks]))
    summary = []
    for look, (dmg, thumbs, befores) in zip(looks, dmgs):
        kinds = [r for r in results if r["look"] == look]
        (out / look / f"{look}.cierra").write_text(format_program(load(source(look, a.looks_dir).read_text())))
        rec = {**dmg, "inputs": kinds,
               "stated_max_de00": {"dctl": max((k["dctl"]["max"] for k in kinds if k.get("dctl") and "max" in k["dctl"]),
                                               default=None),
                                   "lut": max((k["lut"]["max"] for k in kinds if "max" in k["lut"]), default=None)}}
        (out / look / "PACK.json").write_text(json.dumps(rec, indent=1))
        sheet(look, dmg, kinds, thumbs, befores, sheets / f"{look}.png")
        summary.append({k: rec[k] for k in ("look", "output", "program_hash", "flagged_frames",
                                            "stated_max_de00")}
                       | {"lmt": next((k.get("lmt") for k in kinds if k["kind"] == "acescct"), None),
                          "dctl_refused": [k["kind"] for k in kinds if k.get("dctl") and "refused" in k["dctl"]],
                          "lut_refused": bool(all("refused" in k["lut"] for k in kinds)),
                          "lut_refusal": next((k["lut"]["refused"] for k in kinds if "refused" in k["lut"]), "")[:160]})
        print(json.dumps(summary[-1]), flush=True)
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
