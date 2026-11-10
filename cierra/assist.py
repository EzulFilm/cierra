"""The checked route between an assistant and the grading application.

An assistant that writes a raw DCTL into a grading application hands over unverifiable code. This route hands over
a checked program, a compiled DCTL, and measurements that say what it did:

    notes -> program -> [G1 checker + bounds] -> [G2 damage gate] -> [G3 note checks, reference engine]
          -> [G4 DCTL compile, or refusal with the reason] -> [G5 pre-flight: the DCTL itself run on the frame]
          -> package (DCTL, apply script, reference stills, record with hashes)
          -> host applies it in a new colour version, rolls back on any failure (cierra.hosts.grading_app)
          -> [G6 host check: the host's own stills against the engine, pixel parity AND the notes re-measured]
          -> record: verified | contradiction | unverified, re-checkable from its files (verify_record)

prepare() runs G1-G5 and writes the package; host_check() runs G6 on what the host sent back. Nothing here runs code
a model wrote: the model's output is a program in a closed language, the DCTL is generated from it, and the host
script is a fixed template filled with validated data (grading_app.apply_script).

The program can come from any driver (cierra.llm: Anthropic, open-weight, relay, rules) or be written by hand;
prepare() takes the program and the checks (the notes in the note-check phrases, cierra.notes.translate_notes).
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from . import colour as C
from .checker import load
from .errors import CierraError
from .format import format_program, program_hash

PREFLIGHT_MAX = 0.5  # ΔE00, the DCTL parity requirement (spec §11)


def _sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def _de_stats(de):
    return {"mean": round(float(de.mean()), 5), "p99": round(float(np.quantile(de, 0.99)), 5),
            "max": round(float(de.max()), 5)}


def note_rows(probes, regs, base, now):
    """The note checks of `now` (a display image) against `base`: [{note, verdict, ungraded, now}]."""
    from .notes import measure_notes, report
    b = measure_notes(probes, base, regs)
    rows, _ = report(probes, b, b, measure_notes(probes, now, regs))
    return [{"note": r["note"], "verdict": r["vs_ungraded"], "ungraded": r["ungraded"], "now": r["now"]}
            for r in rows]


def _met(rows):
    return sum(r["verdict"] in ("ok", "right way") for r in rows)


def prepare(program, image, checks, folder, stem="grade", input=None, frame=0, project="acescct",
            host_still=r"C:\Cierra\stills\{stem}_{i}.tif", timecodes=("01:00:00:00",), project_name="cierra-test"):
    """G1-G5 on one frame and the package for the host. program: source text or a checked Program; image: (3, H, W)
    in the program's input (or `input`); checks: the notes in check phrases ("warmer, less contrast"); project:
    "acescct" (a colour-managed ACEScct project: the DCTL is ACEScct in / out and the host's output transform makes
    the picture). Only ACEScct projects for now: a Rec.709 project needs the wrapped DCTL (cierra.compile.dctl_wrap)
    and its own pre-flight, not built yet. Returns the record (also written to folder/record.json)."""
    from .compile.dctl import DCTLError, compile_dctl
    from .engine import render
    from .hosts.grading_app import apply_script, version_name
    from .llm import ungraded
    from .notes import probes_for, regions, unmeasured
    from .qc import check_program
    from .tuner import bound
    if project != "acescct":
        raise ValueError(f"project {project!r}: only ACEScct colour-managed projects so far")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"stem": stem, "checks": checks, "project": project, "frame": frame, "gates": {}, "files": {}}

    def stop(status, why):
        rec.update(status=status, reason=why)
        (folder / "record.json").write_text(json.dumps(rec, indent=1, default=str))
        return rec

    # G1: the checker (parse, types, ranges, size caps) and the loop's bounds
    try:
        prog = load(program) if isinstance(program, str) else program
        prog, capped = bound(prog)
    except CierraError as e:
        rec["gates"]["G1_checker"] = {"ok": False, "errors": str(e).splitlines()[:20]}
        return stop("rejected", "the checker rejected the program")
    rec["gates"]["G1_checker"] = {"ok": True, "capped": capped}
    src = format_program(prog)
    rec["program_hash"] = program_hash(prog)
    img = np.asarray(image, dtype=np.float64)
    inp = input or prog.input

    # G2: damage gate (clipping, crushing, milky blacks, posterisation) against the same program without its graph
    qc = check_program(prog, img, input=inp, width=img.shape[-1])
    rec["gates"]["G2_damage"] = {"ok": not qc["flags"], "flags": qc["flags"], "watch": qc["watch"]}

    # G3: the notes, measured on the reference engine's render
    base, graded = render(ungraded(prog), img, input=inp, frame=frame), render(prog, img, input=inp, frame=frame)
    probes, regs = probes_for(checks), regions(base)
    rows = note_rows(probes, regs, base, graded)
    rec["gates"]["G3_notes"] = {"met": _met(rows), "total": len(rows), "rows": rows,
                                "not_measurable": unmeasured(checks)}

    # G4: compile, or refuse with the reason (never approximate)
    try:
        parts = compile_dctl(prog, frame=frame, split=False, stem=stem)
    except DCTLError as e:
        rec["gates"]["G4_compile"] = {"ok": False, "refused": str(e)}
        return stop("refused", f"the DCTL target cannot hold this program: {e}")
    # one DCTL, or a two-pass pair (wide sharpen, local contrast, highlights / shadows) for two consecutive nodes
    dctls = [p["dctl"] for p in parts]
    rec["gates"]["G4_compile"] = {"ok": True, "nodes": len(dctls), "note": "; ".join(p["note"] for p in parts)}

    # G5: pre-flight: the generated DCTL itself, run on this frame (CPU harness), against the engine, and the notes
    # measured on its output must come out the same
    code, _ = C.input_to_working(img, inp)
    shown = C.output_from_working(run_chain(dctls, code), prog.output)
    eotf = C.output_eotf(prog.output) or "srgb"
    de = C.delta_e00_display(np.clip(shown, 0, 1), np.clip(graded, 0, 1), eotf)
    pre_rows = note_rows(probes, regs, base, shown)
    same = [a["verdict"] == b["verdict"] for a, b in zip(rows, pre_rows)]
    rec["gates"]["G5_preflight"] = {"ok": float(de.max()) <= PREFLIGHT_MAX and all(same), "de00": _de_stats(de),
                                    "verdicts_same": f"{sum(same)}/{len(same)}"}
    if not rec["gates"]["G5_preflight"]["ok"]:
        return stop("stopped", "pre-flight: the compiled DCTL does not match the engine on this frame")

    # package: what goes to the host, and what its stills must match
    names = [f"{stem}.dctl"] if len(dctls) == 1 else [f"{stem}_node{i}.dctl" for i in range(1, len(dctls) + 1)]
    files = {f"{stem}.cierra": src, **dict(zip(names, dctls))}
    for name, text in files.items():
        (folder / name).write_text(text)
    np.save(folder / "reference_0.npy", np.asarray(graded, dtype=np.float32))
    np.save(folder / "ungraded_0.npy", np.asarray(base, dtype=np.float32))
    files["reference_0.npy"] = (folder / "reference_0.npy").read_bytes()
    files["ungraded_0.npy"] = (folder / "ungraded_0.npy").read_bytes()
    stills = [(tc, host_still.format(stem=stem, i=i)) for i, tc in enumerate(timecodes)]
    version = version_name(prog)
    if len(dctls) == 1:
        script = apply_script(project_name, version, f"Cierra/{stem}.dctl", stills)
    else:  # node for node (SetLUT per node, rollback), verified there by an exported LUT; stills: not yet (G6)
        from .hosts.grading_app import export_nodes_script
        script = export_nodes_script(project_name, version, [{"index": i, "kind": "dctl", "file": n, "label": n}
                                                             for i, n in enumerate(names, 1)])
        rec["host_note"] = ("two DCTL nodes: the node-for-node script does not export stills yet, so a live host "
                            "check (G6) would come back unverified until it does")
    (folder / "apply_in_host.py").write_text(script)
    files["apply_in_host.py"] = script
    rec["files"] = {k: _sha(v) for k, v in files.items()}
    rec.update(version=version, eotf=eotf, stills=stills, dctl_files=names,
               status="needs_human" if qc["flags"] else "ready",
               reason="the damage gate flagged it: a person looks before it goes to the host" if qc["flags"] else "")
    (folder / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def run_chain(dctls, code):
    """ACEScct (3, H, W) through one DCTL, or a two-pass pair (the first returns RGB + the alpha the second reads)."""
    from .compile.dctl_cpu import run_dctl
    out = np.asarray(code, dtype=np.float32)
    for d in dctls:
        out = run_dctl(d, np.asarray(out, dtype=np.float32))
    return out[:3]


def host_check(folder, result, stills):
    """G6, on what the host sent back: result = the apply script's JSON, stills = its display stills (3, H, W) in the
    order asked. Pixel parity (grading_app.verify) and the notes re-measured on the host's own pixels, which must give
    the same verdicts as the engine. Updates folder/record.json and returns it."""
    from .hosts.grading_app import verify
    from .notes import probes_for, regions
    folder = Path(folder)
    rec = json.loads((folder / "record.json").read_text())
    refs = [np.load(folder / "reference_0.npy").astype(np.float64)]
    base = np.load(folder / "ungraded_0.npy").astype(np.float64)
    par = verify(result, stills, refs, eotf=rec.get("eotf", "srgb"), requested=len(refs))
    g6 = {"parity": par}
    if par["status"] == "verified":
        probes, regs = probes_for(rec["checks"]), regions(base)
        host_rows = note_rows(probes, regs, base, np.clip(np.asarray(stills[0], dtype=np.float64), 0, 1))
        engine = rec["gates"]["G3_notes"]["rows"]
        diff = [(a["note"], a["verdict"], b["verdict"]) for a, b in zip(engine, host_rows) if a["verdict"] != b["verdict"]]
        g6["notes"] = {"met": _met(host_rows), "total": len(host_rows), "rows": host_rows, "differ": diff}
        status = "contradiction" if diff else "verified"
        why = ("the host's picture gives different note verdicts: " + "; ".join(f"{n}: engine {a}, host {b}"
                                                                           for n, a, b in diff)) if diff else ""
    else:
        status, why = par["status"], par["reason"]
    rec["gates"]["G6_host"] = g6
    rec.update(host_status=status, host_reason=why)
    (folder / "record.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def verify_record(folder):
    """Re-check a package from its files: every hash in record.json against the file, and the program re-checked
    (it must still load and hash to the recorded program_hash). Returns (ok, [problems])."""
    folder = Path(folder)
    rec = json.loads((folder / "record.json").read_text())
    bad = []
    for name, h in rec.get("files", {}).items():
        p = folder / name
        if not p.exists():
            bad.append(f"{name}: missing")
        elif _sha(p.read_bytes()) != h:
            bad.append(f"{name}: changed since the record was written")
    stem = rec["stem"]
    try:
        if program_hash(load((folder / f"{stem}.cierra").read_text())) != rec.get("program_hash"):
            bad.append("program: hash differs from the record")
    except (CierraError, OSError) as e:
        bad.append(f"program: {e}")
    return not bad, bad
