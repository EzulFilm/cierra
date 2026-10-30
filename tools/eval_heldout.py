"""Score a grading setup on the held-out notes x frames (tests/data/heldout.json) with the note checks.

    python tools/eval_heldout.py --mode loop   --driver openweight --label 7b_revise
    python tools/eval_heldout.py --mode tuner  --driver openweight --label 1.5b_tuner
    python tools/eval_heldout.py --mode tuner  --driver rules      --label rules_tuner
    python tools/eval_heldout.py --set freeform --mode tuner --driver openweight --label 7b_tuner_free
    python tools/eval_heldout.py --set freeform --mode translate --translator phrases --driver openweight --label t
Modes: loop      = grade_loop with revision (up to --iters rounds); best version by notes met
       tuner     = the model restates the notes in the note-check phrases (cierra.notes.translate_notes, --translator
                   axes or phrases; the rules driver only echoes), one model round for the structure (which ops /
                   masks) with the notes plus that restatement (--structure none skips it: the tuner adds rule
                   nodes), then cierra.tuner sets the numbers against the restatement
       translate = only the restatement, scored against the checks (hit / missed / reversed / extra); no render
Every tuner / translate case also records the restatement's own score ("intent").
Sets: heldout (tests/data/heldout.json, notes in the checks' own words), freeform_b (heldout_freeform_b.json: 40 more
free-form cases written blind, 2026-10-29) and freeform (heldout_freeform.json: notes
outside that vocabulary; each case's hidden "checks" are what it is scored on and are never shown to the model).
Every case is scored the same way: the final program rendered at 480 px, each check against the ungraded frame
(met = right way / ok). Writes out/evals/<UTC>_<label>/results.json and prints a table.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import load  # noqa: E402
from cierra.engine import render  # noqa: E402
from cierra.format import format_program  # noqa: E402
from cierra.history import History  # noqa: E402
from cierra.llm import grade_loop, make_provider, preview, ungraded  # noqa: E402
from cierra.notes import (compare_intents, failing, measure_notes, probes_for, regions, report,  # noqa: E402
                          translate_notes)
from cierra.qc import check_program  # noqa: E402
from cierra.tuner import tune  # noqa: E402


def score(prog, notes, img):
    small = preview(img, 480)
    base = render(ungraded(prog), small)  # same input and output as the program: only the grade differs
    probes, regs = probes_for(notes), regions(base)
    b = measure_notes(probes, base, regs)
    rows, _ = report(probes, b, b, measure_notes(probes, render(prog, small), regs))
    return len(rows) - len(failing(rows)), len(rows), rows


def run_case(case, mode, driver, iters, out, log, translator="axes", structure="model"):
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{case['frame']}.npy").astype(np.float64)
    provider = make_provider(driver)
    t0 = time.monotonic()
    hist = History(out / case["id"] / "history")
    translated = raw = None
    if mode == "loop":
        best, recs = grade_loop(case["notes"], img, provider, hist, max_iters=iters, log=log)
        tuner_info = None
    else:
        translated, raw = translate_notes(case["notes"], provider, style=translator)
        recs = [{"usage": dict(provider.last_usage)}] if driver != "rules" else []
        log(f"translated: {translated or '(nothing)'}")
        best = tuner_info = None
        if mode == "tuner" and structure == "model":
            notes = f"{case['notes']} (In short: {translated}.)" if translated else case["notes"]
            best, more = grade_loop(notes, img, provider, hist, max_iters=1, log=log, revise=False)
            recs += more
        if mode == "tuner":
            best, tuner_info = tune(best or load("cierra 0.1\n"), translated or "", img, log=log)
    seconds = time.monotonic() - t0
    prog = best or load("cierra 0.1\n")
    want = case.get("checks", case["notes"])
    intent = compare_intents(translated, want) if translated is not None else None
    if mode == "translate":
        met, total, rows = len(intent["hit"]), len(probes_for(want)), []
    else:
        met, total, rows = score(prog, want, img)
    qc = check_program(prog, img) if mode != "translate" else None
    usage = []
    for r in recs:
        u = r.get("usage") or {}
        usage += [u["first"], u["reask"]] if "first" in u else [u]
    return {"id": case["id"], "frame": case["frame"], "notes": case["notes"], "checks": case.get("checks"),
            "translated": translated, "translation_reply": raw, "intent": intent, "met": met, "total": total,
            "seconds": round(seconds, 1), "rounds": len(recs),
            "rejected": sum("errors" in r for r in recs), "reasked": sum(bool(r.get("reasked")) for r in recs),
            "capped": [c for r in recs for c in r.get("capped") or []],  # values the loop pulled inside the bounds
            "rejections": [" | ".join(r["errors"].splitlines()[:3])[:300] for r in recs if r.get("errors")],
            "grammar": any((r.get("usage") or {}).get("grammar") for r in recs),
            "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
            "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
            "tuner": tuner_info and {k: tuner_info.get(k) for k in ("evals", "added_nodes", "deficits", "clamped",
                                                                    "unmet")},
            "damage": qc and qc["flags"], "watch": qc and qc["watch"],  # watch: near misses, not flagged
            "note_checks": rows, "program": format_program(prog)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("loop", "tuner", "translate"), default="loop")
    ap.add_argument("--translator", choices=("axes", "phrases"), default="axes")
    ap.add_argument("--structure", choices=("model", "none"), default="model")
    ap.add_argument("--driver", default=None)
    ap.add_argument("--label", required=True)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--cases", nargs="*", help="case ids (default all)")
    ap.add_argument("--set", choices=("heldout", "freeform", "freeform_b"), default="heldout")
    a = ap.parse_args()
    name = {"heldout": "heldout.json", "freeform": "heldout_freeform.json",
            "freeform_b": "heldout_freeform_b.json"}[a.set]
    cases = json.loads((ROOT / "tests" / "data" / name).read_text())["cases"]
    if a.cases:
        cases = [c for c in cases if c["id"] in a.cases]
    out = ROOT / "out" / "evals" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{a.label}")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for c in cases:
        r = run_case(c, a.mode, a.driver, a.iters, out, log=lambda m: print(f"  [{c['id']}] {m}", flush=True),
                     translator=a.translator, structure=a.structure)
        results.append(r)
        print(f"{r['id']:16s} {r['met']}/{r['total']}  {r['seconds']:7.1f} s  rounds {r['rounds']}  "
              f"rejected {r['rejected']}  reasked {r['reasked']}"
              + (f"  intent hit {len(r['intent']['hit'])} reversed {len(r['intent']['reversed'])}" if r["intent"] else "")
              + ("  NEEDS HUMAN: " + "; ".join(r["damage"]) if r["damage"] else ""), flush=True)
    met, total = sum(r["met"] for r in results), sum(r["total"] for r in results)
    intents = [r["intent"] for r in results if r["intent"]]
    intent = {k: sum(len(i[k]) for i in intents) for k in ("hit", "missed", "reversed", "extra")} if intents else None
    summary = {"label": a.label, "set": a.set, "mode": a.mode, "driver": a.driver, "translator": a.translator,
               "structure": a.structure, "intent": intent, "met": met, "total": total,
               "needs_human": sum(bool(r["damage"]) for r in results),
               "seconds": round(sum(r["seconds"] for r in results), 1), "cases": results}
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"TOTAL {a.label}: {met}/{total} notes met, intent {intent}, needs human {summary['needs_human']}, "
          f"{summary['seconds']} s -> {out}")


if __name__ == "__main__":
    main()
