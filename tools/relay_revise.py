"""One revision round after a scored run: does the model revise from feedback?

For every case of out/relay/<label>/results.json with a check not met, the model gets the loop's own revision text
(cierra.notes.revision_text: the notes not met, the node causing each WRONG WAY, what to change or add), the
colourist's original notes, the final program and its measurements, through the relay driver; its reply is checked,
bounded (cierra.tuner.bound) and scored on the same hidden checks with the same damage gate.
A revision is CORRECT when every check that failed is now met, no check that was met is lost, and no new damage flag
appears. The feedback names the failing checks in their own words: this is the colourist saying "still not cooler",
which a real session gives; the checks that were met are not listed.

    python tools/relay_revise.py start --label myrun --set freeform_b
    python tools/relay_revise.py score --label myrun
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def one(label, cid):
    from cierra import load
    from cierra.engine import render
    from cierra.format import format_program
    from cierra.llm import RelayProvider, _check_reply, compact_stats, language_reference, preview, ungraded, extract
    from cierra import analysis
    from cierra.notes import attribute, failing, measure_notes, probes_for, regions, revision_text, report
    from cierra.qc import check_program
    from cierra.tuner import bound
    from eval_heldout import score
    from cierra.errors import CierraError
    base = ROOT / "out" / "relay" / label
    res = json.loads((base / "results.json").read_text())
    case = next(c for c in res["cases"] if c["id"] == cid)
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{case['frame']}.npy").astype(np.float64)
    prog = load(case["program"])
    small = preview(img, 480)
    rf = lambda p: render(p, small)  # noqa: E731
    want = case["checks"]
    probes, regs = probes_for(want), regions(rf(ungraded(prog)))
    b = measure_notes(probes, rf(ungraded(prog)), regs)
    now = measure_notes(probes, rf(prog), regs)
    rows, _ = report(probes, b, b, now)
    causes = attribute(prog, probes, rf, regs, now)
    text, _, _ = revision_text(probes, rows, causes)
    if case["damage"]:  # the loop's damage gate feedback, as grade_loop gives it in the next round
        text = (text + "\n" if text else "") + "Damage gate (needs a human): " + "; ".join(case["damage"]) + \
            ". Fix this without losing the notes already met."
    out = rf(prog)
    fb = {"stats_now": analysis.stats(out), "measure_now": analysis.measure(out, None)}
    if case["damage"]:
        fb["damage"] = case["damage"]
    user = "\n".join([text, "", f"Notes from the colourist: {case['notes']}",
                      "Input colour space: linear_rec709. Revision round.",
                      "Host context (use only what is listed here): " + json.dumps({"input": "linear_rec709"}),
                      "Current program:", "```cierra", format_program(prog).strip(), "```",
                      "Measurements of the current program's preview (numbers only):", json.dumps(fb, indent=1, sort_keys=True),
                      "Write the next version of the program."])
    prov = RelayProvider(base / "revise" / cid)
    rec = {"id": cid, "before_met": case["met"], "total": case["total"], "before_damage": case["damage"],
           "failing_before": [r["note"] for r in failing(case["note_checks"])]}
    errors = None
    for attempt in (1, 2):  # one re-ask after a checker rejection, as the loop does
        reply = prov.complete(language_reference(), user + (f"\nYour previous answer was rejected by the checker; fix these:\n{errors}" if errors else ""))
        src, msg = extract(reply)
        try:
            p2, _ = _check_reply(src)
            p2, capped = bound(p2, start=prog)
            break
        except CierraError as e:
            errors, p2 = str(e), None
    if p2 is None:
        rec.update(after_met=None, correct=False, rejected=errors)
    else:
        met, total, rows2 = score(p2, want, img)
        qc = check_program(p2, img)
        lost = [r["note"] for r in failing(rows2) if r["note"] not in rec["failing_before"]]
        fixed = [n for n in rec["failing_before"] if n not in [r["note"] for r in failing(rows2)]]
        new_dmg = [f for f in qc["flags"] if f.split(":")[0] not in [d.split(":")[0] for d in case["damage"] or []]]
        rec.update(after_met=met, fixed=fixed, lost=lost, after_damage=qc["flags"], new_damage=new_dmg,
                   damage_cleared=not qc["flags"],
                   correct=(len(fixed) == len(rec["failing_before"]) and not lost and not new_dmg
                            and (not case["damage"] or not qc["flags"] or bool(rec["failing_before"]))),
                   capped=capped, message=msg, program=format_program(p2))
    (base / "revise" / cid / "result.json").write_text(json.dumps(rec, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "score", "one"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--case")
    ap.add_argument("--damaged", action="store_true", help="also revise cases met in full but flagged by the gate")
    a = ap.parse_args()
    base = ROOT / "out" / "relay" / a.label
    if a.cmd == "one":
        return one(a.label, a.case)
    res = json.loads((base / "results.json").read_text())
    todo = [c["id"] for c in res["cases"] if c["met"] < c["total"] or (a.damaged and c["damage"])]
    if a.cmd == "start":
        assert not (base / "revise").exists() or not any((base / "revise").iterdir()), "revise/ exists"
        (base / "revise").mkdir(exist_ok=True)
        for cid in todo:
            (base / "revise" / cid).mkdir(exist_ok=True)
            subprocess.Popen([sys.executable, __file__, "one", "--label", a.label, "--case", cid], cwd=ROOT,
                             stdout=open(base / "revise" / f"{cid}.log", "w"), stderr=subprocess.STDOUT)
        print(f"revision round for {len(todo)} cases: {todo}")
    else:
        recs = [json.loads((base / "revise" / c / "result.json").read_text()) for c in todo
                if (base / "revise" / c / "result.json").exists()]
        ok = sum(r["correct"] for r in recs)
        miss = [r for r in recs if r["failing_before"]]
        dmg = [r for r in recs if not r["failing_before"]]
        summ = {"label": a.label, "misses": len(todo), "scored": len(recs), "correct": ok,
                "check_misses": len(miss), "check_misses_correct": sum(r["correct"] for r in miss),
                "damage_only": len(dmg), "damage_only_cleared": sum(r["correct"] for r in dmg),
                "before_met": sum(r["before_met"] for r in recs), "after_met": sum(r["after_met"] or 0 for r in recs),
                "cases": recs}
        (base / "revise" / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
        print(f"{a.label} revision: {ok} of {len(todo)} revised correctly ({len(recs)} scored); check misses "
              f"{summ['check_misses_correct']}/{len(miss)}, damage-only cleared {summ['damage_only_cleared']}/{len(dmg)}")


if __name__ == "__main__":
    main()
