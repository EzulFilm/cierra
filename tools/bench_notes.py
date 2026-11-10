"""Benchmark: frames x notes -> did the program move the picture the right way, without damage?

    python tools/bench_notes.py --run NAME=results.json [--run NAME=results.json ...] [--out out/bench]

A run is any results.json in the eval format (tools/eval_heldout.py, tools/relay_eval.py): per case its id, frame,
notes, hidden checks and final program. Every program is re-scored here by the same code, so runs made at different
times are compared on equal terms.

Cases: the free-form sets A (8 cases, 20 checks) and B (40 cases, 92 checks), notes written blind; frames are
Cierra's test frames: CC0 photos from raw.pixls.us (crowd, pens, window_0/1/2) and a synthetic chart
(tests/data/SOURCES.md). No other footage.

Per case:
  checks      each hidden check measured on the reference render against the ungraded render (cierra.notes):
              met = the right way by more than the check's tolerance (or "ok" for keep-clean checks)
  damage      cierra.qc: clipped highlights, crushed or milky blacks, posterisation, against the ungraded render
  side        cierra.notes.side_effects: qualities no check asks about that moved visibly ("also moved")
  outcome     clean      every check met, no damage, nothing else moved
              met        every check met, no damage, something else moved
              damaged    every check met, but the damage gate flags it
              missed     a check not met (wrong way, or not far enough)
Scores (per run):
  SUCCESS     share of cases "clean" or "met": the picture moved the right way on every check, without damage
              (the headline; side effects are reported, not penalised, until a viewing study sets their weight)
  CLEAN       share of cases "clean"
  CHECKS      checks met / checks
  DAMAGE      share of cases with a damage flag
Comparisons: paired bootstrap over cases (10 000 resamples, seed 0) of the SUCCESS difference, 95 % interval.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import load  # noqa: E402
from cierra.engine import render  # noqa: E402
from cierra.llm import preview, ungraded  # noqa: E402
from cierra.notes import failing, measure_notes, probes_for, regions, report, side_axes, side_effects  # noqa: E402
from cierra.qc import check_program  # noqa: E402

FRAMES = ROOT / "tests" / "data" / "frames"


def score_case(c):
    prog = load(c["program"])
    img = np.load(FRAMES / f"{c['frame']}.npy").astype(np.float64)
    small = preview(img, 480)
    base, now = render(ungraded(prog), small), render(prog, small)
    want = c.get("checks") or c["notes"]
    probes, regs = probes_for(want), regions(base)
    b = measure_notes(probes, base, regs)
    rows, _ = report(probes, b, b, measure_notes(probes, now, regs))
    missed = [r["note"] for r in failing(rows)]
    flags = check_program(prog, img)["flags"]
    side, _ = side_effects(side_axes(probes), base, now, regs)
    moved = [f"{s['axis']}@{s['region']}" for s in side if s["moved"]]
    outcome = "missed" if missed else "damaged" if flags else "met" if moved else "clean"
    return {"id": c["id"], "frame": c["frame"], "met": len(rows) - len(missed), "total": len(rows), "missed": missed,
            "damage": flags, "side_moved": moved, "outcome": outcome}


def summarise(rows):
    n = len(rows)
    return {"cases": n, "SUCCESS": round(sum(r["outcome"] in ("clean", "met") for r in rows) / n, 3),
            "CLEAN": round(sum(r["outcome"] == "clean" for r in rows) / n, 3),
            "CHECKS": f"{sum(r['met'] for r in rows)}/{sum(r['total'] for r in rows)}",
            "DAMAGE": round(sum(bool(r["damage"]) for r in rows) / n, 3),
            "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in ("clean", "met", "damaged", "missed")}}


def paired_bootstrap(a, b, n=10000, seed=0):
    """SUCCESS(b) - SUCCESS(a) over the same cases: (difference, 2.5 %, 97.5 %)."""
    sa = np.array([r["outcome"] in ("clean", "met") for r in a], dtype=float)
    sb = np.array([r["outcome"] in ("clean", "met") for r in b], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), (n, len(sa)))
    d = sb[idx].mean(1) - sa[idx].mean(1)
    return round(float(sb.mean() - sa.mean()), 3), round(float(np.quantile(d, 0.025)), 3), \
        round(float(np.quantile(d, 0.975)), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=path/to/results.json (several allowed)")
    ap.add_argument("--out", default=str(ROOT / "out" / "bench"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        cases = []
        for p in path.split(","):
            cases += json.loads(Path(p).read_text())["cases"]
        runs[name] = sorted((score_case(c) for c in cases), key=lambda r: r["id"])
        print(name, json.dumps(summarise(runs[name])), flush=True)
    names = list(runs)
    ids = [set(r["id"] for r in runs[n]) for n in names]
    common = set.intersection(*ids)
    comp = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            ra = [r for r in runs[x] if r["id"] in common]
            rb = [r for r in runs[y] if r["id"] in common]
            comp[f"{y} - {x}"] = paired_bootstrap(ra, rb)
    res = {"runs": {n: {"summary": summarise(r), "cases": r} for n, r in runs.items()},
           "paired_success_difference_95": comp, "common_cases": len(common)}
    (out / "bench.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(comp))


if __name__ == "__main__":
    main()
