"""Replay a tuner-mode run with the tuner's damage guard off and on, holding the model's work fixed.

    python tools/replay_tuner.py --results a.json,b.json --hist '<glob with {id}>' --label NAME [--workers 8]

For each case: the run's translation (results.json) and its structure program (the last history version of the case,
found with --hist, e.g. 'out/evals/*_myrun__{id}/{id}/history'; none = the empty program), then cierra.tuner.tune with
damage_guard False ("off": the tuner as it was) and True ("on"). Writes out/evals/<UTC>_<label>_{off,on}/results.json
in eval_heldout's format, for tools/bench_notes.py. The model is not called again.
"""

import argparse
import glob
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def one(args):
    from cierra import load
    from cierra.format import format_program
    from cierra.tuner import tune
    case, hist, guard = args
    found = sorted(glob.glob(hist.format(id=case["id"])))
    versions = sorted(Path(found[-1]).glob("v*.cierra")) if found else []
    start = load(versions[-1].read_text()) if versions else load("cierra 0.1\n")
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{case['frame']}.npy").astype(np.float64)
    prog, info = tune(start, case["translated"] or "", img, damage_guard=guard)
    return {**{k: case.get(k) for k in ("id", "frame", "notes", "checks", "translated")},
            "program": format_program(prog), "structure_found": bool(versions), "guard": info.get("guard"),
            "met": None, "total": None, "damage": info.get("damage")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--hist", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cases = []
    for p in a.results.split(","):
        cases += json.loads(Path(p).read_text())["cases"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for arm, guard in (("off", False), ("on", True)):
            rows = list(ex.map(one, [(c, a.hist, guard) for c in cases]))
            out = ROOT / "out" / "evals" / f"{stamp}_{a.label}_{arm}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(json.dumps({"label": f"{a.label}_{arm}", "mode": "tuner", "replay": True,
                                                          "guard": guard, "cases": rows}, indent=1))
            print(f"{arm}: {len(rows)} cases, structure found {sum(r['structure_found'] for r in rows)}, tuner-side "
                  f"damage {sum(bool(r['damage']) for r in rows)}, refused steps "
                  f"{sum((r['guard'] or {}).get('refused_steps', 0) for r in rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
