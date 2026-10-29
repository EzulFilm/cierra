"""Re-run only the tuner on a finished eval's stored translations (no model, no network): the same structure-free
pipeline as `eval_heldout.py --mode tuner --structure none`, so a tuner change can be measured on the model's real
answers without another CPU run. Writes out/evals/<UTC>_<label>/results.json like eval_heldout.

    python tools/retune_eval.py out/evals/<run> --label retuned
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from cierra import load  # noqa: E402
from cierra.format import format_program  # noqa: E402
from cierra.qc import check_program  # noqa: E402
from cierra.tuner import tune  # noqa: E402
from eval_heldout import score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()
    src = json.loads((Path(a.run) / "results.json").read_text())
    out = ROOT / "out" / "evals" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{a.label}")
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for c in src["cases"]:
        img = np.load(ROOT / "tests" / "data" / "frames" / f"{c['frame']}.npy").astype(np.float64)
        prog, info = tune(load("cierra 0.1\n"), c["translated"] or "", img)
        met, total, rows = score(prog, c.get("checks") or c["notes"], img)
        dmg = check_program(prog, img)["flags"]
        cases.append({**{k: c[k] for k in ("id", "frame", "notes", "checks", "translated", "intent")},
                      "met": met, "total": total, "note_checks": rows, "damage": dmg, "program": format_program(prog),
                      "tuner": {k: info.get(k) for k in ("evals", "added_nodes", "deficits", "unmet")}})
        print(f"{c['id']:16s} {met}/{total}" + ("  NEEDS HUMAN: " + "; ".join(dmg) if dmg else ""), flush=True)
    met, total = sum(c["met"] for c in cases), sum(c["total"] for c in cases)
    summary = {"label": a.label, "from": src["label"], "met": met, "total": total,
               "needs_human": sum(bool(c["damage"]) for c in cases), "cases": cases}
    (out / "results.json").write_text(json.dumps(summary, indent=1, default=str))
    print(f"TOTAL {a.label}: {met}/{total} met, needs human {summary['needs_human']} -> {out}")


if __name__ == "__main__":
    main()
