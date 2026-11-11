"""Replay a tuner-mode run from the model's RAW structure replies with a bounds switch off and on.

    python tools/replay_bounds.py --results a.json,b.json --replies '<glob with {id}>' --switch CIERRA_ROLLOFF \
        --label NAME [--workers 8]

tools/replay_tuner.py starts from a run's history, where the loop's bounds were already applied. A change to the
bounds themselves needs the reply as the model wrote it: for each case, the structure reply (the file the --replies
glob finds, e.g. a relay folder's '{id}/002_reply.txt'), checked, then cierra.tuner.bound and cierra.tuner.tune with
the switch "0" and "1", and the run's own translation. A reply the checker rejects gives the empty program, as in
the loop. Writes out/evals/<UTC>_<label>_{off,on}/results.json for tools/bench_notes.py. No model is called.
"""

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def one(args):
    case, replies, switch, value = args
    os.environ[switch] = value
    from cierra import load
    from cierra.errors import CierraError
    from cierra.format import format_program
    from cierra.llm import _check_reply, extract
    from cierra.tuner import bound, tune
    found = sorted(glob.glob(replies.format(id=case["id"])))
    prog, capped, state = load("cierra 0.1\n"), [], "no reply"
    if found:
        try:
            prog, _ = _check_reply(extract(Path(found[-1]).read_text())[0])
            prog, capped = bound(prog)
            state = "ok"
        except CierraError:
            state = "rejected"
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{case['frame']}.npy").astype(np.float64)
    prog, info = tune(prog, case["translated"] or "", img)
    return {**{k: case.get(k) for k in ("id", "frame", "notes", "checks", "translated")},
            "program": format_program(prog), "reply": state, "capped": capped, "damage": info.get("damage")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--switch", required=True, help="an environment switch, run with 0 and with 1")
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    cases = []
    for p in a.results.split(","):
        cases += json.loads(Path(p).read_text())["cases"]
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for arm, value in (("off", "0"), ("on", "1")):
            rows = list(ex.map(one, [(c, a.replies, a.switch, value) for c in cases]))
            out = ROOT / "out" / "evals" / f"{stamp}_{a.label}_{arm}"
            out.mkdir(parents=True, exist_ok=True)
            (out / "results.json").write_text(json.dumps({"label": f"{a.label}_{arm}", "mode": "tuner", "replay": True,
                                                          a.switch: value, "cases": rows}, indent=1))
            print(f"{arm}: {len(rows)} cases, replies ok {sum(r['reply'] == 'ok' for r in rows)}, curve points capped in "
                  f"{sum(any('points' in c for c in r['capped']) for r in rows)}, tuner-side damage "
                  f"{sum(bool(r['damage']) for r in rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
