"""Bench of the checked route (cierra/assist.py) on the programs of eval runs, with a simulated host: no grading
application.

    python tools/route_bench.py --results a.json,b.json [--out out/route_bench]

Programs: the final programs of the given results files (tools/eval_heldout.py or tools/relay_eval.py format), each
with its case's checks and frame. For each: prepare() (G1-G5 and the package), then a simulated host and 8 faulty hosts, then
host_check() (G6) on each, then verify_record() and a one-byte tamper.

The simulated host does what the apply script makes the grading application do in an ACEScct colour-managed
project: the clip's input transform to ACEScct, the package's DCTL on node 1 (run by the CPU harness: the same C
source), the project's output transform, a 16-bit still. Faults (each must NOT come back "verified" when it changes
the picture beyond the verification limits; a fault that leaves the picture within them is counted as not
observable, not as a pass):
  no_readback    the host never read the LUT path back            -> unverified
  not_applied    the still is the ungraded picture
  applied_twice  the DCTL on two nodes
  wrong_dctl     another program's DCTL on the node (a fixed cool / desaturating grade)
  wrong_frame    the playhead 6 px / frame off (the still is shifted)
  bad_pixel      one pixel off by 0.08 code in an unclipped area
  wrong_output   the project's output transform is display_encode instead of ACES 2.0
  half_res       the still at half resolution                     -> unverified
PASS numbers (no grading application):
  P1 every program is packaged ("ready" / "needs_human") or stopped with a reason: no crash, no silent approximation
  P2 pre-flight: every compiled DCTL within ΔE00 max 0.5 of the engine on its frame, note verdicts identical
  P3 honest host: "verified" for every packaged case, notes re-measured on the host still identical
  P4 faulty hosts: 0 observable faults come back "verified"
  P5 records: every package re-verifies from its files; a one-byte change in the DCTL is caught in every package
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.assist import host_check, prepare, verify_record  # noqa: E402
from cierra.hosts.sim import FAULTS, host, observable  # noqa: E402,F401

OTHER = ("cierra 0.1\ninput linear_rec709\noutput aces2(display: srgb)\n"
         "node c = white_balance(temperature: -0.25)\nnode d = saturation(amount: 0.6)\ngraph c -> d\n")


def one(c):
    """Everything for one case (run in a worker process)."""
    from cierra.compile.dctl import compile_dctl
    other_dctl = compile_dctl(load(OTHER))[0]["dctl"]
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{c['frame']}.npy").astype(np.float64)
    folder = ROOT / "out" / "c2_packages" / c["id"]
    row = {"id": c["id"], "set": c["set"]}
    try:
        rec = prepare(c["program"], img, c["checks"], folder, stem=c["id"])
    except Exception as e:  # P1 counts crashes
        row.update(status="CRASH", reason=f"{type(e).__name__}: {e}")
        return row
    row.update(status=rec["status"], reason=rec.get("reason", ""), gates={k: {kk: vv for kk, vv in v.items()
               if kk not in ("rows", "watch")} for k, v in rec["gates"].items()})
    if rec["status"] not in ("ready", "needs_human"):
        return row
    prog = load((folder / f"{c['id']}.cierra").read_text())
    ref = np.load(folder / "reference_0.npy").astype(np.float64)
    honest = host_check(folder, *host(folder, img, prog))
    g6 = honest["gates"]["G6_host"].get("notes", {})
    row["honest"] = {"status": honest["host_status"], "reason": honest["host_reason"],
                     "notes": f"{g6.get('met')}/{g6.get('total')}"}
    row["faults"] = {}
    for f in FAULTS:
        res, st = host(folder, img, prog, f, other_dctl=other_dctl)
        r = host_check(folder, res, st)
        row["faults"][f] = {"status": r["host_status"], "observable": observable(st[0], ref) or f == "no_readback"}
    host_check(folder, *host(folder, img, prog))  # leave the honest result in the record
    ok, bad = verify_record(folder)
    p = folder / json.loads((folder / "record.json").read_text())["dctl_files"][0]
    orig = p.read_bytes()
    p.write_bytes(orig[:-2] + bytes([orig[-2] ^ 1]) + orig[-1:])
    tampered_ok, _ = verify_record(folder)
    p.write_bytes(orig)
    row["record"] = {"reverifies": ok, "problems": bad, "tamper_caught": not tampered_ok}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="eval results.json files, comma-separated")
    ap.add_argument("--out", default=str(ROOT / "out" / "route_bench"))
    a = ap.parse_args()
    out = Path(a.out)
    pkg_root = ROOT / "out" / "c2_packages"
    shutil.rmtree(pkg_root, ignore_errors=True)
    cases = []
    for path in a.results.split(","):
        cases += [dict(c, set=Path(path).parent.name) for c in json.loads(Path(path).read_text())["cases"]]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=16) as ex:
        rows = list(ex.map(one, cases))
    for r in rows:
        print(f"{r['id']:26s} {r['status']:12s} " + (
            f"pre-flight max {r['gates']['G5_preflight']['de00']['max']}  host {r['honest']['status']}  faults "
            + " ".join(f"{k}={v['status'][:5]}{'' if v['observable'] else '(n/o)'}" for k, v in r["faults"].items())
            if "honest" in r else r.get("reason", "")), flush=True)
    n = len(rows)
    pk = [r for r in rows if "honest" in r]
    stopped = [r for r in rows if r["status"] in ("refused", "stopped", "rejected")]
    fault_runs = [(r["id"], f, v) for r in pk for f, v in r["faults"].items()]
    obs = [x for x in fault_runs if x[2]["observable"]]
    escaped = [x for x in obs if x[2]["status"] == "verified"]
    pre = [r["gates"]["G5_preflight"] for r in rows if "G5_preflight" in r.get("gates", {})]
    summary = {
        "programs": n, "packaged": len(pk), "needs_human": sum(r["status"] == "needs_human" for r in rows),
        "stopped_with_reason": len(stopped), "crashes": sum(r["status"] == "CRASH" for r in rows),
        "P1": n == len(pk) + len(stopped) and all(r["reason"] for r in stopped),
        "preflight_worst_max_de00": max((p["de00"]["max"] for p in pre), default=None),
        "preflight_verdicts_same": f"{sum(int(p['verdicts_same'].split('/')[0]) for p in pre)}/"
                                   f"{sum(int(p['verdicts_same'].split('/')[1]) for p in pre)}",
        "P2": all(p["ok"] for p in pre),
        "honest_verified": sum(r["honest"]["status"] == "verified" for r in pk),
        "P3": all(r["honest"]["status"] == "verified" for r in pk) and bool(pk),
        "fault_runs": len(fault_runs), "observable": len(obs), "not_observable": len(fault_runs) - len(obs),
        "escaped": len(escaped), "escaped_list": escaped,
        "fault_outcomes": {f: {s: sum(1 for r in pk if r["faults"][f]["status"] == s and r["faults"][f]["observable"])
                               for s in ("verified", "contradiction", "unverified")} for f in FAULTS},
        "P4": not escaped,
        "records_reverify": sum(r["record"]["reverifies"] for r in pk),
        "tamper_caught": sum(r["record"]["tamper_caught"] for r in pk),
        "P5": all(r["record"]["reverifies"] and r["record"]["tamper_caught"] for r in pk),
    }
    summary["PASS"] = all(summary[k] for k in ("P1", "P2", "P3", "P4", "P5"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "escaped_list"}, indent=1))


if __name__ == "__main__":
    main()
