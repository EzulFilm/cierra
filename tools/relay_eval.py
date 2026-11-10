"""Run tools/eval_heldout.py with the relay driver, one process per case, so every case's prompt of the same round
waits at the same time and can be answered as a batch. Then merge the per-case results into one results.json.

    python tools/relay_eval.py start --set freeform_b --label c1_b --model "assistant chat"
    python tools/relay_eval.py pending --label c1_b          # prompts without a reply
    python tools/relay_eval.py merge --label c1_b

Same protocol as the 7B baseline (tuner mode, axes translator, structure round); only the driver differs."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETS = {("heldout" if p.stem == "heldout" else p.stem.removeprefix("heldout_")): p.name
        for p in sorted((ROOT / "tests" / "data").glob("heldout*.json"))}  # the note sets that are present


def ids(name):
    return [c["id"] for c in json.loads((ROOT / "tests" / "data" / SETS[name]).read_text())["cases"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "pending", "merge"))
    ap.add_argument("--label", required=True)
    ap.add_argument("--set", default="freeform_b", choices=tuple(SETS))
    ap.add_argument("--model", default="relay")
    ap.add_argument("--mode", default="tuner")
    ap.add_argument("--code-root", default=str(ROOT), help="the tree whose tools/eval_heldout.py runs (a frozen worktree)")
    ap.add_argument("--relay-root", help="where the per-case prompt / reply folders go (default: next to the logs). Give "
                    "the answering chat this folder only: logs and results stay under out/relay/<label>")
    a = ap.parse_args()
    base = ROOT / "out" / "relay" / a.label
    code = Path(a.code_root).resolve()
    relay = Path(a.relay_root).expanduser().resolve() if a.relay_root else base
    if a.cmd == "start":
        base.mkdir(parents=True, exist_ok=True)
        for cid in ids(a.set):
            env = {**os.environ, "CIERRA_RELAY_DIR": str(relay / cid), "CIERRA_RELAY_MODEL": a.model}
            log = open(base / f"{cid}.log", "w")
            subprocess.Popen([sys.executable, str(code / "tools" / "eval_heldout.py"), "--set", a.set, "--mode", a.mode,
                              "--translator", "axes", "--driver", "relay", "--label", f"{a.label}__{cid}",
                              "--cases", cid], env=env, stdout=log, stderr=subprocess.STDOUT, cwd=code)
        print(f"started {len(ids(a.set))} cases under {base}")
    elif a.cmd == "pending":
        for d in sorted(p for p in relay.iterdir() if p.is_dir()):
            for pr in sorted(d.glob("*_prompt.txt")):
                if not (d / pr.name.replace("_prompt", "_reply")).exists():
                    print(pr)
    else:
        cases = []
        for cid in ids(a.set):
            runs = sorted((code / "out" / "evals").glob(f"*_{a.label}__{cid}/results.json"))
            if not runs:
                print("missing", cid)
                continue
            cases += json.loads(runs[-1].read_text())["cases"]
        met, total = sum(c["met"] for c in cases), sum(c["total"] for c in cases)
        intents = [c["intent"] for c in cases if c["intent"]]
        intent = {k: sum(len(i[k]) for i in intents) for k in ("hit", "missed", "reversed", "extra")}
        summary = {"label": a.label, "set": a.set, "mode": a.mode, "driver": "relay", "model": a.model,
                   "translator": "axes", "structure": "model", "intent": intent, "met": met, "total": total,
                   "needs_human": sum(bool(c["damage"]) for c in cases), "cases": cases}
        (base / "results.json").write_text(json.dumps(summary, indent=1, default=str))
        print(f"{a.label}: {met}/{total}, intent {intent}, needs human {summary['needs_human']} -> {base}/results.json")


if __name__ == "__main__":
    main()
