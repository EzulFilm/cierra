"""A recorded grading-loop run: notes + a test frame -> versions, costs and a before / after sheet.

    CIERRA_MODEL=claude-opus-5-5 CIERRA_BUDGET_USD=2 python tools/live_run.py [--driver anthropic|rules|auto]

Writes out/live_runs/<UTC time>_<driver>/: history/ (every version + log.jsonl with diffs and usage), summary.json
(driver, model, iterations, tokens, cost, final program) and sheet.png (neutral | each accepted version). The key
comes only from the environment (ANTHROPIC_API_KEY); nothing here reads, stores or prints it.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import load  # noqa: E402
from cierra.engine import render  # noqa: E402
from cierra.format import format_program  # noqa: E402
from cierra.history import History  # noqa: E402
from cierra.llm import grade_loop, make_provider, preview  # noqa: E402

NOTES = ("Warmer faces, less green in the trees, a little more contrast, keep the blacks clean and the highlights "
         "soft.")
FRAME = "crowd"  # a crowd outdoors, raw.pixls.us CC0 (tests/data/SOURCES.md): faces, skin, trees, sky


def tile(d, w=320):
    a = (np.clip(d, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    return im.resize((w, round(w * im.height / im.width)), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", default=None)
    ap.add_argument("--notes", default=NOTES)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--frame", default=FRAME, help="a tests/data/frames name")
    a = ap.parse_args()
    provider = make_provider(a.driver)
    driver = type(provider).__name__.replace("Provider", "").lower()
    out = ROOT / "out" / "live_runs" / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"_{driver}")
    out.mkdir(parents=True, exist_ok=True)
    img = np.load(ROOT / "tests" / "data" / "frames" / f"{a.frame}.npy").astype(np.float64)
    hist = History(out / "history")
    t0 = time.monotonic()
    best, recs = grade_loop(a.notes, img, provider, hist, max_iters=a.iters, width=480)
    seconds = time.monotonic() - t0
    usage = [r.get("usage", {}) for r in recs]
    summary = {
        "driver": driver, "model": getattr(provider, "model", "rules"), "notes": a.notes, "frame": a.frame,
        "budget_usd": float(os.environ.get("CIERRA_BUDGET_USD", 2.0)), "iterations": len(recs),
        "accepted": sum(bool(r.get("accepted")) for r in recs), "rejected_by_checker": sum("errors" in r for r in recs),
        "input_tokens": sum(u.get("input_tokens", 0) for u in usage),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usage),
        "cache_read_tokens": sum(u.get("cache_read_tokens", 0) for u in usage),
        "cost_usd": round(sum(u.get("cost_usd", 0.0) for u in usage), 4), "seconds": round(seconds, 1),
        "records": recs, "final_program": format_program(best) if best else None,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))

    small = preview(img, 480)
    tiles = [("neutral", tile(render(load("cierra 0.1\n"), small)))]
    for e in hist.entries():
        tiles.append((f"v{e['id']} {e['hash']}", tile(render(hist.load(e), small))))
    th = tiles[0][1].height
    sheet = Image.new("RGB", (len(tiles) * 330 + 10, th + 40), "white")
    dr = ImageDraw.Draw(sheet)
    for i, (label, t) in enumerate(tiles):
        sheet.paste(t, (10 + i * 330, 30))
        dr.text((10 + i * 330, 8), label, fill="black")
    sheet.save(out / "sheet.png")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("records", "final_program")}, indent=1))
    print(out)


if __name__ == "__main__":
    main()
