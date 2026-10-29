"""Sheet of held-out results for judgement by eye: per case, ungraded | one column per eval run (final program).

    python tools/eval_sheet.py out/evals/<run_a> out/evals/<run_b> ... --out out/heldout_sheet.png
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import load  # noqa: E402
from cierra.engine import render  # noqa: E402
from cierra.llm import preview  # noqa: E402

W = 320


def tile(d):
    a = (np.clip(d, 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    return Image.fromarray(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="out/heldout_sheet.png")
    a = ap.parse_args()
    runs = [json.loads((Path(r) / "results.json").read_text()) for r in a.runs]
    cases = [c["id"] for c in runs[0]["cases"]]
    rows = []
    for cid in cases:
        per = [next(c for c in r["cases"] if c["id"] == cid) for r in runs]
        img = preview(np.load(ROOT / "tests" / "data" / "frames" / f"{per[0]['frame']}.npy").astype(np.float64), W)
        tiles = [(tile(render(load("cierra 0.1\n"), img)), f"{cid}: ungraded")]
        tiles += [(tile(render(load(c["program"]), img)), f"{r['label']}  {c['met']}/{c['total']}")
                  for r, c in zip(runs, per)]
        rows.append((per[0]["notes"], tiles))
    th = rows[0][1][0][0].height
    sheet = Image.new("RGB", (W * len(rows[0][1]), len(rows) * (th + 36)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (notes, tiles) in enumerate(rows):
        y = j * (th + 36)
        draw.text((4, y + 2), notes, fill="black")
        for i, (t, label) in enumerate(tiles):
            sheet.paste(t, (i * W, y + 18))
            draw.text((i * W + 4, y + 20 + th - 2), label, fill="black")
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
