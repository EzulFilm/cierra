"""Write Cierra's golden frames (tests/golden/cierra) and the per-op / DCTL / analysis goldens. Where the original
engine's reference renders are present (tests/golden/origin/, not distributed), also the parity table (docs/parity.json).
Run after an intended engine or look change, and say why in the commit.

    python tools/make_golden.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cierra import colour as C  # noqa: E402
from conftest import FRAMES  # noqa: E402
from test_looks import ORIGIN, ORIGIN_STEP, GOLDEN, GOLDEN_STEP, LOOK_NAMES, cierra_render  # noqa: E402


def _parity(out, ref):
    de = C.delta_e00_display(out, np.load(ref).astype(np.float64))
    return {"mean": round(float(de.mean()), 4), "p95": round(float(np.quantile(de, 0.95)), 4),
            "max": round(float(de.max()), 4)}


def main():
    GOLDEN.mkdir(parents=True, exist_ok=True)
    table = {look: {} for look in LOOK_NAMES}
    cases = [p.stem for p in sorted(FRAMES.glob("*.npy"))] + ["chart_photo"]
    for look in LOOK_NAMES:
        for case in cases:
            full = cierra_render(look, case)
            np.save(GOLDEN / f"{look}__{case}.npy", full[:, ::GOLDEN_STEP, ::GOLDEN_STEP].astype(np.float32))
            ref = ORIGIN / f"{look}__{case}.npy"
            if ref.exists():
                table[look][case] = row = _parity(full[:, ::ORIGIN_STEP, ::ORIGIN_STEP], ref)
                print(f"{look:18s} {case:22s} dE00 mean {row['mean']:.4f} p95 {row['p95']:.4f} max {row['max']:.4f}")
    if ORIGIN.is_dir():
        (ROOT / "docs" / "parity.json").write_text(json.dumps(table, indent=1, sort_keys=True) + "\n")
    from regression_cases import write_all
    write_all()  # per-op goldens, DCTL text snapshots, analysis snapshot
    print("wrote tests/golden/cierra, ops.json, dctl/ and analysis_beauty_crowd.json")


if __name__ == "__main__":
    main()
