"""Print and save (docs/targets.json) ΔE00 of every compiled target against the reference engine on the test frames,
and write the looks' DCTL / CUDA / CLF / cube files to out/.

    python tools/target_report.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.checker import load_file  # noqa: E402
from cierra.compile.cuda import compile_cuda  # noqa: E402
from cierra.compile.cuda_cpu import run_cuda  # noqa: E402
from cierra.compile.dctl import compile_dctl  # noqa: E402
from cierra.compile.dctl_cpu import run_dctl  # noqa: E402
from cierra.compile.lut import apply_file_lut, export_luts, LUTError  # noqa: E402
from cierra.engine import grade_working, make_ctx, run_graph  # noqa: E402
from test_targets import KITCHEN_SINK  # noqa: E402

FR = ROOT / "tests" / "data" / "frames"


def de_stats(prog, a, b):
    de = C.delta_e00_display(C.output_from_working(a, prog.output), C.output_from_working(b, prog.output)).flatten()
    return {"mean": round(float(de.mean()), 4), "p95": round(float(np.quantile(de, 0.95)), 4),
            "max": round(float(de.max()), 4)}


def worst(rows):
    return {k: max(r[k] for r in rows) for k in ("mean", "p95", "max")}


def main():
    frames = {p.stem: C.to_working(np.load(p).astype(np.float64)) for p in sorted(FR.glob("*.npy"))}
    out_dir = ROOT / "out"
    report = {}
    progs = {"kitchen_sink": load(KITCHEN_SINK)}
    progs.update({n: load_file(ROOT / "looks" / f"{n}.cierra") for n in ("cinematic_modern", "beauty_modern", "teal_night")})
    for name, prog in progs.items():
        rec = {}
        # the C-like targets run on the CPU: DCTL through gcc, the CUDA kernel through g++ (cuda_cpu.py)
        for key, compile_fn, run_fn, ext in (("dctl", compile_dctl, run_dctl, "dctl"), ("cuda", compile_cuda, run_cuda, "cu")):
            parts = compile_fn(prog, frame=5 if name == "kitchen_sink" else 0, split=True)
            (out_dir / key).mkdir(parents=True, exist_ok=True)
            for p in parts:
                if p[key]:
                    (out_dir / key / f"{name}{'' if p['name'] == 'main' else '_' + p['name']}.{ext}").write_text(p[key])
            rows = []
            for code in frames.values():
                ref = grade_working(code, prog, frame=5 if name == "kitchen_sink" else 0, px_scale=code.shape[-1] / prog.reference_width)
                ctx = make_ctx(prog, code.shape[-1], frame=5 if name == "kitchen_sink" else 0)
                o = code
                for p in parts:
                    o = run_fn(p[key], o) if p[key] else run_graph(o, p["graph"], prog, ctx)
                rows.append(de_stats(prog, o, ref))
            rec[key] = {"worst_frame": worst(rows), "parts": [p["note"] for p in parts]}
        try:
            files = export_luts(prog, out_dir / "lut", name)
            for ext in ("cube", "clf"):
                rows = []
                for code in frames.values():
                    c = np.clip(code, 0, 1)
                    rows.append(de_stats(prog, apply_file_lut(out_dir / "lut" / f"{name}_ACEScct.{ext}", c),
                                         grade_working(c, prog, px_scale=1.0)))
                rec[ext] = {"worst_frame": worst(rows)}
            rec["lut_files"] = files
        except LUTError as e:
            rec["lut"] = f"refused: {e}"
        report[name] = rec
        print(name, json.dumps(rec, indent=1))
    (ROOT / "docs" / "targets.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
