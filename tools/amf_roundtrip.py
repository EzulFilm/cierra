"""The AMF export round trip, on the chart frames.

    python tools/amf_roundtrip.py --schema <aces-amf>/schema/acesMetadataFile.xsd [--out out/amf_roundtrip]

For each program: cierra.compile.amf.export_amf -> (1) the .amf validates against the Academy's AMF v2.0 schema
(needs the xmlschema package; github.com/ampas/aces-amf, Apache-2.0); (2) read back with resolve() - transform IDs
to OpenColorIO, the look CLF's SHA-256 checked - and rendered through OpenColorIO alone (render_amf: the look CLF,
then the ACES 2.0 display / view the Output ID names) from ACES2065-1 pixels; (3) compared with Cierra's reference
engine render of the same frame (ΔE00 on the display).
Frames: the synthetic chart (tests/data/frames/chart.npy), scene-linear Rec.709, and the same chart encoded in S-Log3
for the camera-input case (the AMF then carries the S-Log3 CSC ID and the reader decodes through it).
PASS (the AMF route) per program and frame: schema valid, hash checked, and the AMF route adds nothing to Cierra's
own CLF: ΔE00 between the AMF render and the plain ACEScct CLF render (cierra.compile.lut, same lattice) max <= 0.05.
Reported beside it, not part of that PASS: whether the LUT itself holds the program (vs the engine: p99.9 <= 0.5,
mean <= 0.1, the spec's LUT numbers). Programs: the looks in looks/, plus the programs of any --results file (eval results).
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.compile.amf import AMFError, export_amf, render_amf, resolve  # noqa: E402
from cierra.engine import is_lut_expressible, render  # noqa: E402


def stats(de):
    return {"mean": round(float(de.mean()), 4), "p99_9": round(float(np.quantile(de, 0.999)), 4),
            "max": round(float(de.max()), 4)}


EXTRA = []


def programs():
    progs = {f"look_{p.stem}": p.read_text() for p in sorted((ROOT / "looks").glob("*.cierra"))}
    for extra in EXTRA:  # --results: more programs, from eval results files
        for c in json.loads(Path(extra).read_text())["cases"]:
            progs[f"run_{c['id']}"] = c["program"]
    return progs


def one(job):
    """All rows of one program (a worker)."""
    name, src, xsd = job
    import xmlschema
    schema = xmlschema.XMLSchema(xsd)
    pkg = ROOT / "out" / "c8_amf"
    chart = np.load(ROOT / "tests" / "data" / "frames" / "chart.npy").astype(np.float64)
    slog = C.working_to_input(C.to_working(chart), "sony_slog3_sgamut3")
    ap0 = C.apply_matrix(C.matrix("rec709", "ap0"), chart)
    rows = []
    prog = load(src)
    ok, reasons = is_lut_expressible(prog)
    if not ok:
        rows.append({"program": name, "refused": "; ".join(reasons)[:200]})
        return rows
    for frame, kind, img in (("chart", "linear_rec709", chart), ("chart_slog3", "sony_slog3_sgamut3", slog)):
        p = load(re.sub(r"^input .*$", f"input {kind}", src, count=1, flags=re.M) if re.search(r"^input ", src, re.M)
                 else src.replace("cierra 0.1\n", f"cierra 0.1\ninput {kind}\n", 1))
        d = pkg / f"{name}_{frame}"
        try:
            files = export_amf(p, d, name)
        except AMFError as e:
            rows.append({"program": name, "frame": frame, "refused": str(e)[:200]})
            continue
        amf = d / f"{name}.amf"
        errors = [str(e).splitlines()[0][:200] for e in schema.iter_errors(str(amf))]
        r = resolve(amf)
        if kind == "linear_rec709":
            aces = ap0
        else:  # the reader decodes the camera log through the colour space the AMF's input ID names
            import PyOpenColorIO as ocio
            cfg = ocio.Config.CreateFromBuiltinConfig(C.ACES2_CONFIG)
            proc = cfg.getProcessor(ocio.ColorSpaceTransform(src=r["input"], dst="ACES2065-1")).getDefaultCPUProcessor()
            aces = C._ocio_apply(proc, np.ascontiguousarray(img))
        shown = render_amf(amf, np.ascontiguousarray(aces))
        ref = render(p, img, input=kind)
        eotf = C.output_eotf(p.output) or "srgb"
        s = stats(C.delta_e00_display(np.clip(shown, 0, 1), np.clip(ref, 0, 1), eotf))
        from cierra.compile.lut import apply_file_lut, export_luts
        export_luts(p, d / "plain", name)
        plain = C.output_from_working(apply_file_lut(d / "plain" / f"{name}_ACEScct.clf",
                                                     C.input_to_working(img, kind)[0]), p.output)
        added = stats(C.delta_e00_display(np.clip(shown, 0, 1), np.clip(plain, 0, 1), eotf))
        lut_ok = s["p99_9"] <= 0.5 and s["mean"] <= 0.1
        row = {"program": name, "frame": frame, "input_id": bool(r["input"]), "schema_errors": errors,
               "de00_vs_engine": s, "de00_amf_vs_plain_clf": added, "lut_in_spec": lut_ok, "files": sorted(files),
               "PASS": not errors and added["max"] <= 0.05}
        rows.append(row)
        print(f"{name:28s} {frame:12s} schema {'ok' if not errors else 'ERR ' + errors[0][:60]}  added max "
              f"{added['max']}  vs engine {s}  lut {'in spec' if lut_ok else 'OUT'}  {'PASS' if row['PASS'] else 'FAIL'}",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--results", action="append", default=[], help="an eval results.json whose programs are added")
    ap.add_argument("--out", default=str(ROOT / "out" / "amf_roundtrip"))
    a = ap.parse_args()
    EXTRA.extend(a.results)
    import xmlschema
    schema = xmlschema.XMLSchema(a.schema)
    out = Path(a.out)
    pkg = ROOT / "out" / "c8_amf"
    chart = np.load(ROOT / "tests" / "data" / "frames" / "chart.npy").astype(np.float64)
    ap0 = C.apply_matrix(C.matrix("rec709", "ap0"), chart)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "rows.jsonl"  # one row per line, written as each finishes: a stopped run resumes where it was
    done = {(r["program"], r.get("frame")) for r in map(json.loads, part.read_text().splitlines())} if part.exists() else set()
    jobs = [(n, s, a.schema) for n, s in programs().items() if (n, "chart_slog3") not in done and (n, None) not in done]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=a.workers) as ex, part.open("a") as fh:
        for got in ex.map(one, jobs):
            for row in got:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(json.dumps(row)[:220], flush=True)
    rows = [json.loads(x) for x in part.read_text().splitlines()]
    tested = [r for r in rows if "de00_vs_engine" in r]
    summary = {"programs": len({r["program"] for r in rows}), "exported": len({r["program"] for r in tested}),
               "refused": sorted({r["program"] for r in rows if "refused" in r}),
               "runs": len(tested), "schema_valid": sum(not r["schema_errors"] for r in tested),
               "pass": sum(r["PASS"] for r in tested), "lut_in_spec": sum(r["lut_in_spec"] for r in tested),
               "worst_added_max": max(r["de00_amf_vs_plain_clf"]["max"] for r in tested),
               "worst_vs_engine_p99_9": max(r["de00_vs_engine"]["p99_9"] for r in tested),
               "lut_out_of_spec": sorted({r["program"] for r in tested if not r["lut_in_spec"]})}
    out.mkdir(parents=True, exist_ok=True)
    (out / "roundtrip.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
