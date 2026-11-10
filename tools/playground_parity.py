"""Playground parity: the web playground in a real browser (headless Chromium, WebGPU) against the reference engine.

    python tools/playground_parity.py [--out out/playground_parity]     (needs the playwright package + Chromium)

For every program x chart frame, the page grades the frame exactly as a user's browser does (the program's WGSL
shader, then the output transform as a 65^3 LUT) and hands back both buffers. Measured against cierra.engine:
  shader : the WGSL output (ACEScct) vs grade_working, both through the program's output transform in Python, as
           ΔE00 (spec §11: the WGSL target's stated error is ΔE00 <= 0.5 max; the repo's tests also hold mean <= 0.05)
  display: what the page shows (shader + LUT) vs the reference render, ΔE00 (the spec's numbers for an output LUT:
           <= 0.3 mean and <= 0.5 at the 99.9th percentile)
PASS when every program the WGSL target accepts meets both on both chart frames. Programs the target refuses are listed
with the reason (the playground shows the same refusal).
"""

import argparse
import base64
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))

from cierra import colour as C  # noqa: E402
from cierra import load  # noqa: E402
from cierra.engine import grade_working  # noqa: E402
import playground_server as PS  # noqa: E402

ARGS = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--ignore-gpu-blocklist"]
FRAMES = ("chart", "chart_photo")
SHADER_MAX, SHADER_MEAN, DISP_MEAN, DISP_P999 = 0.5, 0.05, 0.3, 0.5


def programs():
    from test_targets import KEYS_PROGRAM, KITCHEN_SINK, PASS_PROGRAM
    app = (ROOT / "web" / "playground" / "app.js").read_text()
    default = app.split("const DEFAULT = `")[1].split("`;")[0]
    progs = {"playground_default": default, "pass": PASS_PROGRAM, "keys": KEYS_PROGRAM, "kitchen_sink": KITCHEN_SINK,
             "rec1886_output": default.replace("aces2(display: srgb)", "aces2(display: rec1886)"),
             "display_encode_output": default.replace("aces2(display: srgb)", "display_encode(eotf: srgb)")}
    for look in ("cinematic_modern", "beauty_modern", "teal_night"):
        progs[f"look_{look}"] = (ROOT / "looks" / f"{look}.cierra").read_text()
    return progs


def stats(de):
    return {"mean": round(float(de.mean()), 5), "p99_9": round(float(np.percentile(de, 99.9)), 5),
            "max": round(float(de.max()), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", help="a folder from tools/build_playground_static.py: serve it with a plain file "
                    "server (no /api), so the page compiles in the browser (Pyodide)")
    ap.add_argument("--out", default=str(ROOT / "out" / "playground_parity"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.static:
        import functools
        from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

        class Quiet(SimpleHTTPRequestHandler):
            def log_message(self, *args):
                pass
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=a.static))
    else:
        httpd = PS.serve("127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    from playwright.sync_api import sync_playwright
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=ARGS)
        page = browser.new_page()
        console = []
        page.on("console", lambda m: console.append(m.text))
        page.goto(url)
        assert page.evaluate("window.cierraPlaygroundReady"), page.inner_text("#status")
        mode = page.evaluate("document.getElementById('status').textContent")
        adapter = page.evaluate("""async () => { const a = await navigator.gpu.requestAdapter();
            const i = a.info || {}; return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(' ') || 'unnamed'; }""")
        ua = page.evaluate("navigator.userAgent")
        for pname, src in programs().items():
            prog = load(src)
            for fname in FRAMES:
                r = page.evaluate("([s, f]) => window.cierraPlayground.run(s, f)", [src, fname])
                row = {"program": pname, "frame": fname}
                if not r["ok"]:
                    row.update(refused=r["status"].strip()[:400])
                    rows.append(row)
                    print(f"{pname:24s} {fname:12s} REFUSED: {row['refused'][:100]}", flush=True)
                    continue
                w, h = r["width"], r["height"]
                code = np.frombuffer(base64.b64decode(r["code"]), np.float32).reshape(h, w, 3).transpose(2, 0, 1)
                disp = np.frombuffer(base64.b64decode(r["disp"]), np.float32).reshape(h, w, 3).transpose(2, 0, 1)
                path, space = PS.BUILTIN[fname]
                src_code, _ = PS.working_frame(prog, PS.downscale(np.load(path).astype(np.float64)), space)
                ref = grade_working(src_code.astype(np.float64), prog, px_scale=w / prog.reference_width)
                eotf = C.output_eotf(prog.output) or "srgb"
                ref_disp = C.output_from_working(ref, prog.output)
                de_shader = C.delta_e00_display(C.output_from_working(code.astype(np.float64), prog.output), ref_disp, eotf)
                de_disp = C.delta_e00_display(disp.astype(np.float64), ref_disp, eotf)
                s1, s2 = stats(de_shader), stats(de_disp)
                ok1 = s1["max"] <= SHADER_MAX and s1["mean"] <= SHADER_MEAN
                ok2 = s2["mean"] <= DISP_MEAN and s2["p99_9"] <= DISP_P999
                row.update(size=[w, h], output=r["output"], shader_de00=s1, display_de00=s2,
                           shader_cv_abs_max=round(float(np.abs(code - ref).max()), 6),
                           shader_pass=ok1, display_pass=ok2)
                rows.append(row)
                print(f"{pname:24s} {fname:12s} shader {s1}  display {s2}  {'PASS' if ok1 and ok2 else 'FAIL'}",
                      flush=True)
        browser.close()
    httpd.shutdown()
    graded = [r for r in rows if "refused" not in r]
    summary = {"mode": "static (Pyodide, no server)" if a.static else "server", "date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "browser": ua, "adapter": adapter,
               "thresholds": {"shader_de00_max": SHADER_MAX, "shader_de00_mean": SHADER_MEAN,
                              "display_de00_mean": DISP_MEAN, "display_de00_p99_9": DISP_P999},
               "graded": len(graded), "refused": len(rows) - len(graded),
               "shader_pass": sum(r["shader_pass"] for r in graded), "display_pass": sum(r["display_pass"] for r in graded),
               "worst_shader_max": max(r["shader_de00"]["max"] for r in graded),
               "worst_display_p99_9": max(r["display_de00"]["p99_9"] for r in graded),
               "worst_display_mean": max(r["display_de00"]["mean"] for r in graded),
               "PASS": all(r["shader_pass"] and r["display_pass"] for r in graded) and bool(graded),
               "rows": rows, "console_errors": [c for c in console if "rror" in c][:20]}
    (out / "parity.json").write_text(json.dumps(summary, indent=1))
    print(f"PASS={summary['PASS']} graded {len(graded)}, shader pass {summary['shader_pass']}, display pass "
          f"{summary['display_pass']}, refused {summary['refused']}; worst shader max ΔE00 {summary['worst_shader_max']}, "
          f"worst display p99.9 {summary['worst_display_p99_9']} -> {out / 'parity.json'}")


if __name__ == "__main__":
    main()
