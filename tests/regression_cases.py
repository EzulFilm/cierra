"""Regression cases shared by tests/test_regression.py (checks) and tools/make_golden.py (writes the goldens).

- OP_CASES: every op and mask with non-trivial parameters, applied to a small chart; the golden holds per-channel
  means and 24 fixed sample pixels (ACEScct), so any change to an op's maths shows up by op name.
- DCTL snapshots: the generated DCTL text of every shipped look and the every-op program, byte for byte.
- ANALYSIS: the numbers the LLM sees for a fixed render (catches changes to the statistics it is prompted with).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "tests" / "golden"

H = "cierra 0.1\noutput acescct()\n"
OP_CASES = {
    "exposure": "node a = exposure(stops: 0.7)",
    "white_balance": "node a = white_balance(temperature: 0.3, tint: -0.2)",
    "printer_lights": "node a = printer_lights(red: 3, green: -1, blue: -4)",
    "wheels": "node a = wheels(lift: [0.02, 0, -0.02], gamma: [-0.01, 0.01, 0], gain: [0, -0.02, 0.03], "
              "offset: [0.005, 0, 0], master_gamma: 0.02)",
    "cdl": "node a = cdl(slope: [1.1, 0.95, 1], offset: [-0.02, 0, 0.01], power: [0.9, 1, 1.2], sat: 1.2)",
    "contrast": "node a = contrast(amount: 1.3, pivot: 0.38)",
    "saturation": "node a = saturation(amount: 1.4, protect_skin: 0.6)",
    "curve_luma": "node a = curve(points: [[0.0729, 0.0729], [0.3, 0.27], [0.5, 0.53], [1, 0.98]])",
    "curve_rgb": "node a = curve(points: [[0, 0.02], [0.45, 0.43], [1, 1]], channel: rgb)",
    "curve_blue": "node a = curve(points: [[0, 0], [0.5, 0.47], [1, 1]], channel: blue)",
    "curve_display": "node a = curve(points: [[0, 0.03], [0.5, 0.52], [1, 0.97]], space: display)",
    "hue_vs_hue": "node a = hue_vs_hue(points: [[206, -20, 40], [123, 10]])",
    "hue_vs_sat": "node a = hue_vs_sat(points: [[350, 1.4, 50], [173, 0.6]])",
    "hue_vs_lum": "node a = hue_vs_lum(points: [[300, 0.05, 60]])",
    "lum_vs_sat": "node a = lum_vs_sat(points: [[0.2, 0.5], [0.45, 1.3], [0.7, 0.6]])",
    "sat_vs_sat": "node a = sat_vs_sat(points: [[0, 0], [0.04, 0.06], [0.1, 0.09], [0.3, 0.15]])",
    "sat_vs_lum": "node a = sat_vs_lum(points: [[0, 0], [0.05, -0.02], [0.15, -0.06]])",
    "rgb_mixer": "node a = rgb_mixer(red: [0.9, 0.1, 0], green: [0.05, 0.9, 0.05], blue: [0, 0.2, 0.8], "
                 "preserve_luma: true)",
    "density": "node a = density(amount: 1.2)",
    "hue_rotate": "node a = hue_rotate(degrees: 25)",
    "vibrance": "node a = vibrance(amount: 0.6)",
    "split_tone": "node a = split_tone(shadow_amount: 0.02, highlight_amount: 0.015, balance: 0.05)",
    "gamut_compress": "node a = gamut_compress(power: 1.5)",
    "vignette": "node a = vignette(amount: 0.3, roundness: 0.3, softness: 0.8)",
    "grain": "node a = grain(amount: 0.6, size_px: 2, chroma: 0.7, seed: 99)",
    "glow": "node a = glow(threshold: 0.5, radius_px: 8, strength: 0.8, warm: 0.2)",
    "halation": "node a = halation(threshold: 0.45, radius_px: 6, strength: 1.0)",
    "diffusion": "node a = diffusion(radius_px: 10, strength: 0.6)",
    "blur": "node a = blur(radius_px: 3)",
    "sharpen": "node a = sharpen(amount: 1.5, radius_px: 2, coring: 0.002)",
    "local_contrast": "node a = local_contrast(amount: 0.6, radius_px: 20)",
    "highlights_shadows": "node a = highlights_shadows(highlights: -1.5, shadows: 1.2, radius_px: 15)",
    "guided_detail": "node a = guided_detail(amount: 1.0, radius_px: 6, edge: 0.03)",
    "mask_qualifier": "node a = exposure(stops: 1)\n  mask qualifier(hue: 100..180, sat: 0.01..0.3, lum: 0.2..0.8)",
    "mask_qualifier_wrap": "node a = exposure(stops: 1)\n  mask qualifier(hue: 300..60, soft: 0.1)",
    "mask_luma_blur": "node a = exposure(stops: 1)\n  mask luma(range: 0.3..0.6, soft: 0.05, blur_px: 4)",
    "mask_window_feather": "node a = exposure(stops: 1)\n  mask window(radius: [0.3, 0.3], soft: 0.1, feather_px: 6)",
    "mask_skin": "node a = saturation(amount: 2)\n  mask skin(width: 40)",
    "mask_box": "node a = exposure(stops: 1)\n  mask box(centre: [0.45, 0.55], size: [0.3, 0.2], angle: 20, roundness: 0.3, "
                "soft: 0.15)",
    "mask_polygon": "node a = exposure(stops: 1)\n  mask polygon(points: [[0.1, 0.2], [0.8, 0.15], [0.5, 0.5], "
                    "[0.85, 0.85], [0.15, 0.8]], soft: 0.03)",
    "mask_rgb_key":"node a = exposure(stops: 1)\n  mask rgb_key(red: 0.35..0.7, blue: 0.2..0.5, soft: 0.05)",
    "mask_clip": "node a = exposure(stops: 1)\n  mask skin(width: 40, clip_black: 0.2, clip_white: 0.8)",
    "mask_window": "node a = exposure(stops: 1)\n  mask window(centre: [0.3, 0.6], radius: [0.2, 0.3], soft: 0.5)",
    "mask_algebra": "node a = exposure(stops: 1)\n  mask (luma(range: 0.2..0.5) or skin()) and not window()",
    "mask_window_rotated": "node a = exposure(stops: 1)\n"
                           "  mask window(centre: [0.4, 0.5], radius: [0.35, 0.15], angle: 30, soft: 0.4)",
    "mask_graduated": "node a = exposure(stops: -1)\n  mask graduated(centre: [0.5, 0.4], angle: 100, falloff: [0.1, 0.3])",
    "parallel": "node a = exposure(stops: 1)\nnode b = saturation(amount: 0.3)",
    "layer_add": "node a = exposure(stops: -1)\nnode b = contrast(amount: 1.5)",
    "keys_smooth": "node a = exposure(stops: keys(smooth, 0: -1, 10: 1, 30: 0))",
}
GRAPHS = {"parallel": "parallel(a, b)", "layer_add": "layer(a, b, mode: add, opacity: 0.4)"}
SAMPLES = [(y, x) for y in (3, 12, 25, 40, 55, 64) for x in (5, 30, 70, 110)]


def op_source(name):
    return H + OP_CASES[name] + "\ngraph " + GRAPHS.get(name, "a") + "\n"


def op_input():
    chart = np.load(ROOT / "tests" / "data" / "frames" / "chart.npy").astype(np.float64)[:, ::4, ::4]
    from cierra import colour as C
    return C.to_working(np.ascontiguousarray(chart))


def op_result(name):
    from cierra import load
    from cierra.engine import grade_working
    out = grade_working(op_input(), load(op_source(name)), frame=7, px_scale=0.5)
    return {"mean": out.reshape(3, -1).mean(1).tolist(), "samples": [out[:, y, x].tolist() for y, x in SAMPLES]}


def dctl_programs():
    from cierra import load
    from cierra.checker import load_file
    from test_targets import KITCHEN_SINK
    progs = {p.stem: load_file(p) for p in sorted((ROOT / "looks").glob("*.cierra"))}
    progs["kitchen_sink"] = load(KITCHEN_SINK)
    return progs


def dctl_texts():
    from cierra.compile.dctl import compile_dctl
    out = {}
    for name, prog in dctl_programs().items():
        for part in compile_dctl(prog, frame=5 if name == "kitchen_sink" else 0, split=True):
            if part["dctl"]:
                out[f"{name}{'' if part['name'] == 'main' else '_' + part['name']}.dctl"] = part["dctl"]
    return out


def analysis_result():
    from cierra.analysis import stats
    from cierra.checker import load_file
    from cierra.engine import render
    img = np.load(ROOT / "tests" / "data" / "frames" / "crowd.npy").astype(np.float64)
    return stats(render(load_file(ROOT / "looks" / "beauty_modern.cierra"), img))


def write_all():
    (GOLD / "ops.json").write_text(json.dumps({n: op_result(n) for n in OP_CASES}, indent=0) + "\n")
    d = GOLD / "dctl"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.dctl"):
        f.unlink()
    for name, text in dctl_texts().items():
        (d / name).write_text(text, encoding="utf-8", newline="\n")
    (GOLD / "analysis_beauty_crowd.json").write_text(json.dumps(analysis_result(), indent=1, sort_keys=True) + "\n")
