"""Regression tests: catch any change in behaviour by name. Goldens are written by tools/make_golden.py; when one
fails after an intended change, regenerate and say why in the commit."""

import json
import random
import subprocess
import sys

import numpy as np
import pytest

from cierra import CierraError, format_program, load
from cierra import ast as A
from cierra.checker import check
from cierra.compile.lut import lattice, write_clf, write_cube, apply_file_lut
from cierra.examples import VALID
from cierra.history import History, diff
from cierra.ops import MASKS, OPS
from cierra.parser import parse
from regression_cases import GOLD, OP_CASES, ROOT, analysis_result, dctl_texts, op_result, op_source

OPS_GOLD = json.loads((GOLD / "ops.json").read_text())


# ---------------------------------------------------------------- op maths, by op

@pytest.mark.parametrize("name", sorted(OP_CASES))
def test_op_golden(name):
    assert name in OPS_GOLD, "new case: run tools/make_golden.py"
    got, want = op_result(name), OPS_GOLD[name]
    assert np.allclose(got["mean"], want["mean"], atol=1e-6), (got["mean"], want["mean"])
    assert np.allclose(got["samples"], want["samples"], atol=1e-6)


def test_op_cases_cover_every_op_and_mask():
    used = {line.split("=")[1].split("(")[0].strip() for src in OP_CASES.values() for line in src.splitlines()
            if line.startswith("node")}
    masks = {m for src in OP_CASES.values() for m in MASKS if f"{m}(" in src}
    assert used | {"photo_to_scene"} == set(OPS), set(OPS) - used
    assert masks >= set(MASKS) - {"depth", "segment"}


def test_op_cases_are_not_identities():
    from regression_cases import op_input
    base = op_input().reshape(3, -1).mean(1)
    for name, r in OPS_GOLD.items():
        assert np.abs(np.array(r["mean"]) - base).max() > 1e-5 or name in ("grain", "sharpen", "blur"), name


def test_depth_and_segment_masks():
    prog = load(H_AUX + "node a = exposure(stops: 1)\n  mask depth(range: 2..8, soft: 0)\n"
                "node b = exposure(stops: -1)\n  mask segment(ids: [2])\ngraph a -> b\n")
    from cierra.engine import grade_working
    code = np.full((3, 4, 4), 0.4)
    depth = np.array([[1, 3, 5, 20]] * 4, dtype=float)
    seg = np.array([[0, 2, 2, 1]] * 4)
    out = grade_working(code, prog, aux={"depth": depth, "segments": seg})
    stop = 1 / 17.52
    assert np.allclose(out[0, 0], [0.4, 0.4 + stop - stop, 0.4 + stop - stop, 0.4])
    with pytest.raises(ValueError, match="needs the 'depth' plane"):
        grade_working(code, prog)


H_AUX = "cierra 0.1\noutput acescct()\nrequires depth, segments\n"


# ---------------------------------------------------------------- compiled text

def test_dctl_text_snapshots():
    got = dctl_texts()
    want = {p.name: p.read_text(encoding="utf-8") for p in (GOLD / "dctl").glob("*.dctl")}
    assert set(got) == set(want)
    for name in got:
        assert got[name] == want[name], f"{name} changed: run tools/make_golden.py if intended"


def test_cube_and_clf_identity_round_trip(tmp_path):
    lut = lattice(9)
    write_cube(tmp_path / "id.cube", lut, "identity")
    write_clf(tmp_path / "id.clf", lut, "identity", "x", "ACEScct", "ACEScct")
    probe = np.random.default_rng(1).random((3, 8, 8))
    for ext in ("cube", "clf"):
        assert np.abs(apply_file_lut(tmp_path / f"id.{ext}", probe) - probe).max() < 1e-6
    lines = (tmp_path / "id.cube").read_text().splitlines()
    assert lines[1] == "LUT_3D_SIZE 9" and len(lines) == 4 + 9 ** 3


# ---------------------------------------------------------------- checker

def _boundary_cases():
    out = []
    for name, spec in OPS.items():
        if name == "photo_to_scene":
            continue
        for key, p in spec.params.items():
            if p.kind in ("num", "int") and not p.required:
                out.append((name, key, p))
    return out


@pytest.mark.parametrize("op,key,p", _boundary_cases(), ids=lambda v: v if isinstance(v, str) else "")
def test_range_boundaries(op, key, p):
    req = ", seed: 1" if op == "grain" and key != "seed" else ""
    fmt = (lambda v: str(int(v))) if p.kind == "int" else (lambda v: repr(float(v)))
    for v in (p.lo, p.hi):
        load(f"cierra 0.1\nnode a = {op}({key}: {fmt(v)}{req})\ngraph a\n")
    eps = 1 if p.kind == "int" else max(1e-6, abs(p.hi - p.lo) * 1e-6)
    for v in (p.lo - eps, p.hi + eps):
        with pytest.raises(CierraError, match="outside"):
            load(f"cierra 0.1\nnode a = {op}({key}: {fmt(v)}{req})\ngraph a\n")


def test_parser_never_crashes_on_mutations():
    """Deterministic fuzz: 3000 mutated programs either load or raise a CierraError; nothing else escapes."""
    rng = random.Random(20261026)
    corpus = [src for _, src in VALID] + [p.read_text() for p in (ROOT / "looks").glob("*.cierra")]
    alphabet = list("()[],:=->.#\"\n abcxyz0123456789-+*/;{}\\") + ["keys(", "node", "graph", "mask", "not", ".."]
    for _ in range(3000):
        s = list(rng.choice(corpus))
        for _ in range(rng.randint(1, 4)):
            i = rng.randrange(len(s))
            r = rng.random()
            if r < 0.4:
                del s[i]
            elif r < 0.8:
                s.insert(i, rng.choice(alphabet))
            else:
                s[i] = rng.choice(alphabet)
        try:
            load("".join(s))
        except CierraError:
            pass


def test_canonical_form_is_stable():
    for _, src in VALID:
        text = format_program(load(src))
        assert format_program(load(text)) == text


# ---------------------------------------------------------------- history and diffs

def test_diff_messages(tmp_path):
    a = load("cierra 0.1\nnode w = white_balance(temperature: 0.1)\nnode s = saturation(amount: 1.1)\n"
             "  mask skin()\ngraph w -> s\n")
    b = load("cierra 0.1\noutput aces2(display: rec1886)\nnode w = white_balance(temperature: 0.2, tint: 0.05)\n"
             "node s = saturation(amount: 1.1)\n  mask skin(width: 40)\n  mix 0.5\nnode c = contrast(amount: 1.1)\n"
             "graph w -> s -> c\n")
    d = diff(a, b)
    for line in ("output: aces2() -> aces2(display: rec1886)",
                 "~ node w: white_balance temperature 0.1 -> 0.2", "~ node w: white_balance tint 0 -> 0.05",
                 "~ node s: mask skin() -> skin(width: 40)", "~ node s: mix 1 -> 0.5",
                 "+ node c added: contrast(amount: 1.1)", "graph: w -> s  ->  w -> s -> c"):
        assert line in d, (line, d)
    assert diff(b, a)[-1].startswith("graph:") and "- node c removed (contrast(amount: 1.1))" in diff(b, a)
    assert diff(a, a) == ["no change"]
    h = History(tmp_path / "h")
    e1, e2 = h.commit(a, "first"), h.commit(b, "second")
    assert e2["parent"] == e1["hash"] and h.load(e2).nodes["s"].mix == 0.5
    assert e1["hash"] != e2["hash"] and len(e1["hash"]) == 12


# ---------------------------------------------------------------- analysis numbers the LLM sees

def test_analysis_snapshot():
    want = json.loads((GOLD / "analysis_beauty_crowd.json").read_text())

    def close(a, b, path=""):
        if isinstance(a, dict):
            assert set(a) == set(b), path
            for k in a:
                close(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                close(x, y, f"{path}[{i}]")
        elif isinstance(a, float):
            assert abs(a - b) <= 2e-3, (path, a, b)
        else:
            assert a == b, (path, a, b)
    close(analysis_result(), want)


# ---------------------------------------------------------------- command line

def _cli(*args):
    r = subprocess.run([sys.executable, "-m", "cierra", *args], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_cli_check_render_dctl_lut(tmp_path):
    code, out, _ = _cli("check", "looks/cinematic_modern.cierra")
    assert code == 0 and out.startswith("cierra 0.1")
    code, out, _ = _cli("render", "looks/cinematic_modern.cierra", "tests/data/frames/chart.npy", str(tmp_path / "c.png"))
    assert code == 0 and (tmp_path / "c.png").exists()
    code, out, _ = _cli("dctl", "looks/teal_night.cierra", str(tmp_path / "d"), "--split")
    assert code == 0 and "NOT IN DCTL" in out and len(list((tmp_path / "d").glob("*.dctl"))) == 2
    code, out, _ = _cli("lut", "looks/cinematic_modern.cierra", str(tmp_path / "l"), "--size", "17")
    assert code == 0 and len(list((tmp_path / "l").iterdir())) == 3
    bad = tmp_path / "bad.cierra"
    bad.write_text("cierra 0.1\nnode a = exposure(stops: 99)\ngraph a\n")
    code, _, err = _cli("check", str(bad))
    assert code != 0 and "outside" in err


def test_check_api_does_not_mutate_reused_sources():
    src = VALID[2][1]
    p1, _ = check(parse(src))
    p2, _ = check(parse(src))
    assert format_program(p1) == format_program(p2)
    assert isinstance(p1.nodes["cool_shadows"].mask, A.MaskBin)


def test_op_source_helper():
    assert op_source("parallel").rstrip().endswith("graph parallel(a, b)")


# ---------------------------------------------------------------- spec review findings

def test_noop_warning_only_for_identity_ops():
    for src, warns in (("node a = vignette()", False), ("node a = glow()", False), ("node a = blur()", False),
                       ("node a = exposure()", True), ("node a = wheels()", True), ("node a = saturation()", True)):
        _, w = check(parse(f"cierra 0.1\n{src}\ngraph a\n"))
        assert any("does nothing" in x for x in w) == warns, (src, w)


@pytest.mark.parametrize("src,msg", [
    ("node a = exposure(stops: 1) mix 0.5\n  mix 0.7", "`mix` twice"),
    ('node a = exposure(stops: 1)\n  label "x"\n  label "y"', "`label` twice"),
    ("node a = exposure(stops: 1) off off", "`off` twice"),
    ("node a = exposure(stops: 1) mask skin()\n  mask luma(range: 0..1)", "two masks"),
])
def test_duplicate_modifiers_rejected(src, msg):
    with pytest.raises(CierraError, match=msg):
        load(f"cierra 0.1\n{src}\ngraph a\n")


def test_duplicate_layer_arguments_rejected():
    base = "cierra 0.1\nnode a = exposure(stops: 1)\nnode b = contrast(amount: 1.1)\n"
    load(base + "graph layer(a, b, mode: add, opacity: 0.5)\n")
    for args in ("mode: add, mode: multiply", "opacity: 0.2, opacity: 0.4"):
        with pytest.raises(CierraError, match="given twice"):
            load(base + f"graph layer(a, b, {args})\n")


def test_layer_and_mix_defaults():
    prog = load("cierra 0.1\nnode a = exposure(stops: 1)\nnode b = contrast(amount: 1.1)\ngraph layer(a, b)\n")
    assert prog.graph.mode == "normal" and prog.graph.opacity == 1.0 and prog.nodes["a"].mix == 1.0
    from cierra.manifest import manifest
    m = manifest()
    assert m["node_modifiers"]["mix"]["default"] == 1.0 and m["node_modifiers"]["mix"]["keyframable"]
    assert m["graph"]["layer"]["mode"]["default"] == "normal" and m["graph"]["layer"]["opacity"]["default"] == 1.0
    assert m["ops"]["vignette"]["identity_with_defaults"] is False and m["ops"]["exposure"]["identity_with_defaults"]


def test_photo_to_scene_placement():
    head = "cierra 0.1\nnode s = photo_to_scene()\nnode a = exposure(stops: 1)\n"
    load(head + "photo_graph s -> a\ngraph a\n")
    load(head + "photo_graph s\ngraph a\n")
    for g in ("parallel(s, a)", "layer(s, a)", "a -> s", "s -> s"):
        with pytest.raises(CierraError, match="must come first"):
            load(head + f"photo_graph {g}\ngraph a\n")
    with pytest.raises(CierraError, match="must be below max_stops"):
        load("cierra 0.1\nnode s = photo_to_scene(knee_stops: 6, max_stops: 6)\nphoto_graph s\n")


def test_each_blurred_mask_call_is_a_spatial_pass():
    node = "node m = exposure(stops: 1)\n  mask skin(blur_px: 5) or window(blur_px: 10)\n"
    load("cierra 0.1\n" + node + "graph " + " -> ".join(["m"] * 8) + "\n")  # 16 passes: at the limit
    with pytest.raises(CierraError, match="17 spatial passes"):
        load("cierra 0.1\n" + node + "node g = blur(radius_px: 1)\ngraph " + " -> ".join(["m"] * 8) + " -> g\n")


def test_recovered_highlights_reachable_with_linear_photo():
    from cierra.engine import render
    on = load("cierra 0.1\ninput linear_photo\noutput acescct()\nnode s = photo_to_scene()\nphoto_graph s\n")
    off = load("cierra 0.1\ninput linear_photo\noutput acescct()\n"
               "node s = photo_to_scene(recovered_highlights: false)\nphoto_graph s\n")
    img = np.full((3, 2, 2), 0.5)
    img[:, 0, 0] = 3.0  # recovered white, 1.6 stops over display white
    a, b = render(on, img), render(off, img)
    # reachable, but the highlight roll-off after it (4.5 -> 6 stops over grey) swallows almost all of it: the
    # inverse output transform already sends display white past the ceiling (spec §5 photo_to_scene)
    assert 0 < a[0, 0, 0] - b[0, 0, 0] < 0.001 and np.allclose(a[:, 1, 1], b[:, 1, 1])
    # the same pixel as an sRGB photo is clipped to white first, so the option cannot change anything
    s_on = render(on, np.clip(img, 0, 1), input="srgb_photo")
    s_off = render(off, np.clip(img, 0, 1), input="srgb_photo")
    assert np.array_equal(s_on, s_off)


def test_delta_e_uses_the_output_eotf():
    from cierra import colour as C
    assert C.output_eotf(load("cierra 0.1\noutput aces2(display: rec1886)\n").output) == "gamma24"
    assert C.output_eotf(load("cierra 0.1\noutput display_encode()\n").output) == "srgb"
    assert C.output_eotf(load("cierra 0.1\noutput acescct()\n").output) is None
    a, b = np.full((3, 1, 1), 0.10), np.full((3, 1, 1), 0.12)
    assert C.delta_e00_display(a, b, "gamma24")[0, 0] != C.delta_e00_display(a, b, "srgb")[0, 0]


def test_photo_graph_lint_knows_linear_photo():
    from cierra.checker import lint
    prog = load("cierra 0.1\nnode s = photo_to_scene()\nphoto_graph s\n")
    assert lint(prog, {"input": "linear_photo"}) == [] and lint(prog, {"input": "srgb_photo"}) == []
    assert "will not run" in lint(prog, {"input": "aces2065_1"})[0]


# ---------------------------------------------------------------- grain hash parity (spec §5 / §12)

@pytest.mark.skipif(__import__("shutil").which("gcc") is None, reason="gcc needed for the DCTL CPU harness")
def test_grain_hash_is_bit_exact_between_reference_and_dctl():
    """The claim the spec keeps: same seed + frame selects the same random cells on every target. The DCTL's
    integer hash (as emitted by the compiler) and the reference engine's agree bit for bit, also for negative cell
    indices; the float cell noise built on it agrees within float32 rounding."""
    from cierra.compile.dctl import GRAIN_HELPERS, PRELUDE
    from cierra.compile.dctl_cpu import run_dctl
    from cierra.engine import _cell_noise, hash_u32
    frame, seed = 7, 123456789
    src = (PRELUDE + GRAIN_HELPERS + r"""
__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B) {
    int ix = p_X - 40, iy = p_Y - 3;
    unsigned int h = 0x9E3779B9u;
    h = cr_hash(h, (unsigned int)ix); h = cr_hash(h, (unsigned int)iy); h = cr_hash(h, (unsigned int)%d);
    h = cr_hash(h, (unsigned int)%d); h = cr_hash(h, (unsigned int)2);
    return make_float3((float)(h >> 8), (float)(h & 255u), cr_cell_noise(ix, iy, %d, %d, 2));
}
""" % (frame, seed, frame, seed))
    w, hgt = 96, 8
    out = run_dctl(src, np.zeros((3, hgt, w)))
    ix = (np.arange(w)[None, :] - 40) * np.ones((hgt, 1), dtype=np.int64)
    iy = (np.arange(hgt)[:, None] - 3) * np.ones((1, w), dtype=np.int64)
    ref = hash_u32(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)
    assert np.array_equal(out[0].astype(np.uint64), (ref >> np.uint32(8)).astype(np.uint64))
    assert np.array_equal(out[1].astype(np.uint64), (ref & np.uint32(255)).astype(np.uint64))
    noise = _cell_noise(ix & 0xFFFFFFFF, iy & 0xFFFFFFFF, frame, seed, 2)
    assert np.abs(out[2] - noise).max() < 1e-5
