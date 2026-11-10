"""Neighbour sampling in DCTL (G5): one Gaussian node per DCTL (blur, sharpen, glow, halation, diffusion within the
kernel-radius budget) reads its input through the textures, re-running the nodes before it on every sampled pixel.
Checked on the gcc harness against the engine."""

import shutil

import numpy as np
import pytest

from cierra import colour as C
from cierra.checker import load
from cierra.compile.dctl import DCTLError, compile_dctl
from cierra.engine import grade_working
from cierra.targets import DCTL_TAP_RADIUS, dctl_node_reasons, dctl_reasons
from conftest import frame

gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc needed for the DCTL CPU harness")
OPS = {
    "blur": "blur(radius_px: 1.5)",
    "sharpen": "sharpen(amount: 1.5, radius_px: 1.2, coring: 0.002)",
    "glow": "glow(threshold: 0.45, radius_px: 2, strength: 0.8, warm: 0.2)",
    "halation": "halation(threshold: 0.4, radius_px: 1.8, strength: 1.0)",
    "diffusion": "diffusion(radius_px: 2.5, strength: 0.6)",
}


def run(part, img):
    from cierra.compile.dctl_cpu import run_dctl
    return run_dctl(part["dctl"], img, luts=part.get("luts"))


def crop():
    return np.ascontiguousarray(frame("window_0")[:, 200:236, 300:348])  # bright edges and a window


@gcc
@pytest.mark.parametrize("op", sorted(OPS))
def test_tap_node_matches_engine_with_nodes_before_and_after(op):
    """A position op and a pointwise op before the tap node (re-run per sampled pixel at the sample's own position),
    a pointwise op after it, a mask on it; reference width 64 on a 48 px crop, so p_scale is 0.75."""
    src = ("cierra 0.1\nreference_width 64\nnode v = vignette(amount: 0.4)\nnode e = exposure(stops: 0.4)\n"
           f"node s = {OPS[op]}\n  mix 0.8\n  mask luma(range: 0.2..1.2, soft: 0.05)\nnode c = saturation(amount: 1.2)\n"
           "graph v -> e -> s -> c\noutput acescct()\n")
    prog = load(src)
    part, = compile_dctl(prog)
    assert "__TEXTURE__ p_TexR" in part["dctl"] and "cr_pre(" in part["dctl"]
    img = crop()
    out, ref = run(part, img), grade_working(img, prog)
    assert np.abs(out - ref).max() < 2e-3, np.abs(out - ref).max()


@gcc
def test_tap_node_first_and_with_wrapper():
    """The tap node first in the chain, on Rec.709 video through the wrapper: the decode runs per sampled pixel."""
    prog = load("cierra 0.1\nnode s = sharpen(amount: 1.0, radius_px: 1.5)\ngraph s\noutput display_encode(eotf: gamma24)\n")
    part, = compile_dctl(prog, input="rec1886_photo", output="program")
    img = C.display_encode(crop(), "gamma24")
    ref = C.display_encode(grade_working(C.input_to_working(img, "rec1886_photo")[0], prog), "gamma24")
    assert np.abs(run(part, img) - ref).max() < 2e-3


def test_radius_budget_and_structure():
    big = load("cierra 0.1\nnode s = blur(radius_px: 4.5)\ngraph s\n")  # int(13.5) = 13 > 12
    assert any("wider than the DCTL" in r for r in dctl_node_reasons(big.nodes["s"]))
    ok = load(f"cierra 0.1\nnode s = blur(radius_px: {DCTL_TAP_RADIUS / 3})\ngraph s\n")
    assert dctl_node_reasons(ok.nodes["s"]) == []
    two = load("cierra 0.1\nnode a = blur(radius_px: 1)\nnode b = exposure()\nnode c = sharpen()\ngraph a -> b -> c\n")
    assert any("one per DCTL" in r for r in dctl_reasons(two))
    with pytest.raises(DCTLError, match="one per DCTL"):
        compile_dctl(two)
    parts = compile_dctl(two, split=True)
    assert [p["nodes"] for p in parts] == [["a", "b"], ["c"]] and all(p["dctl"] for p in parts)
    branch = load("cierra 0.1\nnode a = blur(radius_px: 1)\nnode b = exposure()\ngraph parallel(a, b)\n")
    with pytest.raises(DCTLError, match="parallel / layer"):
        compile_dctl(branch)
    with pytest.raises(DCTLError, match="parallel / layer"):
        compile_dctl(branch, split=True)


@gcc
def test_same_tap_node_twice_is_two_parts():
    """Uses are counted, not nodes (review): b -> e -> b is two neighbour-sampling passes, so split gives two parts,
    and together they equal the engine."""
    prog = load("cierra 0.1\nreference_width 64\nnode b = blur(radius_px: 1.5)\nnode e = exposure(stops: 1)\n"
                "graph b -> e -> b\noutput acescct()\n")
    with pytest.raises(DCTLError, match="more than one"):
        compile_dctl(prog)
    p1, p2 = compile_dctl(prog, split=True)
    img = crop()
    assert np.abs(run(p2, run(p1, img)) - grade_working(img, prog)).max() < 2e-3


def test_carried_key_elsewhere_does_not_block_a_tap_part():
    """The carried-key check looks at the part being compiled only (review)."""
    prog = load("cierra 0.1\nnode a = exposure(stops: 0.5)\n  mask luma(range: 0.3..1)\nnode k = saturation(amount: 0.8)\n"
                "  mask key(a)\nnode g = glow(radius_px: 40)\nnode b = blur(radius_px: 1)\ngraph a -> k -> g -> b\n")
    parts = compile_dctl(prog, split=True)
    assert [p["nodes"] for p in parts] == [["a", "k"], ["g"], ["b"]] and parts[2]["dctl"]


# ---------------------------------------------------------------- two-pass luma Gaussian (G5 B)
LUMA_OPS = {
    "sharpen": "sharpen(amount: 1.5, radius_px: 6, coring: 0.002)",
    "local_contrast": "local_contrast(amount: 0.6, radius_px: 40, protect_blacks: true)",
    "highlights_shadows": "highlights_shadows(highlights: -0.8, shadows: 0.5, radius_px: 12)",
}


def run_parts(parts, img):
    for part in parts:
        assert part["dctl"] is not None, part["note"]
        img = run(part, img)
    return img


@gcc
@pytest.mark.parametrize("op", sorted(LUMA_OPS))
def test_two_pass_luma_matches_engine_with_prefix_suffix_mask_and_mix(op):
    """The horizontal pass returns prefix RGB plus alpha; the vertical pass applies the op and suffix. A wide radius
    exercises the path small sharpen never uses, with the same separable kernel and replicated edges as the engine."""
    src = ("cierra 0.1\nreference_width 64\nnode v = vignette(amount: 0.4)\nnode e = exposure(stops: 0.4)\n"
           f"node s = {LUMA_OPS[op]}\n  mix 0.8\n  mask luma(range: 0.2..1.2, soft: 0.05)\n"
           "node c = saturation(amount: 1.2)\ngraph v -> e -> s -> c\noutput acescct()\n")
    prog = load(src)
    horizontal, vertical = compile_dctl(prog)
    assert (horizontal["pass"], vertical["pass"]) == ("horizontal", "vertical")
    assert "float4 transform" in horizontal["dctl"] and "p_TexA" in vertical["dctl"]
    img = crop()
    mid = run(horizontal, img)
    assert mid.shape[0] == 4
    prefix = load("cierra 0.1\nreference_width 64\nnode v = vignette(amount: 0.4)\n"
                  "node e = exposure(stops: 0.4)\ngraph v -> e\noutput acescct()\n")
    assert np.abs(mid[:3] - grade_working(img, prefix)).max() < 2e-3
    out = run(vertical, mid)
    assert np.abs(out - grade_working(img, prog)).max() < 3e-3, np.abs(out - grade_working(img, prog)).max()


@gcc
def test_two_pass_luma_with_display_wrapper():
    prog = load("cierra 0.1\nreference_width 64\nnode s = local_contrast(amount: 0.4, radius_px: 30)\n"
                "graph s\noutput display_encode(eotf: gamma24)\n")
    parts = compile_dctl(prog, input="rec1886_photo", output="program")
    img = C.display_encode(crop(), "gamma24")
    ref = C.display_encode(grade_working(C.input_to_working(img, "rec1886_photo")[0], prog), "gamma24")
    assert np.abs(run_parts(parts, img) - ref).max() < 3e-3


@gcc
def test_split_expands_each_luma_node_and_keeps_small_taps_unchanged():
    prog = load("cierra 0.1\nreference_width 64\nnode a = local_contrast(radius_px: 40)\n"
                "node b = sharpen(amount: 0.8, radius_px: 1.5)\ngraph a -> b\noutput acescct()\n")
    with pytest.raises(DCTLError, match="one per DCTL"):
        compile_dctl(prog)
    parts = compile_dctl(prog, split=True)
    assert [p.get("pass") for p in parts] == ["horizontal", "vertical", None]
    assert "float4 transform" in parts[0]["dctl"] and "cr_pre" in parts[2]["dctl"]
    img = crop()
    assert np.abs(run_parts(parts, img) - grade_working(img, prog)).max() < 3e-3


def test_small_sharpen_stays_on_direct_taps_and_other_spatial_ops_are_still_refused():
    small = load("cierra 0.1\nnode s = sharpen(radius_px: 1.5)\ngraph s\n")
    part, = compile_dctl(small)
    assert "float3 transform" in part["dctl"] and "p_TexA" not in part["dctl"]
    for op in ("guided_detail()", "denoise()", "glow(radius_px: 40)", "blur(radius_px: 40)"):
        prog = load(f"cierra 0.1\nnode s = {op}\ngraph s\n")
        with pytest.raises(DCTLError, match="spatial"):
            compile_dctl(prog)
