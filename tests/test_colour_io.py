"""Camera log inputs and P3 / HDR outputs through the OpenColorIO studio config: each input lands on the same ACEScct
values OCIO gives directly, outputs match OCIO's display-view transform, and the loop only measures SDR Rec.709."""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.engine import render
from cierra.history import History
from cierra.llm import ScriptedProvider, grade_loop
from cierra.ops import INPUTS, OUTPUTS


def _ocio(src, dst, rgb):
    import PyOpenColorIO as ocio
    cfg = ocio.Config.CreateFromBuiltinConfig(C.ACES2_CONFIG)
    p = cfg.getProcessor(ocio.ColorSpaceTransform(src=src, dst=dst)).getDefaultCPUProcessor()
    a = np.ascontiguousarray(np.array(rgb, dtype=np.float32).reshape(-1, 3))
    p.applyRGB(a)
    return a.astype(np.float64)


@pytest.mark.parametrize("kind", sorted(C.CAMERA_INPUTS))
def test_camera_input_lands_on_the_same_working_values(kind):
    assert kind in INPUTS
    # 18 % grey and a saturated red in ACES2065-1, encoded into the camera's space by OCIO itself
    ap0 = [[0.18, 0.18, 0.18], [0.4, 0.08, 0.05]]
    cam = _ocio("ACES2065-1", C.CAMERA_INPUTS[kind], ap0)
    want = _ocio("ACES2065-1", "ACEScct", ap0)
    got, display_referred = C.input_to_working(cam.T.reshape(3, 2, 1), kind)
    assert not display_referred
    assert np.abs(got.reshape(3, 2).T - want).max() < 2e-4


def test_camera_input_through_a_program():
    prog = load("cierra 0.1\ninput arri_logc4\noutput aces2(display: srgb)\nnode e = exposure(stops: 1)\ngraph e\n")
    grey = _ocio("ACES2065-1", C.CAMERA_INPUTS["arri_logc4"], [[0.18, 0.18, 0.18]]).T.reshape(3, 1, 1)
    out = render(prog, np.repeat(np.repeat(grey, 4, 1), 4, 2))
    assert np.isfinite(out).all() and 0.3 < out.mean() < 0.9


@pytest.mark.parametrize("display", OUTPUTS["aces2"].params["display"].choices)
def test_every_display_matches_ocio_and_has_no_black_holes(display):
    import PyOpenColorIO as ocio
    cfg = ocio.Config.CreateFromBuiltinConfig(C.ACES2_CONFIG)
    disp, view = C.ACES2_DISPLAYS[display]
    proc = cfg.getProcessor(ocio.DisplayViewTransform(src="ACEScct", display=disp, view=view)).getDefaultCPUProcessor()
    rng = np.random.default_rng(1)
    code = rng.uniform(0.1, 0.9, (3, 8, 8))
    ref = np.ascontiguousarray(code.reshape(3, -1).T, dtype=np.float32)
    proc.applyRGB(ref)
    assert np.abs(C.aces2_display(code, display).reshape(3, -1).T - np.clip(ref, 0, 1)).max() < 1e-6
    # the bright-blue singular point of the ACES 2.0 gamut compression: never black on any display
    blue = C.aces2_display(np.array([0.12630391, 0.10264291, 1.11055911]).reshape(3, 1, 1), display)
    assert np.isfinite(blue).all() and blue.max() > 0.01


def test_hdr_output_renders_and_the_loop_refuses_it(tmp_path, chart):
    hdr = "cierra 0.1\noutput aces2(display: pq_p3_1000)\nnode e = exposure(stops: 0.5)\ngraph e\n"
    out = render(load(hdr), chart[:, ::8, ::8])
    assert np.isfinite(out).all() and out.max() <= 1
    assert C.output_eotf(load(hdr).output) is None and not C.measurable(load(hdr).output)
    with pytest.raises(ValueError, match="SDR Rec.709"):
        grade_loop("warmer", chart[:, ::8, ::8], ScriptedProvider([]), History(tmp_path / "a"), start=hdr,
                   log=lambda *_: None)
    # a reply that switches to HDR is a rejected round, not a measured one
    p = ScriptedProvider([f"```cierra\n{hdr}```\nmessage: hdr\n"])
    _, recs = grade_loop("warmer", chart[:, ::8, ::8], p, History(tmp_path / "b"), max_iters=1, log=lambda *_: None)
    assert not recs[0]["accepted"] and "SDR Rec.709" in recs[0]["errors"]
