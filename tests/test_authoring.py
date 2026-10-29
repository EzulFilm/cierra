"""The authoring profile: every valid example passes with no warnings, every invalid example fails with its listed
message, generated docs are current, host lints fire, new semantics (identity graph, full-range hue, photo stage,
spatial pass count) hold."""

import json

import numpy as np
import pytest

from cierra import CheckError, ParseError, load
from cierra import colour as C
from cierra.checker import check, lint
from cierra.engine import grade_working, mask_value, make_ctx
from cierra.examples import INVALID, VALID
from cierra.parser import parse
from conftest import ROOT, small


@pytest.mark.parametrize("title,src", VALID)
def test_valid_examples(title, src):
    _, warnings = check(parse(src))
    assert warnings == []


@pytest.mark.parametrize("title,src,msg", INVALID)
def test_invalid_examples(title, src, msg):
    with pytest.raises((ParseError, CheckError)) as e:
        check(parse(src))
    assert msg in str(e.value), str(e.value)


def test_generated_docs_are_current():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from make_authoring import authoring_text, manifest_text
    assert (ROOT / "docs" / "ops_manifest.json").read_text(encoding="utf-8") == manifest_text()
    assert (ROOT / "docs" / "authoring.md").read_text(encoding="utf-8") == authoring_text()
    m = json.loads(manifest_text())
    assert m["ops"]["glow"]["targets"] == {"reference": True, "dctl": False, "lut": False}
    assert m["ops"]["vignette"]["targets"]["dctl"] and not m["ops"]["vignette"]["targets"]["lut"]
    assert m["ops"]["exposure"]["params"]["stops"]["keyframable"]


def test_omitted_graph_is_identity(chart):
    prog = load("cierra 0.1\noutput acescct()\n")
    code = C.to_working(small(chart))
    assert np.array_equal(grade_working(code, prog), code)


def test_full_range_hue_selects_everything(chart):
    code = C.to_working(small(chart))
    prog = load("cierra 0.1\nnode a = exposure(stops: 1)\n  mask qualifier(hue: 0..360)\ngraph a\n")
    m = mask_value(code, prog.nodes["a"].mask, make_ctx(prog, code.shape[-1]))
    assert np.all(m == 1.0)


def test_photo_to_scene_must_be_first_once():
    with pytest.raises(CheckError, match="first in photo_graph"):
        load("cierra 0.1\nnode a = exposure(stops: 1)\nnode s = photo_to_scene()\nphoto_graph a -> s\ngraph a\n")


def test_spatial_passes_count_uses_and_blurred_mattes():
    nodes = "node g = blur(radius_px: 1)\nnode m = exposure(stops: 1)\n  mask luma(range: 0..1, blur_px: 2)\n"
    ok = "cierra 0.1\n" + nodes + "graph " + " -> ".join(["g"] * 8 + ["m"] * 8) + "\n"
    load(ok)
    with pytest.raises(CheckError, match="spatial passes"):
        load("cierra 0.1\n" + nodes + "graph " + " -> ".join(["g"] * 9 + ["m"] * 8) + "\n")


def test_noop_and_off_warnings():
    _, w = check(parse("cierra 0.1\nnode a = saturation()\nnode b = exposure(stops: 1)\n  off\ngraph a -> b\n"))
    assert any("does nothing" in x for x in w) and any("is off" in x for x in w)


def test_host_lints():
    prog = load("cierra 0.1\nrequires segments\nnode a = exposure(stops: keys(linear, 0: 0, 200: 1))\n"
                "  mask segment(ids: [7])\nnode p = photo_to_scene()\nnode g = glow()\nphoto_graph p\ngraph a -> g\n")
    msgs = lint(prog, {"frames": 100, "planes": [], "segment_ids": [1, 2], "input": "linear_rec709",
                       "targets": ["dctl", "lut"]})
    text = "\n".join(msgs)
    for needle in ("past the clip", "does not provide", "do not exist", "will not run", "not expressible in DCTL",
                   "not expressible as a LUT"):
        assert needle in text, (needle, text)
    assert lint(prog, {}) == []
