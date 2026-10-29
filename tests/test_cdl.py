"""ASC CDL export (cierra/compile/cdl.py): the folded CDL, read back by OpenColorIO, matches the reference engine; programs
that are not one CDL are refused with every reason."""

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.compile.cdl import CDLError, export_cdl, fold
from cierra.engine import grade_working
from conftest import frame, small

CHAIN = """cierra 0.1
name "cdl chain"
node e = printer_lights(red: 5, green: 5, blue: 5)
node w = printer_lights(red: 2, green: -1, blue: -2)
node p = printer_lights(red: 2, blue: -1)
node c = contrast(amount: 1.1, pivot: 0.4)
node d = cdl(slope: [1.02, 1, 0.98], offset: [0.01, 0, -0.01], power: [0.95, 1, 1.05], sat: 1.1)
node s = saturation(amount: 0.9)
graph e -> w -> p -> c -> d -> s
"""


def _ocio(path, code, cccid=None):
    import PyOpenColorIO as ocio
    t = ocio.FileTransform(src=str(path), interpolation=ocio.INTERP_LINEAR)
    if cccid:
        t.setCCCId(cccid)
    return C._ocio_apply(ocio.Config.CreateRaw().getProcessor(t).getDefaultCPUProcessor(), code)


@pytest.mark.parametrize("ext", ["cc", "ccc", "cdl"])
def test_cdl_files_match_the_engine(tmp_path, ext):
    prog = load(CHAIN)
    export_cdl(prog, tmp_path, "look")
    for name in ("chart", "window_1", "crowd"):
        code = np.clip(C.to_working(small(frame(name), 4)), 0, 1)
        ref = grade_working(code, prog, px_scale=1.0)
        out = _ocio(tmp_path / f"look.{ext}", code, cccid=None if ext == "cc" else "0")
        # ASC readers clamp to 0..1 after slope / offset / power; Cierra's cdl is unclamped (spec): compare where the
        # unclamped SOP stays inside 0..1 (the window's brightest pixels leave it)
        s, o, _, _ = fold(prog)
        sop = code * s[:, None, None] + o[:, None, None]
        inside = ((sop >= 0) & (sop <= 1)).all(0) & ((ref >= 0) & (ref <= 1)).all(0)
        assert inside.mean() > 0.9
        err = np.abs(out - ref)[:, inside]
        assert err.max() < 2e-5, (name, ext, err.max())


def test_fold_composes_in_order():
    pt = 0.025 / (17.52 * np.log10(2))
    slope, offset, power, sat = fold(load("cierra 0.1\nnode a = contrast(amount: 2, pivot: 0.5)\n"
                                          "node b = printer_lights(red: 3, green: 3, blue: 3)\ngraph a -> b\n"))
    assert np.allclose(slope, 2) and np.allclose(offset, -0.5 + 3 * pt) and np.allclose(power, 1) and sat == 1
    twice = fold(load("cierra 0.1\nnode a = printer_lights(red: 1, green: 1, blue: 1)\ngraph a -> a\n"))
    assert np.allclose(twice[1], 2 * pt)  # a node used twice counts twice
    assert np.allclose(fold(load("cierra 0.1\nnode a = exposure()\ngraph a\n"))[1], 0)  # at its default: identity


@pytest.mark.parametrize("src,why", [
    ("node a = contrast(amount: 1.1)\n  mask skin()\ngraph a\n", "has a mask"),
    ("node a = contrast(amount: 1.1)\n  mix 0.5\ngraph a\n", "mix is not 1"),
    ("node a = contrast(amount: keys(linear, 0: 1, 10: 1.2))\ngraph a\n", "keyframed"),
    ("node a = exposure(stops: 1)\ngraph a\n", "linear-light gain"),
    ("node a = white_balance(temperature: 0.1)\ngraph a\n", "linear-light gain"),
    ("node a = exposure()\nnode b = contrast()\ngraph parallel(a, b)\n", "parallel() or layer()"),
    ("node a = curve(points: [[0, 0], [1, 1]])\ngraph a\n", "not a CDL operation"),
    ("node a = cdl(power: [1.1, 1, 1])\nnode b = printer_lights(red: 3)\ngraph a -> b\n", "after a power"),
    ("node a = saturation(amount: 0.9)\nnode b = printer_lights(red: 3)\ngraph a -> b\n", "after a power or saturation"),
    ("node a = saturation(amount: 0.9, protect_skin: 1)\ngraph a\n", "protect_skin"),
])
def test_not_one_cdl_is_refused_with_reasons(tmp_path, src, why):
    with pytest.raises(CDLError, match=why.replace("(", r"\(").replace(")", r"\)")):
        export_cdl(load("cierra 0.1\n" + src), tmp_path, "x")


def test_cli_cdl(tmp_path):
    from cierra.__main__ import main
    src = tmp_path / "look.cierra"
    src.write_text(CHAIN)
    assert main(["cdl", str(src), str(tmp_path / "out")]) in (None, 0)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["look.cc", "look.ccc", "look.cdl"]
    src.write_text("cierra 0.1\nnode a = curve(points: [[0, 0], [1, 1]])\ngraph a\n")
    assert main(["cdl", str(src), str(tmp_path / "out2")]) == 1
