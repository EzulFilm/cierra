"""AMF v2.0 export (cierra/compile/amf.py): written, read back, rendered through OpenColorIO alone, hash checked.
Schema validation runs when CIERRA_AMF_XSD points at github.com/ampas/aces-amf schema/acesMetadataFile.xsd and the
xmlschema package is installed. The full round trip on the chart: tools/amf_roundtrip.py."""

import os

import numpy as np
import pytest

from cierra import colour as C
from cierra import load
from cierra.checker import load_file
from cierra.compile.amf import AMFError, export_amf, render_amf, resolve
from cierra.engine import render
from conftest import LOOKS, small


def test_amf_round_trip_through_opencolorio(tmp_path, chart):
    prog = load_file(LOOKS / "cinematic_modern.cierra")
    files = export_amf(prog, tmp_path, "cm")
    assert set(files) == {"cm_look_ACES2065-1.clf", "cm.amf"}
    r = resolve(tmp_path / "cm.amf")
    assert r["display"] == "sRGB - Display" and r["input"] is None and len(r["looks"]) == 1
    img = small(chart)
    shown = render_amf(tmp_path / "cm.amf", np.ascontiguousarray(C.apply_matrix(C.matrix("rec709", "ap0"), img)))
    de = C.delta_e00_display(np.clip(shown, 0, 1), np.clip(render(prog, img), 0, 1))
    assert de.mean() <= 0.1 and np.quantile(de, 0.999) <= 0.5, (de.mean(), np.quantile(de, 0.999))
    xsd = os.environ.get("CIERRA_AMF_XSD")
    if xsd:
        xmlschema = pytest.importorskip("xmlschema")
        xmlschema.XMLSchema(xsd).validate(str(tmp_path / "cm.amf"))


def test_camera_input_id_and_tampered_look_is_refused(tmp_path):
    prog = load("cierra 0.1\ninput sony_slog3_sgamut3\noutput aces2(display: rec1886)\n"
                "node w = white_balance(temperature: 0.1)\ngraph w\n")
    export_amf(prog, tmp_path, "s")
    text = (tmp_path / "s.amf").read_text()
    assert "CSC.Sony.SLog3_SGamut3_to_ACES" in text and "Rec709-D65_BT1886" in text
    assert resolve(tmp_path / "s.amf")["input"] == C.CAMERA_INPUTS["sony_slog3_sgamut3"]
    clf = tmp_path / "s_look_ACES2065-1.clf"
    clf.write_text(clf.read_text().replace("</ProcessList>", "<!-- x --></ProcessList>"))
    with pytest.raises(AMFError, match="hash"):
        resolve(tmp_path / "s.amf")


def test_refusals(tmp_path):
    with pytest.raises(AMFError, match="LUT"):
        export_amf(load("cierra 0.1\noutput aces2(display: srgb)\nnode g = grain(amount: 0.2, seed: 1)\ngraph g\n"),
                   tmp_path, "g")
    with pytest.raises(AMFError, match="Output"):
        export_amf(load("cierra 0.1\noutput display_encode()\nnode e = exposure(stops: 0.2)\ngraph e\n"), tmp_path, "d")


def test_cli_amf(tmp_path, capsys):
    from cierra.__main__ import main
    assert main(["amf", str(LOOKS / "cinematic_modern.cierra"), str(tmp_path / "out"), "--in", "arri_logc4"]) in (None, 0)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["cinematic_modern.amf",
                                                                   "cinematic_modern_look_ACES2065-1.clf"]
    assert "CSC.Arri.LogC4_to_ACES" in (tmp_path / "out" / "cinematic_modern.amf").read_text()
    assert resolve(tmp_path / "out" / "cinematic_modern.amf")["looks"]
    src = tmp_path / "g.cierra"
    src.write_text("cierra 0.1\noutput aces2(display: srgb)\nnode g = grain(amount: 0.2, seed: 1)\ngraph g\n")
    assert main(["amf", str(src), str(tmp_path / "out2")]) == 1
    assert "NOT AN AMF" in capsys.readouterr().err
