"""The stand-alone tools that have no test of their own: the look-pack builder, the set freezer and the bounds
replay, each on a small case."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import ROOT

TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (the DCTL CPU harness)")


@needs_gcc
def test_look_pack_measures_a_dctl_and_refuses_a_lut_with_the_reason(tmp_path):
    import look_packs as LP
    look = tmp_path / "looks" / "tiny.cierra"
    look.parent.mkdir()
    look.write_text("cierra 0.1\nname \"tiny\"\ninput linear_rec709\noutput aces2(display: srgb)\n"
                    "node warm = white_balance(temperature: 0.1)\nnode edge = vignette(amount: 0.2)\ngraph warm -> edge\n")
    LP.FRAMES[:] = ["chart"]  # one frame keeps the test short
    r = LP.one_kind(("tiny", "acescct", str(tmp_path / "out"), str(look.parent)))
    assert r["dctl"]["max"] <= 0.5 and "vignette" in r["lut"]["refused"] and "refused" in r["lmt"]
    assert (tmp_path / "out" / "tiny" / "acescct" / "tiny_acescct.dctl").exists()
    dmg, thumbs, _ = LP.damage(("tiny", str(look.parent)))
    assert dmg["source"] == "tiny.cierra" and "intended_black_lift" in dmg and set(thumbs) == {"window_0", "pens", "chart"}


def test_freeze_set_prints_counts_only_and_refuses_a_bad_set(tmp_path):
    good = {"about": "t", "cases": [
        {"id": "x_chart_a", "frame": "chart", "notes": "A dentist's waiting room, please.", "checks": "cooler, less colour"},
        {"id": "x_pens_b", "frame": "pens", "notes": "Hold the felt where it is and add snap.",
         "checks": "more contrast, keep the blacks clean"}]}
    raw = tmp_path / "raw.txt"
    raw.write_text("some preamble\n" + json.dumps(good))
    out = tmp_path / "set.json"
    r = subprocess.run([sys.executable, str(TOOLS / "freeze_set.py"), str(raw), str(out), "--max-hold", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "sha256" in r.stdout and "dentist" not in r.stdout and out.exists()
    bad = dict(good, cases=good["cases"] + [{"id": "x_chart_c", "frame": "chart", "notes": "Protect the top.",
                                             "checks": "keep the highlights soft"}])
    raw.write_text(json.dumps(bad))
    r = subprocess.run([sys.executable, str(TOOLS / "freeze_set.py"), str(raw), str(tmp_path / "bad.json"), "--max-hold", "1"],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "hold checks only" in r.stdout and not (tmp_path / "bad.json").exists()


def test_replay_bounds_starts_from_the_raw_reply(tmp_path, monkeypatch):
    import replay_bounds as RB
    reply = tmp_path / "case1" / "002_reply.txt"
    reply.parent.mkdir()
    reply.write_text("```cierra\ncierra 0.1\noutput aces2(display: srgb)\n"
                     "node top = curve(points: [[0.0, 0.0], [0.55, 0.55], [1.1, 0.86]])\ngraph top\n```\nmessage: x\n")
    case = {"id": "case1", "frame": "chart", "notes": "n", "checks": "", "translated": ""}
    off = RB.one((case, str(tmp_path / "{id}" / "002_reply.txt"), "CIERRA_ROLLOFF", "0"))
    on = RB.one((case, str(tmp_path / "{id}" / "002_reply.txt"), "CIERRA_ROLLOFF", "1"))
    monkeypatch.delenv("CIERRA_ROLLOFF", raising=False)
    assert off["reply"] == on["reply"] == "ok" and any("top.points" in c for c in off["capped"]) and not on["capped"]
    assert RB.one((dict(case, id="none"), str(tmp_path / "{id}" / "x.txt"), "CIERRA_ROLLOFF", "0"))["reply"] == "no reply"
