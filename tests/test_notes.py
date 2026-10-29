"""Note checks and the short reference for small models."""

import json

import numpy as np

from cierra import load
from cierra.engine import render
from cierra.history import History
from cierra.llm import (ScriptedProvider, grade_loop, language_reference, language_reference_short, preview,
                        SHORT_OPS)
from cierra.notes import measure_notes, probes_for, regions, report, unmeasured
from cierra.ops import OPS
from conftest import frame

NOTES = "Warmer faces, less green in the trees, a little more contrast, keep the blacks clean and the highlights soft."
# what Qwen2.5-7B wrote in a first local run
FIRST_RUN = """cierra 0.1
node warm_faces = white_balance(temperature: 0.2)
node less_green = hue_vs_sat(points: [[150, 0.9, 60], [206, 0.8, 60]])
node more_contrast = contrast(amount: 1.2, pivot: 0.3)
node clean_blacks = exposure(stops: -0.5)
graph warm_faces -> less_green -> more_contrast -> clean_blacks
"""
FOLLOWS_NOTES = """cierra 0.1
node warm_faces = white_balance(temperature: 0.15)
  mask skin()
node calm_greens = hue_vs_sat(points: [[206, 0.8, 40]])
node shape = curve(points: [[0.0729, 0.0729], [0.25, 0.238], [0.4136, 0.4136], [0.55, 0.552], [1, 0.97]])
graph warm_faces -> calm_greens -> shape
"""


def _verdicts(src):
    img = preview(frame("crowd"), 480)
    base = render(load("cierra 0.1\n"), img)
    probes, regs = probes_for(NOTES), regions(base)
    b = measure_notes(probes, base, regs)
    n = measure_notes(probes, render(load(src), img), regs)
    rows, lines = report(probes, b, b, n)
    return {r["note"].split(":")[0]: r["vs_ungraded"] for r in rows}, lines


def test_notes_become_probes():
    keys = [row[1] for row in probes_for(NOTES)]
    assert keys == ["skin_warmth", "rest_warmth", "foliage_chroma", "contrast_up", "blacks", "highlights",
                    "highlights_white"]
    assert unmeasured(NOTES) == []
    assert unmeasured("warmer, make it feel like autumn") == ["make it feel like autumn"]


def test_first_live_run_is_flagged_where_it_went_wrong():
    v, lines = _verdicts(FIRST_RUN)
    assert v["warmer faces"] == "right way"            # the faces did get warmer ...
    assert v["warmer faces only"] == "WRONG WAY"       # ... but so did everything else
    assert v["keep the blacks clean"] == "WRONG WAY"   # crushed 15.7 % -> 18.9 %
    assert v["highlights soft"] == "WRONG WAY"         # p99.5 brighter
    assert v["more contrast"] == "right way"
    assert any("WRONG WAY" in line and "crushed" in line for line in lines)


def test_a_grade_that_follows_the_notes_passes_them():
    v, _ = _verdicts(FOLLOWS_NOTES)
    assert v == {"warmer faces": "right way", "warmer faces only": "ok", "less green": "right way",
                 "more contrast": "right way", "keep the blacks clean": "ok", "highlights soft": "ok",
                 "highlights soft (white)": "ok"}, v


def test_short_reference_is_short_and_from_the_op_table():
    short, full = language_reference_short(), language_reference()
    assert len(short) < 0.3 * len(full)
    assert all(op in OPS for op in SHORT_OPS)
    assert "exposure(stops: -6..6 =0)" in short and "wheels(lift: [rgb] -0.3..0.3 =[0, 0, 0]" in short
    assert "[t, -n, -t]" not in full and "[t, -n, -t]" not in short


class _Small(ScriptedProvider):
    prompt_profile = "short"


def test_loop_feeds_note_checks_back(tmp_path):
    img = frame("crowd")
    provider = _Small([f"```cierra\n{FIRST_RUN}```\nmessage: v1\n", f"```cierra\n{FOLLOWS_NOTES}```\nmessage: v2\n"])
    _, recs = grade_loop(NOTES, img, provider, History(tmp_path / "h"), max_iters=2, width=480, log=lambda *_: None)
    second_prompt = provider.calls[1]
    assert "WRONG WAY" in second_prompt and "keep the blacks clean" in second_prompt
    assert "Hints:" not in second_prompt  # the reference is the system prompt, the user turn carries numbers
    payload = json.loads(second_prompt.split("numbers only):\n", 1)[1].split("\nWrite the next", 1)[0])
    assert set(payload["stats_now"]) >= {"crushed_black_pct", "white_p99.5"}  # compact numbers for small models
    assert [r["vs_previous"] for r in recs[1]["note_checks"]][-2:] == ["ok", "ok"]
    assert len(second_prompt) < 4000


# ---------------------------------------------------------------- explicit revision turn

def test_revision_names_the_failing_notes_and_their_nodes(tmp_path):
    img = frame("crowd")
    same = f"```cierra\n{FIRST_RUN}```\nmessage: again\n"
    provider = _Small([f"```cierra\n{FIRST_RUN}```\nmessage: v1\n", same, f"```cierra\n{FOLLOWS_NOTES}```\nmessage: fixed\n"])
    best, recs = grade_loop(NOTES, img, provider, History(tmp_path / "h"), max_iters=4, width=480,
                            log=lambda *_: None)
    second = provider.calls[1]
    assert second.startswith("REVISE FIRST.")  # failing notes come first
    assert "Caused by node clean_blacks" in second and "The next version must change node(s)" in second
    # 7 checks: the highlights note has two (p99.5 and pixels at white); FIRST_RUN darkens, so at-white holds
    assert recs[0]["causes"]["blacks"] == "clean_blacks" and recs[0]["notes_met"] == "3/7"
    # round 2 resent the same program: re-asked once at a higher temperature, which then fixed it
    assert provider.temperatures == [None, None, 0.7] and "It must be different" in provider.calls[2]
    assert recs[1]["reasked"] and recs[1]["accepted"] and recs[1]["notes_met"] == "7/7"
    assert len(recs) == 2 and "calm_greens" in best.nodes  # every note met: stops; best = most notes met


def test_unchanged_after_reask_stops(tmp_path):
    img = frame("crowd")
    same = f"```cierra\n{FIRST_RUN}```\nmessage: same\n"
    provider = _Small([same, same, same])
    _, recs = grade_loop(NOTES, img, provider, History(tmp_path / "h"), max_iters=4, width=480, log=lambda *_: None)
    assert recs[-1]["no_change"] and recs[-1]["reasked"] and len(provider.calls) == 3


def test_revision_off_is_the_old_loop(tmp_path):
    img = frame("crowd")
    same = f"```cierra\n{FIRST_RUN}```\nmessage: same\n"
    provider = _Small([same, same])
    _, recs = grade_loop(NOTES, img, provider, History(tmp_path / "h"), max_iters=4, width=480, log=lambda *_: None,
                         revise=False)
    assert "REVISE" not in provider.calls[1] and recs[-1]["no_change"] and not recs[-1]["reasked"]
    assert len(provider.calls) == 2


def test_pixels_at_white_catch_what_p99_5_hides():
    # a light source in frame (the window pushed 4 stops): p99.5 is already display white, so a brighter grade cannot
    # move it, but more of the frame clips
    img = preview(frame("window_1") * 16, 480)
    base = render(load("cierra 0.1\n"), img)
    probes, regs = probes_for("keep the highlights soft"), regions(base)
    b = measure_notes(probes, base, regs)
    n = measure_notes(probes, render(load("cierra 0.1\nnode e = exposure(stops: 0.3)\ngraph e\n"), img), regs)
    rows, _ = report(probes, b, b, n)
    v = {r["note"]: r["vs_ungraded"] for r in rows}
    assert v["highlights soft: brightest highlights"] == "ok"
    assert v["highlights soft (white): pixels at white"] == "WRONG WAY"


def test_a_start_program_is_not_the_ungraded_baseline(tmp_path):
    # starting one stop up, the "ungraded" value of each check is still the frame without any grade
    img = preview(frame("crowd"), 480)
    start = "cierra 0.1\nnode up = exposure(stops: 1)\ngraph up\n"
    plain = render(load("cierra 0.1\n"), img)
    probes, regs = probes_for("darker"), regions(plain)
    want = measure_notes(probes, plain, regs)["darker"]
    provider = _Small(["```cierra\ncierra 0.1\nnode up = exposure(stops: 0.5)\ngraph up\n```\nmessage: less\n"])
    _, recs = grade_loop("darker", img, provider, History(tmp_path / "h"), start=start, max_iters=1, width=480,
                         log=lambda *_: None)
    row = recs[0]["note_checks"][0]
    assert abs(row["ungraded"] - round(want, 4)) < 1e-4 and row["previous"] > row["ungraded"]
