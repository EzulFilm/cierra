"""The LLM loop with a scripted provider (no network): rejected answers are fed back, accepted ones are versioned
with readable diffs, the loop stops at the ΔE target; the prompt carries numbers, never pixels."""

import json

import numpy as np
import pytest

from cierra import colour as C
from cierra.analysis import stats
from cierra.history import History
from cierra.llm import ScriptedProvider, extract, grade_loop, language_reference
from cierra.engine import render
from cierra import load
from conftest import small

TARGET = """cierra 0.1
output aces2(display: srgb)
node warm = white_balance(temperature: 0.15)
node rich = saturation(amount: 1.1)
graph warm -> rich
"""


def answer(body, msg):
    return f"Here you go.\n```cierra\n{body}```\nmessage: {msg}\n"


def test_loop_rejects_then_converges(frames, tmp_path):
    img = small(frames["crowd"], 2)
    ref = render(load(TARGET), img)
    bad = answer("cierra 0.1\nnode warm = white_balance(temperature: 9)\ngraph warm\n", "too warm")
    first = answer(TARGET.replace("0.15", "0.05").replace("1.1", "1.0"), "start warmer")
    second = answer(TARGET, "more warmth and colour")
    provider = ScriptedProvider([bad, first, second])
    hist = History(tmp_path / "hist")
    best, recs = grade_loop("warmer, a little richer", img, provider, hist, reference=ref, max_iters=5,
                            target_de=0.05, width=240, log=lambda *_: None)
    assert [r["accepted"] for r in recs] == [False, True, True]
    assert "outside" in provider.calls[1]  # checker errors fed back
    assert recs[2]["measure"]["delta_e00_to_reference"]["mean"] < 0.05
    entries = hist.entries()
    assert len(entries) == 2 and entries[1]["parent"] == entries[0]["hash"]
    assert "~ node warm: white_balance temperature 0.05 -> 0.15" in entries[1]["diff"]
    assert entries[1]["message"] == "more warmth and colour"
    assert best is not None and hist.load(entries[1]).nodes["rich"].call.args["amount"] == 1.1


def test_prompt_has_numbers_not_pixels(frames, tmp_path):
    img = small(frames["chart"], 2)
    provider = ScriptedProvider([answer(TARGET, "x")])
    grade_loop("warmer", img, provider, History(tmp_path / "h"), max_iters=1, width=240, log=lambda *_: None)
    msg = provider.calls[0]
    assert len(msg) < 20_000
    payload = msg.split("numbers only):\n", 1)[1].split("\nWrite the next", 1)[0]
    json.loads(payload)  # the measurements are plain JSON numbers
    for op in ("saturation()", "qualifier()", "grain()"):
        assert op in language_reference()


def test_extract():
    src, msg = extract(answer("cierra 0.1\n", "why"))
    assert src == "cierra 0.1\n" and msg == "why"


def test_stats_and_delta_e(chart):
    d = render(load(TARGET), small(chart))
    s = stats(d)
    assert set(s) >= {"tone", "histogram", "waveform", "vectorscope", "neutral_cast", "skin", "clipping"}
    assert abs(sum(s["histogram"]["luma_bins_pct"]) - 100) < 0.1


def test_ciede2000_against_reference_data():
    # Sharma, Wu, Dalal (2005) test pairs 1, 7, 17, 25 and their published ΔE00
    pairs = [((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
             ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
             ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
             ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644)]
    for a, b, want in pairs:
        got = C.delta_e00(np.array(a).reshape(3, 1), np.array(b).reshape(3, 1))[0]
        assert abs(got - want) < 1e-4, (a, b, got, want)


@pytest.mark.parametrize("w", [1, 2, 3, 5])
def test_narrow_images(w, tmp_path):
    # statistics and a loop round on images narrower than the waveform's six bands
    from cierra.llm import RulesProvider
    img = np.full((3, 4, w), 0.18) * np.linspace(0.5, 1.5, w)[None, None, :]
    s = stats(render(load("cierra 0.1\n"), img))
    assert s["waveform"]["columns"] == min(6, w)
    best, recs = grade_loop("warmer", img, RulesProvider(), History(tmp_path / "h"), max_iters=1, log=lambda *_: None)
    assert recs
