"""The deterministic tuner and the held-out evaluation harness (offline: no model, no network)."""

import json
import subprocess
import sys
from pathlib import Path

from cierra import load
from cierra.format import format_program
from cierra.tuner import add_rule_nodes, tune
from conftest import ROOT, frame

NOTES = "Cooler and a bit darker for a night feel, less colour overall, keep the blacks clean."


def test_tuner_meets_the_notes_from_nothing():
    img = frame("window_1")
    tuned, info = tune(load("cierra 0.1\n"), NOTES, img)
    assert set(info["added_nodes"]) == {"cool", "darker", "muted"}  # a node for every note nothing moved
    assert all(v == 0 for v in info["deficits"].values()), info["deficits"]
    assert format_program(load(format_program(tuned))) == format_program(tuned)  # valid, canonical


def test_tuner_is_deterministic():
    img = frame("window_1")
    a, _ = tune(load("cierra 0.1\n"), NOTES, img)
    b, _ = tune(load("cierra 0.1\n"), NOTES, img)
    assert format_program(a) == format_program(b)


def test_tuner_fixes_wrong_numbers_and_keeps_structure():
    img = frame("pens")
    start = load("cierra 0.1\nnode cool = white_balance(temperature: 0.2)\nnode dim = exposure(stops: 0.5)\n"
                 "node calm = saturation(amount: 1.2)\ngraph cool -> dim -> calm\n")  # every number the wrong way
    tuned, info = tune(start, NOTES, img)
    assert info["added_nodes"] == [] and set(tuned.nodes) == {"cool", "dim", "calm"}
    assert tuned.nodes["cool"].call.args["temperature"] < 0 and tuned.nodes["dim"].call.args["stops"] < 0
    assert all(v == 0 for v in info["deficits"].values())
    # "a bit darker" is a slight note (weighted lower), so "less colour" is reached partly with saturation: it moves
    # down from the wrong-way 1.2, never up (on the pens the cooler, darker frame already reads less colourful, so it
    # can stay at 1.2)
    assert tuned.nodes["calm"].call.args["amount"] <= 1.2
    assert "0000000" not in format_program(tuned)  # tuned values are rounded


def test_tuner_leaves_a_grade_that_already_meets_the_notes():
    img = frame("window_1")
    good, _ = tune(load("cierra 0.1\n"), NOTES, img)
    again, info = tune(good, NOTES, img)
    assert format_program(again) == format_program(good) and info["added_nodes"] == []


def test_add_rule_nodes_appends_to_the_graph():
    prog, added = add_rule_nodes(load("cierra 0.1\nnode a = contrast(amount: 1.1)\ngraph a\n"), ["coolness", "blacks"])
    assert added == ["cool"] and format_program(prog).rstrip().endswith("graph a -> cool")


def test_eval_harness_offline(tmp_path):
    r = subprocess.run([sys.executable, "tools/eval_heldout.py", "--mode", "tuner", "--driver", "rules", "--label",
                        "pytest", "--cases", "night_window"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip().splitlines()[-1].split("-> ")[1]
    res = json.loads((Path(out) / "results.json").read_text())
    assert res["met"] == res["total"] == 4 and res["cases"][0]["tuner"]["evals"] > 0


def test_note_weights_mark_slight_notes():
    from cierra.tuner import SLIGHT, note_weights
    w = note_weights("cooler, slightly darker, a little more contrast")
    assert w == {"coolness": 1.0, "darker": SLIGHT, "contrast_up": SLIGHT}


def test_slight_note_gives_way_to_a_plain_one():
    """On window_0 warmer and less colour pull Cr-Cb opposite ways: marked slight, less colour gives way."""
    img = frame("window_0")
    _, plain = tune(load("cierra 0.1\n"), "warmer, less contrast, keep the highlights soft, less colour", img)
    _, slight = tune(load("cierra 0.1\n"), "warmer, less contrast, keep the highlights soft, slightly less colour",
                     img)
    # the plain note is met; when the two conflict, the slight one is what gives way
    assert slight["deficits"]["warmth"] == 0
    assert slight["deficits"]["muted"] >= plain["deficits"]["muted"]


def test_unmet_notes_are_explained():
    img = frame("chart")  # 5 % crushed already: darker + more contrast cannot keep the blacks clean
    _, info = tune(load("cierra 0.1\n"), "cooler, darker, more contrast, keep the blacks clean", img)
    assert "keep the blacks clean" in info["unmet"]
    assert all(v.startswith(("conflicts with", "out of reach")) for v in info["unmet"].values())
    _, ok = tune(load("cierra 0.1\n"), NOTES, frame("window_1"))
    assert ok["unmet"] == {}


def test_tuner_never_moves_what_no_note_measures():
    """Seen in testing: tint pushed to +0.3 made neo-noir "cooler" by turning it magenta; a "brighter" node was tuned to
    -0.15 stops. Frozen parameters stay put, the green-magenta balance is held, rule nodes keep their direction."""
    from cierra.tuner import FROZEN, _tint
    from cierra.engine import render
    img = frame("crowd")
    start = load("cierra 0.1\nnode cool = white_balance(temperature: -0.05, tint: 0.02)\n"
                 "node contrast_up = contrast(amount: 1.05, pivot: 0.4)\ngraph cool -> contrast_up\n")
    tuned, _ = tune(start, "cooler, darker, more contrast, keep the blacks clean", img)
    assert tuned.nodes["cool"].call.args["tint"] == 0.02 and tuned.nodes["contrast_up"].call.args["pivot"] == 0.4
    assert ("white_balance", "tint") in FROZEN
    small = img[:, ::8, ::8]
    assert abs(_tint(render(tuned, small)) - _tint(render(start, small))) < 0.006
    summer, _ = tune(load("cierra 0.1\n"), "warmer, slightly brighter, less colour, more contrast, "
                     "keep the highlights soft", frame("pens"))
    assert summer.nodes["brighter"].call.args.get("stops", 0.0) >= 0


def test_bound_lets_a_look_keep_its_own_values_but_not_go_further():
    """A vivid look with saturation 1.8, beyond the cap (1.4): a loop that starts from such a look may
    keep or soften it, not push it further, and a value the look does not have is capped as usual."""
    from cierra import load
    from cierra.tuner import bound
    look = load("cierra 0.1\nnode vivid = saturation(amount: 1.8)\ngraph vivid\n")

    def reply(sat):
        return load(f"cierra 0.1\nnode vivid = saturation(amount: {sat})\n"
                    "node tint = wheels(gamma: [0.1, 0, -0.1])\ngraph vivid -> tint\n")

    def sat(p):
        return p.nodes["vivid"].call.args["amount"]
    kept, _ = bound(reply(1.8), start=look)
    softer, _ = bound(reply(1.6), start=look)
    further, moved = bound(reply(1.95), start=look)
    assert (sat(kept), sat(softer), sat(further)) == (1.8, 1.6, 1.8)
    assert further.nodes["tint"].call.args["gamma"] == (0.05, 0.0, -0.05)
    assert any(m.startswith("vivid.amount: 1.95 -> 1.8") for m in moved), moved
    assert sat(bound(reply(1.8))[0]) == 1.4  # from neutral, the cap holds


def test_bound_keeps_a_curve_monotone_after_restoring_a_looks_values():
    """A look's curve point beyond the cap is restored; its neighbour from the reply must not then sit below it."""
    from cierra import load
    from cierra.tuner import bound
    look = load("cierra 0.1\nnode c = curve(points: [[0.3, 0.6], [0.4, 0.7]])\ngraph c\n")
    reply = load("cierra 0.1\nnode c = curve(points: [[0.3, 0.6], [0.4, 0.45]])\ngraph c\n")
    ys = [float(r[1]) for r in bound(reply, start=look)[0].nodes["c"].call.args["points"]]
    assert ys == sorted(ys), ys
