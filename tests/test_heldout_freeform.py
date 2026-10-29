"""The free-form held-out set, the note translation step and the tuner's bounds (offline: no model, no network)."""

import json

from cierra import load
from cierra.format import format_program
from cierra.llm import RulesProvider, ScriptedProvider
from cierra.notes import probes_for, translate_notes, unmeasured
from cierra.tuner import CAPS, tune
from conftest import ROOT, frame

CASES = json.loads((ROOT / "tests" / "data" / "heldout_freeform.json").read_text())["cases"]


def test_freeform_notes_are_outside_the_check_vocabulary():
    for c in CASES:
        assert probes_for(c["notes"]) == [], c["id"]  # otherwise the set would not test translation


def test_freeform_checks_are_fully_measured():
    for c in CASES:
        parts = [p.strip() for p in c["checks"].split(",")]
        assert unmeasured(c["checks"]) == [] and len(probes_for(c["checks"])) == len(parts), c["id"]


def test_translate_keeps_only_check_phrases():
    p = ScriptedProvider(["Cooler, darker, and a little less colour; also add film grain."])
    phrases, raw = translate_notes("late-autumn evening", p)
    assert phrases == "cooler, darker, less colour" and "grain" in raw
    assert p.calls[0].startswith("Translate the colourist's notes") and "late-autumn evening" in p.calls[0]


def test_rules_translation_only_echoes():
    assert translate_notes("Bleach bypass.", RulesProvider())[0] == ""
    assert translate_notes("Cooler and darker.", RulesProvider())[0] == "cooler, darker"


def test_tuner_pulls_extreme_values_inside_the_caps():
    img = frame("crowd")
    start = load("cierra 0.1\nnode cool = white_balance(temperature: -0.52)\n"
                 "node shape = curve(points: [[0.0729, -0.0121], [0.3, 0.38], [0.4136, 0.3286], [1, 0.98]])\n"
                 "graph cool -> shape\n")
    tuned, info = tune(start, "cooler, darker", img)
    assert tuned.nodes["cool"].call.args["temperature"] >= -CAPS[("white_balance", "temperature")]
    pts = tuned.nodes["shape"].call.args["points"]
    ys = [float(r[1]) for r in pts]
    assert ys == sorted(ys), ys  # non-decreasing
    assert all(abs(float(r[1]) - float(r[0])) <= CAPS[("curve", "points")] + 1e-9 for r in pts), pts
    assert any(m.startswith("cool.temperature") for m in info["clamped"])
    assert format_program(load(format_program(tuned))) == format_program(tuned)


def test_tuner_without_measurable_notes_changes_nothing_but_bounds():
    img = frame("pens")
    start = load("cierra 0.1\nnode s = saturation(amount: 1.1)\ngraph s\n")
    tuned, info = tune(start, "", img)
    assert format_program(tuned) == format_program(start) and info["added_nodes"] == []


def test_axes_translation_reads_one_direction_per_axis():
    reply = ("warmth: cooler\nbrightness: same\ncolour: less\ncontrast: More\nblacks: keep clean\n"
             "highlights: same\nfaces: same\ngreens: same\n")
    p = ScriptedProvider([reply])
    phrases, _ = translate_notes("Bleach bypass.", p)
    assert phrases == "cooler, less colour, more contrast, keep the blacks clean"
    assert "warmth (" in p.calls[0] and p.calls[0].rstrip().endswith("Notes from the colourist: Bleach bypass.")


def test_axes_examples_are_kept_apart_from_the_held_out_sets():
    from cierra.notes import AXES_EXAMPLES

    def trigrams(t):
        w = [x.strip(".,:;'").lower() for x in t.split()]
        return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

    held = [c["notes"] for c in CASES]
    held += [c["notes"] for c in json.loads((ROOT / "tests" / "data" / "heldout.json").read_text())["cases"]]
    for ex, _ in AXES_EXAMPLES:
        assert all(not (trigrams(ex) & trigrams(h)) for h in held), ex


def test_compare_intents():
    from cierra.notes import compare_intents
    r = compare_intents("less contrast, keep the blacks clean, darker", "cooler, more contrast, darker, "
                        "keep the blacks clean")
    assert r == {"hit": ["blacks", "darker"], "missed": ["coolness"], "reversed": ["contrast_up"], "extra": []}


def test_axes_translation_keeps_slight_directions():
    reply = "warmth: slightly warmer\nbrightness: same\ncolour: Slightly less\ncontrast: less\n"
    phrases, _ = translate_notes("x", ScriptedProvider([reply]))
    assert phrases == "slightly warmer, slightly less colour, less contrast"
    assert [row[1] for row in probes_for(phrases)] == ["warmth", "contrast_down", "muted"]
