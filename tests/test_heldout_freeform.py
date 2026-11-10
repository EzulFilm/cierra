"""The free-form held-out set, the note translation step and the tuner's bounds (offline: no model, no network)."""

import json

from cierra import load
from cierra.format import format_program
from cierra.llm import RulesProvider, ScriptedProvider
from cierra.notes import probes_for, translate_notes, unmeasured
from cierra.tuner import CAPS, tune
from conftest import ROOT, frame

CASES = json.loads((ROOT / "tests" / "data" / "heldout_freeform.json").read_text())["cases"]
# set B: 40 more, written blind (2026-10-29) by an author who had not read the prompt, its examples or the tuner
CASES += json.loads((ROOT / "tests" / "data" / "heldout_freeform_b.json").read_text())["cases"]


def test_freeform_notes_are_outside_the_check_vocabulary():
    for c in CASES:
        assert probes_for(c["notes"]) == [], c["id"]  # otherwise the set would not test translation


def test_freeform_checks_are_fully_measured():
    for c in CASES:
        parts = [p.strip() for p in c["checks"].split(",")]
        # every phrase measured on its own ("warmer faces", "keep the highlights soft" make two probes each)
        assert unmeasured(c["checks"]) == [] and all(probes_for(p) for p in parts), c["id"]


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
    from cierra.notes import AXES_EXAMPLES, COLOUR_ANCHORED_DOC

    def trigrams(t):
        w = [x.strip(".,:;'").lower() for x in t.split()]
        return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}

    held = [c["notes"] for c in CASES]
    for name in ("heldout.json", "heldout_freeform_b.json"):
        held += [c["notes"] for c in json.loads((ROOT / "tests" / "data" / name).read_text())["cases"]]
    for ex in [ex for ex, _ in AXES_EXAMPLES] + [COLOUR_ANCHORED_DOC]:
        assert all(not (trigrams(ex) & trigrams(h)) for h in held), ex
    # the translation guide was written after the later free-form sets existed: it is held apart from every set
    # that is present (the older examples above predate those sets)
    from cierra.notes import GUIDE, GUIDE_EXAMPLES
    for extra in sorted((ROOT / "tests" / "data").glob("heldout_freeform_?.json")):
        held += [c["notes"] for c in json.loads(extra.read_text())["cases"]]
    for ex in [ex for ex, _ in GUIDE_EXAMPLES] + GUIDE:
        assert all(not (trigrams(ex) & trigrams(h)) for h in held), ex


def test_colour_anchors_change_only_the_colour_axis_and_only_when_asked(monkeypatch):
    from cierra.notes import COLOUR_ANCHORED_DOC, _axes_prompt
    monkeypatch.delenv("CIERRA_COLOUR_ANCHORS", raising=False)
    plain = _axes_prompt("Fog.")
    assert COLOUR_ANCHORED_DOC not in plain
    monkeypatch.setenv("CIERRA_COLOUR_ANCHORS", "1")
    anchored = _axes_prompt("Fog.")
    assert anchored == plain.replace("saturation of the whole picture", COLOUR_ANCHORED_DOC)


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


def test_translation_guide_is_off_by_default_and_only_adds(monkeypatch):
    from cierra.notes import GUIDE_EXAMPLES, _axes_prompt
    monkeypatch.delenv("CIERRA_TRANSLATE_GUIDE", raising=False)
    off = _axes_prompt("x")
    monkeypatch.setenv("CIERRA_TRANSLATE_GUIDE", "1")
    on = _axes_prompt("x")
    assert "Trade words" not in off and "Trade words" in on and GUIDE_EXAMPLES[0][0] in on
    assert on.startswith(off.split("Example notes:")[0]) and on.rstrip().endswith("Notes from the colourist: x")


def test_translation_guide_is_the_default_for_open_weight_models_only(monkeypatch):
    from cierra.llm import OpenWeightProvider, RulesProvider
    from cierra.notes import guide_on, translate_notes
    monkeypatch.delenv("CIERRA_TRANSLATE_GUIDE", raising=False)
    local = OpenWeightProvider(url="http://127.0.0.1:9/v1")
    assert guide_on(local) and not guide_on(RulesProvider()) and not guide_on(ScriptedProvider([])) and not guide_on()
    hosted = ScriptedProvider(["warmth: same"])
    translate_notes("x", hosted)
    assert "Trade words" not in hosted.calls[0]
    as_local = ScriptedProvider(["warmth: same"])
    as_local.translate_guide = True
    translate_notes("x", as_local)
    assert "Trade words" in as_local.calls[0]
    monkeypatch.setenv("CIERRA_TRANSLATE_GUIDE", "0")  # the switch still wins, both ways
    assert not guide_on(local)
    monkeypatch.setenv("CIERRA_TRANSLATE_GUIDE", "1")
    assert guide_on(RulesProvider())
