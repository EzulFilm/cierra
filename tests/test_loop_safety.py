"""Loop safety: damage gates the loop and the tuner, the rules driver revises
the current program, baselines keep the program's own output, the budget cap holds before a re-ask. The host rollback
and verification findings are in test_host_grading_app.py."""

import numpy as np

from cierra import load
from cierra.engine import render
from cierra.format import format_program
from cierra.history import History
from cierra.llm import RulesProvider, ScriptedProvider, grade_loop, loop_status, ungraded
from cierra.tuner import tune
from conftest import frame, small

MILKY = ("```cierra\ncierra 0.1\nnode milk = wheels(master_lift: 0.12)\nnode brighter = exposure(stops: 0.3)\n"
         "graph milk -> brighter\n```\nmessage: lifted\n")
CLEAN = "```cierra\ncierra 0.1\nnode brighter = exposure(stops: 0.3)\ngraph brighter\n```\nmessage: clean\n"


def test_a_damaged_version_is_recorded_never_finished_and_loses_to_a_clean_one(tmp_path):
    img = small(frame("pens"), 4)
    best, recs = grade_loop("brighter", img, ScriptedProvider([MILKY, CLEAN]), History(tmp_path / "h"),
                            max_iters=3, log=lambda m: None, revise=False)
    assert recs[0]["accepted"] and recs[0]["needs_human"] and any("milky" in d for d in recs[0]["damage"])
    assert len(recs) == 2  # "brighter" was met in round 1 but the version was damaged: the loop went on
    assert not recs[1]["needs_human"] and "milk" not in best.nodes and loop_status(recs) == "done"


def test_only_damaged_versions_come_back_as_a_review_candidate(tmp_path):
    img = small(frame("pens"), 4)
    best, recs = grade_loop("brighter", img, ScriptedProvider([MILKY, MILKY]), History(tmp_path / "h"),
                            max_iters=2, log=lambda m: None, revise=False)
    assert best is not None and "milk" in best.nodes and loop_status(recs) == "needs_human"


def test_tuner_reports_damage():
    start = load("cierra 0.1\nnode milk = wheels(master_lift: 0.12)\ngraph milk\n")
    _, info = tune(start, "brighter", frame("window_1"), explain=False)
    assert info["needs_human"] and any("milky" in d for d in info["damage"])


def test_rules_driver_revises_the_current_program():
    cur = ("cierra 0.1\noutput display_encode(eotf: gamma24)\nnode soft = wheels(master_lift: 0.02)\n"
           "  mask luma(range: 0.1..0.4)\nnode cool = white_balance(temperature: -0.1)\n"
           "graph parallel(soft, cool)\n")
    user = f"Notes from the colourist: cooler and brighter\nCurrent program:\n```cierra\n{cur}```\nWrite the next version."
    reply = RulesProvider().complete("", user)
    prog = load(reply.split("```cierra\n")[1].split("```")[0])
    text = format_program(prog)
    assert prog.output.name == "display_encode" and "mask luma" in text and "parallel(soft, cool)" in text
    assert set(prog.nodes) == {"soft", "cool", "brighter"}  # "cool" already there: not added twice
    assert text.rstrip().endswith("graph (parallel(soft, cool)) -> brighter") or "-> brighter" in text


def test_baselines_keep_the_programs_own_output():
    """A non-default output: the tuner's ungraded frame goes through that output too, so a do-nothing graph is not
    'warmer' just because display_encode and ACES 2.0 differ."""
    prog = load("cierra 0.1\noutput display_encode()\nnode n = exposure()\ngraph n\n")
    img = small(frame("chart"), 4)
    assert np.abs(render(ungraded(prog), img) - render(prog, img)).max() < 1e-12
    _, info = tune(prog, "warmer", img, explain=False)
    assert info["added_nodes"] == ["warm"]  # nothing moved warmth yet: same output on both sides


class _Paid(ScriptedProvider):
    def complete(self, system, user, temperature=None):
        self.last_usage = {"cost_usd": 0.40}
        return super().complete(system, user, temperature)


def test_budget_cap_holds_before_a_re_ask(tmp_path):
    """Round 1 (0.40) misses "warmer", round 2 (0.80 > 0.50 cap) sends the same program: the re-ask must go to the
    rules driver, not to the paid model (before the fix: a third paid call)."""
    p = _Paid([CLEAN, CLEAN, CLEAN])
    _, recs = grade_loop("warmer", small(frame("chart"), 4), p, History(tmp_path / "h"), max_iters=2,
                         log=lambda m: None, budget_usd=0.5, revise=True)
    assert recs[1].get("reasked") and len(p.calls) == 2
