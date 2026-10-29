"""LLM drivers: deterministic offline rules driver, provider choice from the environment, per-call
usage / cost records, a budget cap that falls back to the rules driver. No network, no key."""

import types

import numpy as np
import pytest

from cierra.history import History
from cierra.llm import (PRICES, RulesProvider, ScriptedProvider, grade_loop, make_provider, usage_record)


def test_rules_driver_is_deterministic_and_checks(chart, tmp_path):
    img = np.ascontiguousarray(chart[:, ::4, ::4])
    best, recs = grade_loop("warmer, a bit brighter, less green, warmer faces", img, RulesProvider(),
                            History(tmp_path / "h"), max_iters=3, width=120, log=lambda *_: None)
    assert recs[0]["accepted"] and recs[1].get("no_change")  # second round has nothing new: the loop stops
    assert set(best.nodes) == {"skin_warmth", "warm", "brighter", "less_green"}
    assert recs[0]["usage"] == {"model": "rules", "cost_usd": 0.0}
    again = RulesProvider().complete("", "Notes from the colourist: warmer\nInput colour space: linear_rec709")
    assert again == RulesProvider().complete("", "Notes from the colourist: warmer\nInput colour space: linear_rec709")


def test_rules_driver_without_a_match_changes_nothing(chart, tmp_path):
    img = np.ascontiguousarray(chart[:, ::4, ::4])
    best, recs = grade_loop("make it feel like autumn", img, RulesProvider(), History(tmp_path / "h"), max_iters=2,
                            width=120, log=lambda *_: None)
    assert best is None and recs[0].get("no_change")


def test_make_provider_follows_the_environment(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CIERRA_DRIVER", "CIERRA_MODEL", "CIERRA_LLM_URL"):
        monkeypatch.delenv(k, raising=False)
    assert isinstance(make_provider(), RulesProvider)  # no key: offline
    monkeypatch.setenv("CIERRA_DRIVER", "rules")
    assert isinstance(make_provider(), RulesProvider)
    monkeypatch.setenv("CIERRA_DRIVER", "auto")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    monkeypatch.setenv("CIERRA_MODEL", "claude-haiku-5-5")
    p = make_provider()
    assert type(p).__name__ == "AnthropicProvider" and p.model == "claude-haiku-5-5"  # constructed, never called
    with pytest.raises(ValueError, match="CIERRA_DRIVER"):
        make_provider("openai")


def test_usage_record_prices():
    u = types.SimpleNamespace(input_tokens=1_000_000, output_tokens=100_000, cache_read_input_tokens=2_000_000,
                              cache_creation_input_tokens=0)
    r = usage_record("claude-opus-5-5", u, 1.234)
    assert r["cost_usd"] == pytest.approx(4.0 + 2.0 + 0.4) and r["seconds"] == 1.23
    assert usage_record("claude-haiku-5-5", u, 0)["cost_usd"] == pytest.approx(0.1 + 0.05 + 0.02)
    assert set(PRICES) == {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}


class _Costly(ScriptedProvider):
    """Scripted answers that each 'cost' 1 USD."""

    def complete(self, system, user):
        self.last_usage = {"model": "test", "cost_usd": 1.0}
        return super().complete(system, user)


def test_budget_cap_falls_back_to_rules(chart, tmp_path):
    img = np.ascontiguousarray(chart[:, ::4, ::4])
    answers = [f"```cierra\ncierra 0.1\nnode a = exposure(stops: {s})\ngraph a\n```\nmessage: v{s}\n"
               for s in (0.1, 0.2, 0.3)]
    logs = []
    _, recs = grade_loop("cooler", img, _Costly(answers), History(tmp_path / "h"), max_iters=4, width=120,
                         log=logs.append, budget_usd=1.5)
    models = [r["usage"].get("model") for r in recs]
    assert models[:2] == ["test", "test"] and models[2] == "rules"  # 2 USD spent >= 1.5: the rules driver takes over
    assert recs[1]["spent_usd"] == 2.0 and any("budget reached" in m for m in logs)


def test_cli_loop_runs_offline(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    from conftest import ROOT
    env = {k: v for k, v in __import__("os").environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
    r = subprocess.run([sys.executable, "-m", "cierra", "loop", "cooler and less green", "tests/data/frames/chart.npy",
                        str(tmp_path / "hist"), "--iters", "2"], cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    recs, _ = json.JSONDecoder().raw_decode(r.stdout[r.stdout.index("[\n"):])  # log lines, records, then the program
    assert recs[0]["accepted"] and recs[0]["usage"]["model"] == "rules"
    assert "node cool = white_balance(temperature: -0.12)" in r.stdout


def test_defaults(monkeypatch):
    # Sonnet 5.5 at effort low unless the environment says otherwise
    for k in ("CIERRA_MODEL", "CIERRA_EFFORT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    p = make_provider("anthropic")
    assert (p.model, p.effort) == ("claude-sonnet-5-5", "low")
    monkeypatch.setenv("CIERRA_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("CIERRA_EFFORT", "medium")
    p = make_provider("anthropic")
    assert (p.model, p.effort) == ("claude-opus-5-5", "medium")
