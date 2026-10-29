"""The open-weight driver against a stand-in OpenAI-compatible server on localhost (no GPU, no model, no cost):
request shape, the grading loop end to end, usage / cost records, driver selection."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from cierra.history import History
from cierra.llm import OpenWeightProvider, grade_loop, make_provider

ANSWER = ("Here is the grade.\n```cierra\ncierra 0.1\noutput aces2(display: srgb)\n"
          "node warm = white_balance(temperature: 0.1)\ngraph warm\n```\nmessage: warmer\n")


@pytest.fixture()
def server():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            out = json.dumps({"model": body["model"], "choices": [{"message": {"role": "assistant", "content": ANSWER}}],
                              "usage": {"prompt_tokens": 7000, "completion_tokens": 300}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen
    httpd.shutdown()


def test_request_shape_and_usage(server, monkeypatch):
    url, seen = server
    monkeypatch.setenv("CIERRA_GPU_USD_PER_HOUR", "1.09")
    p = OpenWeightProvider(url=url, model="Qwen/Qwen2.5-7B-Instruct")
    assert p.complete("SYSTEM", "USER") == ANSWER
    path, body = seen[0]
    assert path == "/v1/chat/completions" and body["model"] == "Qwen/Qwen2.5-7B-Instruct" and body["temperature"] == 0
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    u = p.last_usage
    assert (u["input_tokens"], u["output_tokens"]) == (7000, 300) and u["cost_usd"] >= 0


def test_grammar_is_sent_for_program_turns_only(server, chart, tmp_path, monkeypatch):
    url, seen = server
    monkeypatch.setenv("CIERRA_GRAMMAR", "0")
    assert not OpenWeightProvider(url=url, model="m").uses_grammar  # can be turned off
    monkeypatch.delenv("CIERRA_GRAMMAR")
    p = OpenWeightProvider(url=url, model="m")  # on by default
    p.complete("S", "U")
    assert "grammar" not in seen[-1][1]  # only program turns of the loop carry it
    img = np.ascontiguousarray(chart[:, ::4, ::4])
    _, recs = grade_loop("warmer", img, p, History(tmp_path / "h"), max_iters=1, width=120, log=lambda *_: None)
    body = seen[-1][1]
    assert body["grammar"].startswith("root ::=") and "op-white-balance" in body["grammar"]
    assert recs[0]["accepted"] and recs[0]["usage"]["grammar"]


def test_loop_runs_on_the_open_weight_driver(server, chart, tmp_path):
    url, seen = server
    img = np.ascontiguousarray(chart[:, ::4, ::4])
    best, recs = grade_loop("warmer", img, OpenWeightProvider(url=url, model="m"), History(tmp_path / "h"),
                            max_iters=3, width=120, log=lambda *_: None)
    assert recs[0]["accepted"] and recs[0]["notes_met"] == "1/1" and len(recs) == 1  # every note met: stops
    assert best.nodes["warm"].call.args["temperature"] == 0.1
    assert recs[0]["usage"]["model"] == "m" and len(seen) == 1 and seen[0][1]["seed"] == 1


def test_driver_selection(server, monkeypatch):
    url, _ = server
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CIERRA_DRIVER", "CIERRA_MODEL", "CIERRA_LLM_URL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CIERRA_LLM_URL", url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-real-key")
    assert isinstance(make_provider(), OpenWeightProvider)  # an open-weight server wins over the paid API
    monkeypatch.delenv("CIERRA_LLM_URL")
    with pytest.raises(ValueError, match="CIERRA_LLM_URL"):
        make_provider("openweight")


def test_timeout_from_environment(monkeypatch):
    monkeypatch.setenv("CIERRA_LLM_TIMEOUT", "3600")
    assert OpenWeightProvider(url="http://127.0.0.1:1/v1").timeout == 3600.0
    monkeypatch.delenv("CIERRA_LLM_TIMEOUT")
    assert OpenWeightProvider(url="http://127.0.0.1:1/v1").timeout == 300.0
