"""The grading loop: text notes + image statistics -> program -> check -> render a preview -> measure -> revise.

The LLM never sees pixels: it gets the language reference (static, cached), the notes, the numbers from
analysis.stats / analysis.measure, the previous program and any checker errors. Every program that passes the checker
is committed to a History with the LLM's one-line reason and the measurements.

Providers are pluggable: anything with `complete(system, user) -> str` (and optionally `last_usage`, a dict with
model / tokens / seconds / cost_usd of the last call). make_provider() picks one:
  CIERRA_DRIVER      auto (default): an open-weight server when CIERRA_LLM_URL is set, else the Anthropic API when
                     ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN is set, else the deterministic rules driver;
                     "openweight", "anthropic" (e.g. an `ant auth login` profile) or "rules" force one.
  CIERRA_LLM_URL     base URL of an OpenAI-compatible server for open-weight models (vLLM, llama.cpp server, Ollama),
                     e.g. http://localhost:8000/v1; CIERRA_MODEL is then the served model name (e.g.
                     Qwen/Qwen2.5-7B-Instruct). CIERRA_GPU_USD_PER_HOUR (optional) prices its time for the budget;
                     CIERRA_LLM_TIMEOUT (seconds per request, default 300; raise it for CPU inference).
  CIERRA_MODEL       claude-sonnet-5-5 (default) | claude-opus-5-5 (quality) | claude-haiku-5-5 (cheap)
  CIERRA_EFFORT      low (default) | medium | high
  CIERRA_BUDGET_USD  spending cap per loop run (default 2.00); when reached the loop continues with the rules driver.
Credentials come only from the environment through the official SDK; code never reads, writes or logs them.
RulesProvider (offline, deterministic) and ScriptedProvider (fixed answers) never need a key, so tests and CI don't.
"""

import json
import os
import re
import time

import numpy as np

from . import analysis
from . import colour as C
from .checker import check, lint, load
from .engine import render
from .errors import CheckError, CierraError
from .format import format_program
from .grammar import loop_grammar
from .parser import parse
from .ops import BLEND_MODES, INPUTS, MASKS, OPS, OUTPUTS

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_BUDGET_USD = 2.0
# USD per million tokens (list prices, 2026-10): input, output, cache read. Cache writes cost 1.25x input.
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20), "claude-sonnet-5-5": (2.0, 10.0, 0.20),
          "claude-haiku-5-5": (0.10, 0.50, 0.01)}


# ---------------------------------------------------------------- the language reference (system prompt)

def _param_line(key, p):
    if p.kind in ("num", "int"):
        rng = f"{p.kind} {p.lo:g}..{p.hi:g}"
    elif p.kind == "enum":
        rng = "one of " + "|".join(p.choices)
    elif p.kind in ("rgb", "pair"):
        rng = f"{p.kind} each {p.lo:g}..{p.hi:g}"
    elif p.kind == "range":
        rng = f"range lo..hi in {p.lo:g}..{p.hi:g}"
    elif p.kind == "points":
        cols = ", ".join(f"{c[2]} {c[0]:g}..{c[1]:g}" for c in p.cols + p.opt_cols)
        rng = f"points {p.min_len}..{p.max_len} of [{cols}]" + (" (last optional)" if p.opt_cols else "")
    elif p.kind == "ints":
        rng = f"list of ints {p.lo}..{p.hi}"
    else:
        rng = p.kind
    dflt = "required" if p.required else (f"default {p.default}" if p.default is not None else "optional")
    unit = f" [{p.unit}]" if p.unit else ""
    return f"    {key}: {rng}{unit}, {dflt}. {p.doc}".rstrip()


def language_reference():
    lines = [
        "You write grading programs in Cierra 0.1, a small checked colour-grading language. Reply with exactly one",
        "program in a ```cierra fenced block, then one line `message: <why this version>` (what you changed and why,",
        "for the colourist's history). No other code. You never see the image, only the numbers you are given.",
        "",
        "Keys: `key NAME = <mask>` names a mask (made again where used); in a mask, key(NAME, gain: 1, offset: 0) is",
        "that key, or a node's own key carried from where that node ran (it must run once, before the reader).",
        "`not key(face)` is everything face did not select. Combine keys with and (multiply), or (max), + (add),",
        "- (subtract); mixing or / + / - needs brackets.",
        "",
        "Grammar (one statement per line; # comments):",
        "  cierra 0.1                                  (first line)",
        '  name "..."   about "..."',
        "  input " + "|".join(INPUTS),
        "  output aces2(display: srgb|rec1886) | display_encode(eotf: srgb|gamma24) | acescct()",
        "  node <id> = <op>(param: value, ...)         values: numbers, true/false, identifiers, [r, g, b],",
        "                                              ranges lo..hi, [[x, y], ...] points,",
        "                                              keys(linear|step|smooth, <frame>: <value>, ...)",
        "    mask <mask>(...) [and|or <mask>(...)] | not <mask>(...)   (on the line after a node)",
        "    mix 0..1                                  label \"...\"     off",
        "  graph a -> b -> parallel(c, d) -> layer(e, f, mode: " + "|".join(BLEND_MODES) + ", opacity: 0.5)",
        "  graph layer(pass, glow, mode: screen)       pass = the unchanged input (a layer bottom, a branch)",
        "  photo_graph <...>                           runs first for srgb_photo / linear_photo (photo_to_scene() first)",
        "  (no graph line = identity: an output-only program)",
        "",
        "Working values are ACEScct code values (cv): 18 % grey = 0.4136, 1 stop = 0.0571 cv, black ~0.073.",
        "Keep grades gentle: 0.01 cv wheel offsets and 0.05 saturation steps are visible.",
        "",
        op_table(),
        "",
        "Rules:",
    ]
    from .examples import GUIDANCE, INVALID, VALID
    lines += [f"  - {g}" for g in GUIDANCE]
    lines += ["", "Examples (valid):"]
    for title, src in VALID:
        lines += [f"# {title}", "```cierra", src.rstrip(), "```"]
    lines += ["", "Mistakes the checker rejects (with the start of its message):"]
    for title, src, msg in INVALID:
        lines.append(f"  - {title}: {src.splitlines()[1].strip()}  ->  {msg}")
    lines += ["", "Reply format example:", "```cierra", VALID[1][1].rstrip(), "```",
              "message: warmer and a little richer, contrast pivoted low so blacks stay clean"]
    return "\n".join(lines)


SHORT_OPS = ("exposure", "white_balance", "wheels", "contrast", "saturation", "curve", "hue_vs_hue", "hue_vs_sat",
             "lum_vs_sat", "highlights_shadows")
SHORT_MASKS = ("skin", "luma", "qualifier")
SHORT_HINTS = [
    "Warm only the faces: put `mask skin()` on the node; a global white_balance warms the sky and everything else too.",
    "Keep the blacks clean: never lower exposure for it. contrast() moves the darkest pixels too; to add contrast "
    "and keep blacks, use a luma curve with points at black [0.0729, 0.0729] and grey [0.4136, 0.4136], one point "
    "slightly below the diagonal between them and one slightly above it after grey.",
    "Soft highlights: end that curve at or below [1, 1] (e.g. [1, 0.97]) so the top is not pushed brighter.",
    "Less green in foliage: hue_vs_sat(points: [[206, 0.8, 40]]) changes greens only (skin is near 141).",
    "'A little' means small: contrast 1.03-1.08, temperature 0.05-0.1, saturation 0.9-1.1.",
    "Read 'Note checks' in the numbers: fix every note marked WRONG WAY; keep the ones marked right way.",
]


def language_reference_short():
    """A compact reference for small open-weight models (about a quarter of the full one): the core ops and masks
    from the same op table (ranges cannot drift), note-to-op hints, one example. CPU time is dominated by the prompt."""
    lines = [
        "You write colour grades in Cierra 0.1, a small checked language. Reply with one program in a ```cierra block, "
        "then a line `message: <what you changed>`. You never see the image, only numbers.",
        "",
        "Syntax: first line `cierra 0.1`; `output aces2(display: srgb)`; one node per line "
        "`node <id> = <op>(param: value, ...)`; optional next line `  mask <mask>(...)` (use `not` / `and` / `or`) or "
        "`  mix 0.5`; last line `graph a -> b -> c`. Values: numbers, [r, g, b], lo..hi ranges, [[x, y], ...] points.",
        "Working values are ACEScct: 18 % grey = 0.4136, black = 0.0729, +1 stop = +0.057.",
        "",
        "Operations (param: kind range, default):",
    ]
    for name in SHORT_OPS:
        spec = OPS[name]
        params = "; ".join(_short_param(k, p) for k, p in spec.params.items())
        lines.append(f"  {name}({params}) - {spec.doc.split('. ')[0].rstrip('.')}")
    lines.append("Masks:")
    for name in SHORT_MASKS:
        spec = MASKS[name]
        params = "; ".join(_short_param(k, p) for k, p in spec.params.items() if k not in ("blur_px", "feather_px"))
        lines.append(f"  {name}({params})")
    lines += ["", "Hints:"] + [f"  - {h}" for h in SHORT_HINTS]
    lines += ["", "Example:", "```cierra", "cierra 0.1", "output aces2(display: srgb)",
              "node skin_glow = saturation(amount: 1.08)", "  mask skin()",
              "node calm_greens = hue_vs_sat(points: [[206, 0.85, 40]])",
              "node shape = curve(points: [[0.0729, 0.0729], [0.3, 0.295], [0.4136, 0.4136], [0.6, 0.605], [1, 0.98]])",
              "graph skin_glow -> calm_greens -> shape", "```",
              "message: richer skin, calmer greens, a touch of contrast that keeps black and the top in place"]
    return "\n".join(lines)


def _short_param(key, p):
    if p.kind in ("num", "int"):
        rng = f"{p.lo:g}..{p.hi:g}"
    elif p.kind == "enum":
        rng = "|".join(p.choices)
    elif p.kind in ("rgb", "pair"):
        rng = f"[{p.kind}] {p.lo:g}..{p.hi:g}"
    elif p.kind == "range":
        rng = f"lo..hi in {p.lo:g}..{p.hi:g}"
    elif p.kind == "points":
        rng = "[[" + ", ".join(c[2] for c in p.cols + p.opt_cols) + "], ...]"
    else:
        rng = p.kind
    dflt = p.default
    if isinstance(dflt, tuple):
        dflt = "[" + ", ".join(f"{x:g}" for x in dflt) + "]"
    elif isinstance(dflt, float):
        dflt = f"{dflt:g}"
    d = "required" if p.required else (f"={dflt}" if p.default is not None and p.kind != "points" else "")
    return f"{key}: {rng} {d}".rstrip()


def op_table():
    """Operations, masks and outputs with every parameter: type, range, unit, default."""
    lines = ["Operations:"]
    for name, spec in OPS.items():
        flags = "pointwise" if spec.pointwise else ("position" if spec.position else "spatial")
        lines.append(f"  {name}()  [{spec.category}, {flags}] {spec.doc}")
        lines += [_param_line(k, p) for k, p in spec.params.items()]
    lines.append("")
    lines.append("Masks (qualifiers):")
    for name, spec in MASKS.items():
        lines.append(f"  {name}()  {spec.doc}")
        lines += [_param_line(k, p) for k, p in spec.params.items()]
    lines.append("")
    lines.append("Outputs:")
    for name, spec in OUTPUTS.items():
        lines.append(f"  {name}()  {spec.doc}")
    return "\n".join(lines)


# ---------------------------------------------------------------- providers

class AnthropicProvider:
    """Claude through the official Anthropic SDK. Credentials come from the SDK's own resolution (ANTHROPIC_API_KEY /
    ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile); nothing is read, written or logged here. Model from
    CIERRA_MODEL. Each call's model, tokens, seconds and estimated cost are kept in `last_usage`."""

    def __init__(self, model=None, effort=None, max_tokens=16000):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=1)
        self.model = model or os.environ.get("CIERRA_MODEL") or DEFAULT_MODEL
        self.effort = effort or os.environ.get("CIERRA_EFFORT") or DEFAULT_EFFORT
        self.max_tokens, self.last_usage = max_tokens, {}

    def complete(self, system, user, temperature=None):
        # temperature is ignored: current Claude models reject sampling parameters (the re-ask relies on its text)
        t0 = time.monotonic()
        resp = self.client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            thinking={"type": "adaptive"}, output_config={"effort": self.effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}])
        self.last_usage = usage_record(getattr(resp, "model", None) or self.model, resp.usage, time.monotonic() - t0)
        if resp.stop_reason == "refusal":
            raise RuntimeError("the model declined this request")
        return "".join(b.text for b in resp.content if b.type == "text")


def usage_record(model, usage, seconds):
    """Tokens and an estimated cost (PRICES) of one call; unknown models are priced as the default model."""
    get = lambda k: int(getattr(usage, k, 0) or 0)  # noqa: E731
    inp, out = get("input_tokens"), get("output_tokens")
    c_read, c_write = get("cache_read_input_tokens"), get("cache_creation_input_tokens")
    pi, po, pc = PRICES.get(model, PRICES[DEFAULT_MODEL])
    cost = (inp * pi + out * po + c_read * pc + c_write * pi * 1.25) / 1e6
    return {"model": model, "input_tokens": inp, "output_tokens": out, "cache_read_tokens": c_read,
            "cache_write_tokens": c_write, "seconds": round(seconds, 2), "cost_usd": round(cost, 5)}


# Deterministic offline driver: words in the notes -> fixed, gentle nodes. Not clever, but always available, free,
# and the same answer every time (tests, CI, no key, budget reached).
RULES = [
    (r"\b(warm|warmer|healthier|richer)\s+(faces?|skin)", "skin_warmth",
     "saturation(amount: 1.08)\n  mask skin(width: 30)"),
    (r"\bwarm(er)?\b(?!\s+(faces?|skin))", "warm", "white_balance(temperature: 0.12)"),
    (r"\b(cool|cooler|colder)\b", "cool", "white_balance(temperature: -0.12)"),
    (r"\b(bright|brighter|lift|lighter)\b", "brighter", "exposure(stops: 0.3)"),
    (r"\b(dark|darker|moodier)\b", "darker", "exposure(stops: -0.3)"),
    (r"\b(more\s+(saturation|colou?r)|richer(?!\s+(faces?|skin))|vivid)\b", "richer",
     "saturation(amount: 1.1, protect_skin: 0.5)"),
    (r"\b(less\s+(saturation|colou?r)|muted|desaturated?)\b", "muted", "saturation(amount: 0.88)"),
    (r"\b(more contrast|punchier|punch)\b", "contrast_up", "contrast(amount: 1.08, pivot: 0.38)"),
    (r"\b(less contrast|softer|flatter)\b", "contrast_down", "contrast(amount: 0.93, pivot: 0.38)"),
    (r"\bless green\b", "less_green", "hue_vs_sat(points: [[206, 0.85, 40]])"),
]


class RulesProvider:
    """Offline driver: keywords in the colourist's notes (RULES) become nodes. Model-free and deterministic.
    When the prompt carries a current program (the loop's "Current program:" block), the rules revise it: the
    program, its output, masks, photo_graph and graph are kept, and only rule nodes it does not have yet are appended
    after its graph (a budget fallback must not throw a model's work away). Without one, a new program is built."""

    last_usage = {"model": "rules", "cost_usd": 0.0}

    def complete(self, system, user, temperature=None):
        notes = re.search(r"^Notes from the colourist: (.*)$", user, re.M)
        notes = (notes.group(1) if notes else "").lower()
        if user.startswith("Translate the colourist's notes"):  # cierra.notes.translate_notes: rules only echo
            return notes
        cur = re.search(r"^Current program:\n```cierra\n(.*?)```", user, re.M | re.S)
        if cur:
            revised = self._revise(cur.group(1), notes)
            if revised is not None:
                return revised
        inp = re.search(r"^Input colour space: (\w+)", user, re.M)
        lines = ["cierra 0.1", f"input {inp.group(1) if inp else 'linear_rec709'}", "output aces2(display: srgb)", ""]
        used = []
        for pattern, nid, call in RULES:
            if re.search(pattern, notes):
                lines.append(f"node {nid} = {call}")
                used.append(nid)
        if used:
            lines += ["", "graph " + " -> ".join(used)]
        why = ("rules driver: " + ", ".join(used)) if used else "rules driver: no rule matched the notes"
        return "```cierra\n" + "\n".join(lines) + "\n```\nmessage: " + why + "\n"

    @staticmethod
    def _revise(src, notes):
        """The current program with the missing rule nodes appended, or None when it does not parse."""
        try:
            prog = load(src)
        except CierraError:
            return None
        text = format_program(prog).rstrip("\n").split("\n")
        graph = next((ln for ln in text if ln.startswith("graph ")), None)
        body = [ln for ln in text if not ln.startswith("graph ")]
        added = []
        for pattern, nid, call in RULES:
            if re.search(pattern, notes) and nid not in prog.nodes:
                body.append(f"node {nid} = {call}")
                added.append(nid)
        chain = ([f"({graph[len('graph '):]})"] if graph else []) + added
        if chain:
            body.append("graph " + " -> ".join(chain))
        why = ("rules driver: added " + ", ".join(added)) if added else "rules driver: nothing to add"
        return "```cierra\n" + "\n".join(body) + "\n```\nmessage: " + why + "\n"


class OpenWeightProvider:
    """An open-weight model behind an OpenAI-compatible /chat/completions endpoint (vLLM, llama.cpp server, Ollama).
    Standard library only (no client package). Greedy decoding (temperature 0) for repeatable answers. Usage records
    tokens and seconds; cost is the server's time at CIERRA_GPU_USD_PER_HOUR when set, else 0."""

    def __init__(self, url=None, model=None, max_tokens=4096, timeout=None):
        self.url = (url or os.environ.get("CIERRA_LLM_URL") or "").rstrip("/")
        if not self.url:
            raise ValueError("the open-weight driver needs CIERRA_LLM_URL (e.g. http://localhost:8000/v1)")
        self.model = model or os.environ.get("CIERRA_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
        self.prompt_profile = os.environ.get("CIERRA_PROMPT") or "short"  # small models: the short reference
        self.timeout = float(timeout or os.environ.get("CIERRA_LLM_TIMEOUT") or 300.0)  # seconds; CPU runs are slow
        self.max_tokens, self.last_usage = max_tokens, {}
        self.usd_per_hour = float(os.environ.get("CIERRA_GPU_USD_PER_HOUR") or 0.0)
        # program replies are sampled under cierra.grammar's GBNF (llama.cpp server `grammar` field); CIERRA_GRAMMAR=0
        # turns it off (a server that rejects the field)
        self.uses_grammar = os.environ.get("CIERRA_GRAMMAR", "1") != "0"

    def complete(self, system, user, temperature=None, grammar=None):
        import urllib.request
        req = {"model": self.model, "temperature": temperature or 0, "seed": 1,  # repeatable re-asks too
               "max_tokens": self.max_tokens,
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if grammar:
            req["grammar"] = grammar
        body = json.dumps(req).encode()
        req = urllib.request.Request(self.url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        seconds = time.monotonic() - t0
        u = data.get("usage") or {}
        self.last_usage = {"model": data.get("model") or self.model, "input_tokens": int(u.get("prompt_tokens", 0)),
                           "output_tokens": int(u.get("completion_tokens", 0)), "seconds": round(seconds, 2),
                           "cost_usd": round(seconds * self.usd_per_hour / 3600, 5)}
        return data["choices"][0]["message"]["content"] or ""


def make_provider(driver=None):
    """The provider CIERRA_DRIVER asks for (see the module doc); auto falls back to the rules driver without a key."""
    driver = driver or os.environ.get("CIERRA_DRIVER") or "auto"
    if driver == "rules":
        return RulesProvider()
    if driver == "openweight" or (driver == "auto" and os.environ.get("CIERRA_LLM_URL")):
        return OpenWeightProvider()
    if driver == "anthropic" or (driver == "auto" and (os.environ.get("ANTHROPIC_API_KEY")
                                                       or os.environ.get("ANTHROPIC_AUTH_TOKEN"))):
        return AnthropicProvider()
    if driver == "auto":
        return RulesProvider()
    raise ValueError(f"unknown CIERRA_DRIVER {driver!r} (auto, openweight, anthropic, rules)")


class ScriptedProvider:
    """Replays a fixed list of answers (tests and offline demos); records what it was asked."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def complete(self, system, user, temperature=None):
        self.calls.append(user)
        self.temperatures = getattr(self, "temperatures", []) + [temperature]
        if not self.answers:
            raise RuntimeError("ScriptedProvider ran out of answers")
        return self.answers.pop(0)


# ---------------------------------------------------------------- loop

_BLOCK = re.compile(r"```(?:cierra)?\s*\n(.*?)```", re.S)


def compact_stats(d):
    """The few numbers a small model can use (a fraction of analysis.stats)."""
    s = analysis.stats(d)
    t, c, k, v = s["tone"], s["clipping"], s["skin"], s["vectorscope"]
    return {"luma_p5_p50_p95": [t["p5"], t["p50"], t["p95"]], "white_p99.5": t["white_p99.5"],
            "crushed_black_pct": c["crushed_black_pct"], "clipped_white_pct": c["clipped_white_pct"],
            "cast": s["neutral_cast"].get("reads_as"), "chroma_p50": v["chroma_p50"],
            "skin_hue_offset_deg": k.get("hue_offset_deg_p50"), "skin_chroma": k.get("chroma_p50")}


def extract(text):
    """LLM reply -> (program source, message)."""
    m = _BLOCK.search(text)
    src = m.group(1) if m else text
    msg = re.search(r"^message:\s*(.+)$", text, re.M)
    return src, (msg.group(1).strip() if msg else "")


def preview(img, width=480):
    """Box-downsample (3, H, W) to about `width` pixels wide (deterministic)."""
    h, w = img.shape[1:]
    s = max(1, w // width)
    if s == 1:
        return img
    hh, ww = h // s * s, w // s * s
    return img[:, :hh, :ww].reshape(3, hh // s, s, ww // s, s).mean(axis=(2, 4))


NEUTRAL = """cierra 0.1
input {input}
output aces2(display: srgb)
"""


REASK_TEMPERATURE = 0.7


def _ask(provider, system, user, temperature=None, grammar=None):
    kw = {}
    if grammar and getattr(provider, "uses_grammar", False):
        kw["grammar"] = grammar
    if temperature is not None:
        kw["temperature"] = temperature
    if not kw:
        return provider.complete(system, user)
    try:
        return provider.complete(system, user, **kw)
    except TypeError:  # a provider without the argument
        return provider.complete(system, user)


def _check_reply(src):
    """check(parse(src)), plus the loop's own rule: the output must stay measurable (SDR Rec.709 or ACEScct)."""
    prog, warnings = check(parse(src))
    if not C.measurable(prog.output):
        raise CheckError(["output: the loop measures on an SDR Rec.709 display; keep output aces2(display: srgb or "
                          "rec1886) while grading"])
    return prog, warnings


def ungraded(prog):
    """The same program without its graph: same input, photo decode and output, so a comparison measures the grade
    alone (cierra.qc, cierra.tuner, tools/eval_heldout.py)."""
    import copy
    base = copy.copy(prog)
    base.graph = None
    return base


def loop_status(records):
    """"done" when the best accepted version has no damage flags, "needs_human" when it has, "no_program" when
    nothing was accepted (the best is the undamaged version with most notes met, else the damaged one)."""
    acc = [r for r in records if r.get("accepted")]
    if not acc:
        return "no_program"
    return "done" if any(not r.get("needs_human") for r in acc) else "needs_human"


def grade_loop(notes, image, provider, history, input="linear_rec709", reference=None, start=None, max_iters=4,
               target_de=1.0, width=480, log=print, host=None, aux=None, budget_usd=None, revise=None):
    """Run the loop. image: (3, H, W) in `input`; reference: optional display image (same framing) of the look to
    reach; host: what the host provides, told to the model and used for linting (frames = clip length, planes,
    segment_ids with their meanings in segment_names, targets such as ["dctl", "lut"]); aux: the planes themselves
    ({"depth": (H, W) metres, "segments": (H, W) ints}, at the preview size). Returns (best Program or None,
    list of iteration records). budget_usd (default CIERRA_BUDGET_USD or 2.00): once the provider's recorded cost
    reaches it, the remaining iterations use the rules driver.

    Revision (revise, default on; CIERRA_REVISE=0 turns it off): every round's prompt opens with the notes not met yet
    (cierra.notes note checks), the node causing each WRONG WAY note (found by switching nodes off one at a time) and
    what the next version must change or add. A reply identical to the current program is re-asked once at a higher
    temperature with that requirement repeated; identical again ends the loop. The loop also ends when every note is
    met. The best version is the one meeting the most notes (or, with a reference, the lowest ΔE00).

    Damage gate (cierra.qc): every accepted version is checked for clipping, crushing, milky blacks and posterisation
    against the same program without its graph; flags go into the record ("damage", "needs_human") and into the
    next round's feedback, a damaged version never ends the loop as "every note met", and an undamaged version is
    always preferred as the best. When every version is damaged the best one is still returned, as a review candidate:
    loop_status(records) says "needs_human"."""
    from .format import program_hash
    from .notes import attribute, failing, measure_notes, probes_for, regions, report, revision_text, unmeasured
    from .qc import damage
    if revise is None:
        revise = os.environ.get("CIERRA_REVISE", "1") != "0"
    host = {"input": input, **(host or {})}
    budget = float(budget_usd if budget_usd is not None else os.environ.get("CIERRA_BUDGET_USD", DEFAULT_BUDGET_USD))
    spent = 0.0
    img = preview(np.asarray(image, dtype=np.float64), width)
    ref = None if reference is None else preview(np.asarray(reference, dtype=np.float64), width)
    if ref is not None and ref.shape != img.shape:
        raise ValueError(f"reference preview {ref.shape} does not match the image preview {img.shape}")

    def render_fn(prog):
        return render(prog, img, input=input, px_scale=img.shape[-1] / prog.reference_width, aux=aux)

    current = load(start) if isinstance(start, str) else (start or load(NEUTRAL.format(input=input)))
    if not C.measurable(current.output):
        raise ValueError("the loop measures on an SDR Rec.709 display: start from output aces2(display: srgb | "
                         "rec1886) and switch the output for the P3 / HDR deliverable afterwards")
    out = render_fn(current)
    profile = getattr(provider, "prompt_profile", None) or os.environ.get("CIERRA_PROMPT") or "full"
    stats_fn = compact_stats if profile == "short" else analysis.stats
    # note checks: probes on regions of the ungraded frame (the start program without its graph), compared round by
    # round; "previous" starts from the start program itself
    base_out = render_fn(ungraded(current))
    probes, regs = probes_for(notes), regions(base_out)
    base_vals = measure_notes(probes, base_out, regs)
    prev_vals = measure_notes(probes, out, regs)
    feedback = {"stats_now": stats_fn(out), "measure_now": analysis.measure(out, ref)}
    if probes:
        feedback["note_checks"] = [f"{row[2]} ({row[5]}): ungraded {round(base_vals[row[1]], 4)}" for row in probes]
    missing = unmeasured(notes)
    if missing:
        feedback["notes_not_measured"] = missing
    if ref is not None:
        feedback["reference_stats"] = stats_fn(ref)
    system = language_reference_short() if profile == "short" else language_reference()
    records, errors, best, revision, must = [], None, None, "", ([], [])
    for it in range(1, max_iters + 1):
        user = "\n".join([
            *([revision, ""] if revise and revision else []),
            f"Notes from the colourist: {notes}",
            f"Input colour space: {input}. Iteration {it} of {max_iters}.",
            "Host context (use only what is listed here): " + json.dumps(host, sort_keys=True),
            "Current program:", "```cierra", format_program(current).strip(), "```",
            "Measurements of the current program's preview (numbers only):",
            json.dumps(feedback, indent=None if profile == "short" else 1, sort_keys=True),
            *([f"Your previous answer was rejected by the checker; fix these:\n{errors}"] if errors else []),
            "Write the next version of the program."])
        if spent >= budget and not isinstance(provider, RulesProvider):
            log(f"budget reached ({spent:.4f} of {budget:.2f} USD): continuing with the rules driver")
            provider = RulesProvider()
        gbnf = loop_grammar(current, profile) if getattr(provider, "uses_grammar", False) else None
        reply = _ask(provider, system, user, grammar=gbnf)
        usage = dict(getattr(provider, "last_usage", None) or {})
        if gbnf:
            usage["grammar"] = True
        spent += float(usage.get("cost_usd", 0.0))
        src, message = extract(reply)
        try:
            prog, warnings = _check_reply(src)
        except CierraError as e:
            errors = str(e)
            records.append({"iteration": it, "accepted": False, "errors": errors, "usage": usage})
            log(f"iteration {it}: rejected: {errors.splitlines()[0]}")
            continue
        errors = None
        reasked = False
        if program_hash(prog) == program_hash(current) and revise and revision:
            req = []
            if must[0]:
                req.append("change node(s) " + ", ".join(must[0]))
            if must[1]:
                req.append("add node(s) for: " + "; ".join(must[1]))
            again = (user + "\nYou sent back the same program. It must be different: "
                     + (" and ".join(req) if req else "fix the notes listed at the top") + ".")
            if spent >= budget and not isinstance(provider, RulesProvider):  # the cap holds within a round too
                log(f"budget reached ({spent:.4f} of {budget:.2f} USD) before the re-ask: using the rules driver")
                provider = RulesProvider()
            log(f"iteration {it}: same program; re-asking once at temperature {REASK_TEMPERATURE}")
            reply = _ask(provider, system, again, temperature=REASK_TEMPERATURE, grammar=gbnf)
            u2 = dict(getattr(provider, "last_usage", None) or {})
            spent += float(u2.get("cost_usd", 0.0))
            usage = {"first": usage, "reask": u2}
            reasked = True
            src, message = extract(reply)
            try:
                prog, warnings = _check_reply(src)
            except CierraError as e:
                errors = str(e)
                records.append({"iteration": it, "accepted": False, "errors": errors, "usage": usage, "reasked": True})
                log(f"iteration {it}: re-ask rejected: {errors.splitlines()[0]}")
                continue
        if program_hash(prog) == program_hash(current):
            records.append({"iteration": it, "accepted": False, "no_change": True, "usage": usage,
                            "reasked": reasked})
            log(f"iteration {it}: no change; stopping")
            break
        warnings = warnings + lint(prog, host)
        out = render_fn(prog)
        dmg = damage(render_fn(ungraded(prog)), out)["flags"]
        m = analysis.measure(out, ref)
        entry = history.commit(prog, message, m, parent_prog=current)
        now_vals = measure_notes(probes, out, regs)
        rows, note_lines = report(probes, base_vals, prev_vals, now_vals)
        prev_vals = now_vals
        bad = failing(rows)
        causes = attribute(prog, probes, render_fn, regs, now_vals) if (revise and bad) else {}
        revision, must_change, must_add = revision_text(probes, rows, causes) if revise else ("", [], [])
        must = (must_change, must_add)
        score = len(rows) - len(bad)
        rec = {"iteration": it, "accepted": True, "hash": entry["hash"], "message": message, "measure": m,
               "note_checks": rows, "notes_met": f"{score}/{len(rows)}", "causes": causes, "reasked": reasked,
               "diff": entry["diff"], "warnings": warnings, "usage": usage, "spent_usd": round(spent, 5),
               "damage": dmg, "needs_human": bool(dmg)}
        records.append(rec)
        de = m.get("delta_e00_to_reference", {}).get("mean")
        log(f"iteration {it}: v{entry['id']} {entry['hash']} notes met {score}/{len(rows)}"
            + (f" dE00 mean {de}" if de is not None else ""))
        if dmg:
            log(f"iteration {it}: NEEDS HUMAN: " + "; ".join(dmg))
        clean = not dmg
        if ref is not None:
            key = (clean, -de)
            better = best is None or key > best[1]
        else:
            key = (clean, score)
            better = best is None or key >= best[1]  # ties: the later version
        if better:
            best = (prog, key)
        current = prog
        feedback = {"stats_now": stats_fn(out), "measure_now": m}
        if note_lines:
            feedback["note_checks"] = note_lines
        if missing:
            feedback["notes_not_measured"] = missing
        if warnings:
            feedback["checker_warnings"] = warnings
        if dmg:
            feedback["damage_fix_first"] = dmg
        if ref is not None:
            feedback["reference_stats"] = stats_fn(ref)
        if de is not None and de <= target_de and clean:
            break
        if probes and not bad and ref is None and clean:
            log(f"iteration {it}: every note is met; stopping")
            break
    return (best[0] if best else None), records
