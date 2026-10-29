"""Deterministic tuner: set a program's numbers so the colourist's notes are met, with the smallest change.

The model (or the rules driver) chooses the structure: which operations and masks. The tuner then
1. adds a rule node (cierra.llm.RULES) for every note that nothing in the program moves yet;
2. runs a coordinate search over the numeric parameters (scalars, rgb components, the value column of curve and hue
   points; masks and keyframes are left alone): it tries +/- a step on each parameter, keeps the best improvement,
   halves the step when nothing improves, and stops when every note is met, the step is tiny, or the render budget is
   spent. Bounds: every value stays within CAPS of neutral (a gentle grade: no 0.5-stop white balance, no crushed
   black point), and curve values stay non-decreasing. Values the program starts outside the bounds are pulled in
   first (listed in info["clamped"]).
Objective: for each note check, how far it is from met (in units of its tolerance), plus a small penalty for moving
away from the starting values (gentle grades, and the model's intent kept where it already works). Same input, same
result: no randomness. Renders a small preview (default 240 px wide) for speed.
"""

import copy
import re

import numpy as np

from . import ast as A
from .checker import load
from .engine import render
from .format import format_program
from .llm import RULES, preview, ungraded
from .notes import measure_notes, probes_for, regions
from .ops import IDENTITY_DEFAULTS, OPS

# note-check key -> the RULES node that moves it (hold notes have none: they are kept by tuning the others)
PROBE_RULE = {"skin_warmth": "skin_warmth", "warmth": "warm", "coolness": "cool", "brighter": "brighter",
              "darker": "darker", "richer": "richer", "muted": "muted", "contrast_up": "contrast_up",
              "contrast_down": "contrast_down", "foliage_chroma": "less_green"}
POINT_OPS = ("curve", "hue_vs_hue", "hue_vs_sat", "hue_vs_lum", "lum_vs_sat", "sat_vs_sat")
DIAGONAL_OPS = ("curve", "sat_vs_sat")  # identity is y = x; values kept non-decreasing
# how far the tuner may move a value from neutral, in the parameter's units (judgement: the edge of a gentle grade).
# Parameters not listed (pivots, radii, protect_skin, ...) only keep their op range.
CAPS = {("exposure", "stops"): 1.0, ("white_balance", "temperature"): 0.3, ("white_balance", "tint"): 0.3,
        ("printer_lights", "red"): 6.0, ("printer_lights", "green"): 6.0, ("printer_lights", "blue"): 6.0,
        ("wheels", "lift"): 0.05, ("wheels", "gamma"): 0.05, ("wheels", "gain"): 0.05, ("wheels", "offset"): 0.05,
        ("wheels", "master_lift"): 0.05, ("wheels", "master_gamma"): 0.05, ("wheels", "master_gain"): 0.05,
        ("wheels", "master_offset"): 0.05, ("cdl", "slope"): 0.25, ("cdl", "offset"): 0.05, ("cdl", "power"): 0.25,
        ("cdl", "sat"): 0.4, ("contrast", "amount"): 0.3, ("saturation", "amount"): 0.4,
        ("highlights_shadows", "highlights"): 1.5, ("highlights_shadows", "shadows"): 1.5,
        ("curve", "points"): 0.1, ("sat_vs_sat", "points"): 0.05, ("hue_vs_hue", "points"): 20.0,
        ("hue_vs_sat", "points"): 0.4, ("hue_vs_lum", "points"): 0.05, ("lum_vs_sat", "points"): 0.4}
POINT_NEUTRAL = {"hue_vs_hue": 0.0, "hue_vs_sat": 1.0, "hue_vs_lum": 0.0, "lum_vs_sat": 1.0}


def _deficits(probes, vals, base):
    out = {}
    for row in probes:
        key, want, tol = row[1], row[6], row[7]
        d = vals[key] - base[key]
        if not np.isfinite(d):
            out[key] = 0.0
        elif want == 0:
            out[key] = max(0.0, d - 0.5 * tol) / tol
        else:
            out[key] = max(0.0, 2 * tol - want * d) / tol
    return out


# Parameters no note check measures: the tuner leaves them where the program put them. Moving them only ever helped
# a check by side effect (seen in testing: tint pushed to +0.3 made neo-noir "cooler" by turning it magenta; a contrast
# pivot moved to 0.09 to game mid-tone contrast).
FROZEN = {("white_balance", "tint"), ("contrast", "pivot"), ("saturation", "protect_skin"),
          ("highlights_shadows", "radius_px")}
# Rule nodes (cierra.llm.RULES ids, also when a model uses the same names) keep to their own side of neutral:
# In testing, a node named "brighter" was tuned to -0.15 stops when its slight note gave way to a held highlight.
RULE_SIGN = {("warm", "white_balance", "temperature"): +1, ("cool", "white_balance", "temperature"): -1,
             ("brighter", "exposure", "stops"): +1, ("darker", "exposure", "stops"): -1,
             ("richer", "saturation", "amount"): +1, ("muted", "saturation", "amount"): -1,
             ("contrast_up", "contrast", "amount"): +1, ("contrast_down", "contrast", "amount"): -1}
TINT_TOL = 0.003  # implicit hold: green-magenta balance of the frame (display G - (R + B) / 2) may move this much


def _tint(d):
    return float((d[1] - 0.5 * (d[0] + d[2])).mean())


def _knobs(prog):
    """(node id, param, index) for every tunable number, with its range (FROZEN parameters excluded)."""
    from .walk import graph_nodes
    out = []
    for node in graph_nodes(prog, prog.graph):
        if not node.enabled:
            continue
        for key, p in OPS[node.call.name].params.items():
            v = node.call.args.get(key)
            if isinstance(v, A.Keys) or v is None or (node.call.name, key) in FROZEN:
                continue
            if p.kind == "num":
                out.append((node.id, key, None, p.lo, p.hi))
            elif p.kind == "rgb" and not any(isinstance(x, A.Keys) for x in v):
                out += [(node.id, key, i, p.lo, p.hi) for i in range(3)]
            elif p.kind == "points" and node.call.name in POINT_OPS:
                lo, hi = p.cols[1][0], p.cols[1][1]
                out += [(node.id, key, i, lo, hi) for i in range(len(v))]
    return out


def _get(prog, knob):
    nid, key, i, _, _ = knob
    v = prog.nodes[nid].call.args[key]
    if i is None:
        return float(v)
    return float(v[i][1]) if isinstance(v[0], tuple) else float(v[i])


def _limits(prog, knob):
    """(lo, hi) for this value now: the op range, the cap around neutral, and (diagonal ops) its neighbours."""
    nid, key, i, lo, hi = knob
    call = prog.nodes[nid].call
    v = call.args[key]
    cap = CAPS.get((call.name, key))
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        neutral = float(v[i][0]) if call.name in DIAGONAL_OPS else POINT_NEUTRAL.get(call.name)
        if call.name in DIAGONAL_OPS:
            lo = max(lo, float(v[i - 1][1])) if i > 0 else lo
            hi = min(hi, float(v[i + 1][1])) if i + 1 < len(v) else hi
    else:
        p = OPS[call.name].params[key].default
        neutral = (p[i] if isinstance(p, tuple) else p) if call.name in IDENTITY_DEFAULTS else None
    if cap is not None and neutral is not None:
        lo, hi = max(lo, neutral - cap), min(hi, neutral + cap)
    sign = RULE_SIGN.get((nid, call.name, key))
    if sign is not None and neutral is not None:  # a node named for a direction never goes the other way
        lo, hi = (max(lo, neutral), hi) if sign > 0 else (lo, min(hi, neutral))
    return lo, hi


def _repair(prog, knobs):
    """Pull every value inside its bounds (diagonal curves made non-decreasing first). Returns what moved (beyond
    the 4-decimal rounding every value gets)."""
    moved = []
    for nid in {k[0] for k in knobs}:
        call = prog.nodes[nid].call
        if call.name in DIAGONAL_OPS:
            v, run = call.args["points"], -1e9
            fixed = []
            for r in v:
                run = max(run, float(r[1]))
                fixed.append((r[0], run, *r[2:]))
            call.args["points"] = tuple(fixed)
    for k in knobs:
        before = _get(prog, k)
        _set(prog, k, before)
        if abs(_get(prog, k) - before) > 1e-4:
            moved.append(f"{k[0]}.{k[1]}{'' if k[2] is None else f'[{k[2]}]'}: {before:g} -> {_get(prog, k):g}")
    return moved


def _set(prog, knob, x):
    nid, key, i, _, _ = knob
    lo, hi = _limits(prog, knob)
    x = round(float(min(max(x, lo), hi)), 4)  # readable values; 1e-4 is below every tolerance
    args = prog.nodes[nid].call.args
    v = args[key]
    if i is None:
        args[key] = x
    elif isinstance(v[0], tuple):  # points: the value column
        args[key] = tuple((r[0], x, *r[2:]) if j == i else r for j, r in enumerate(v))
    else:
        args[key] = tuple(x if j == i else c for j, c in enumerate(v))


def add_rule_nodes(prog, keys):
    """Append RULES nodes for these note-check keys (serially, after the existing graph)."""
    src = format_program(prog).rstrip("\n").split("\n")
    graph_line = next((ln for ln in src if ln.startswith("graph ")), None)
    body = [ln for ln in src if not ln.startswith("graph ")]
    chain = [graph_line[len("graph "):]] if graph_line else []
    added = []
    for key in keys:
        rule = next((r for r in RULES if r[1] == PROBE_RULE.get(key)), None)
        if rule is None or rule[1] in prog.nodes or rule[1] in added:
            continue
        body.append(f"node {rule[1]} = {rule[2]}")
        added.append(rule[1])
    if not added:
        return prog, []
    chain += added
    return load("\n".join(body) + "\ngraph " + " -> ".join(chain) + "\n"), added


SLIGHT = 0.3  # weight of a note marked "slightly ..." against a plain one when they cannot both be met


def note_weights(notes):
    """{probe key: weight} from the notes: 1 for a plain note, SLIGHT for one written "slightly ..." / "a bit ..."
    (the translation marks weak directions that way). The first mention of a key wins."""
    out = {}
    for part in re.split(r"[,.;]", notes or ""):
        w = SLIGHT if re.search(r"\b(slight(ly)?|a (little|bit))\b", part, re.I) else 1.0
        for row in probes_for(part):
            out.setdefault(row[1], w)
    return out


def tune(prog, notes, image, input="linear_rec709", width=240, max_evals=800, penalty=0.02, log=None,
         explain=True):
    """Return (tuned Program, info dict). image: the full frame in `input` (a preview is made here).
    Search: single-value moves; when none improves at a step size, moves of two values on different nodes together
    (the pairs that "warmer, highlights held" needs); then the step halves. explain: for each note still unmet, tune
    for that note alone; info["unmet"] says whether it conflicts with the other notes on this frame (met alone) or is
    out of reach within the bounds (not even alone)."""
    img = preview(np.asarray(image, dtype=np.float64), width)

    def render_fn(p):
        return render(p, img, input=input, px_scale=img.shape[-1] / p.reference_width)

    base_out = render_fn(ungraded(prog))  # the same input, photo decode and output: only the grade differs
    probes, regs = probes_for(notes), regions(base_out)
    base = measure_notes(probes, base_out, regs)
    weights = note_weights(notes)
    info = {"evals": 0, "added_nodes": [], "start": format_program(prog), "weights": weights}

    def cost(p, x0):
        info["evals"] += 1
        out = render_fn(p)
        vals = measure_notes(probes, out, regs)
        d = _deficits(probes, vals, base)
        move = sum(abs(_get(p, k) - x0[i]) / (k[4] - k[3]) for i, k in enumerate(knobs))
        # nothing else moved: the green-magenta balance stays where the starting program had it (no note asks for it)
        tint = max(0.0, abs(_tint(out) - tint0) - TINT_TOL) / TINT_TOL
        return sum(weights.get(key, 1.0) * v for key, v in d.items()) + tint + penalty * move, d

    # 1. a node for every note nothing moves yet
    vals = measure_notes(probes, render_fn(prog), regs)
    d0 = _deficits(probes, vals, base)
    still = [row[1] for row in probes if row[6] != 0 and d0[row[1]] > 0
             and abs(vals[row[1]] - base[row[1]]) <= row[7]]
    prog, added = add_rule_nodes(copy.deepcopy(prog), still)
    info["added_nodes"] = added

    # 2. coordinate search
    knobs = _knobs(prog)
    info["clamped"] = _repair(prog, knobs)
    x0 = [_get(prog, k) for k in knobs]
    tint0 = _tint(render_fn(prog))
    best, (best_cost, best_d) = prog, cost(prog, x0)
    step = 0.05
    while step >= 0.004 and info["evals"] < max_evals and sum(best_d.values()) > 0:
        improved = False
        for k in knobs:
            cur = _get(best, k)
            for sign in (+1, -1):
                trial = copy.deepcopy(best)
                _set(trial, k, cur + sign * step * (k[4] - k[3]))
                if _get(trial, k) == cur:
                    continue
                c, d = cost(trial, x0)
                if c < best_cost - 1e-9:
                    best, best_cost, best_d, improved = trial, c, d, True
                    break
            if info["evals"] >= max_evals:
                break
        if not improved:
            improved = _pair_step(best, knobs, step, cost, x0, best_cost, max_evals, info)
            if improved:
                best, best_cost, best_d = improved
        if not improved:
            step /= 2
    tuned = load(format_program(best))  # re-checked: every value is in range by construction
    from .qc import damage
    dmg = damage(render_fn(ungraded(tuned)), render_fn(tuned))["flags"]  # meeting the notes is not enough
    info.update({"damage": dmg, "needs_human": bool(dmg)})
    info.update({"deficits": {k: round(v, 3) for k, v in best_d.items()}, "cost": round(best_cost, 4),
                 "final": format_program(tuned)})
    if explain:
        info["unmet"] = _explain_unmet(tuned, probes, best_d, image, input, width, penalty)
    if log:
        log(f"tuner: {info['evals']} renders, added {added or 'nothing'}, deficits {info['deficits']}"
            + (f", unmet {info['unmet']}" if info.get("unmet") else ""))
    return tuned, info


def _pair_step(best, knobs, step, cost, x0, best_cost, max_evals, info):
    """One move of two values on different nodes together (each +/- step); the first that lowers the cost wins.
    Returns (program, cost, deficits) or None."""
    for a in range(len(knobs)):
        for b in range(a + 1, len(knobs)):
            ka, kb = knobs[a], knobs[b]
            if ka[0] == kb[0]:
                continue
            for sa in (+1, -1):
                for sb in (+1, -1):
                    if info["evals"] >= max_evals:
                        return None
                    trial = copy.deepcopy(best)
                    _set(trial, ka, _get(best, ka) + sa * step * (ka[4] - ka[3]))
                    _set(trial, kb, _get(best, kb) + sb * step * (kb[4] - kb[3]))
                    if _get(trial, ka) == _get(best, ka) or _get(trial, kb) == _get(best, kb):
                        continue
                    c, d = cost(trial, x0)
                    if c < best_cost - 1e-9:
                        return trial, c, d
    return None


def _explain_unmet(prog, probes, deficits, image, input, width, penalty):
    """{note label: why} for every note the tuned program still misses: tune for that note alone from the tuned
    program; met alone = it conflicts with the other notes on this frame; not even alone = out of reach within the
    bounds (CAPS)."""
    from .notes import PHRASES
    out = {}
    for row in probes:
        key = row[1]
        if deficits.get(key, 0) <= 0 or key not in PHRASES:
            continue
        _, alone = tune(prog, PHRASES[key], image, input=input, width=width, max_evals=200, penalty=penalty,
                        explain=False)
        if all(v == 0 for v in alone["deficits"].values()):
            others = [r[1] for r in probes if r[1] != key]
            out[PHRASES[key]] = "conflicts with " + ", ".join(PHRASES.get(o, o) for o in others) + " on this frame"
        else:
            out[PHRASES[key]] = "out of reach within the tuner's bounds"
    return out
