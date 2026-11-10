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
# Highlight roll-off (CIERRA_ROLLOFF=1, off by default until measured): the +-0.1 cap round the diagonal pulled a
# model's shoulder curve back to the identity (a point at x 1.0 could not go below 0.9), which blocked the standard
# fix for clipped highlights (first strong-model run, 2026-11-09). With it on, a log-space curve point at x may sit
# down to 0.1 + ROLLOFF_SLOPE * (x - ROLLOFF_FROM) below the diagonal: 0.19 at x 0.7, 0.37 at x 1.0, 0.43 at 1.1.
ROLLOFF_FROM, ROLLOFF_SLOPE = 0.55, 0.6


def _rolloff_on():
    import os
    return os.environ.get("CIERRA_ROLLOFF", "0") == "1"


POINT_NEUTRAL = {"hue_vs_hue": 0.0, "hue_vs_sat": 1.0, "hue_vs_lum": 0.0, "lum_vs_sat": 1.0}


def _deficits(probes, vals, base):
    out = {}
    for row in probes:
        key, want, tol = row[1], row[6], row[7]
        d = vals[key] - base[key]
        if not np.isfinite(base[key]):  # not measurable on this frame (e.g. no faces): no pull either way
            out[key] = 0.0
        elif not np.isfinite(d):  # measurable before, not after (a NaN render): as far from met as it gets
            out[key] = 1e3
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


def _neutral(prog, knob):
    """The value that leaves the picture as it is for this knob, or None (pivots, radii and such have none)."""
    nid, key, i, _, _ = knob
    call = prog.nodes[nid].call
    v = call.args[key]
    if isinstance(v, tuple) and v and isinstance(v[0], tuple):
        return float(v[i][0]) if call.name in DIAGONAL_OPS else POINT_NEUTRAL.get(call.name)
    p = OPS[call.name].params[key].default
    return (p[i] if isinstance(p, tuple) else p) if call.name in IDENTITY_DEFAULTS else None


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
        down = cap
        if _rolloff_on() and call.name == "curve" and call.args.get("space", "log") == "log" and isinstance(v, tuple) \
                and neutral >= ROLLOFF_FROM:
            # a highlight roll-off: above ROLLOFF_FROM a curve point may sit further below the diagonal, more the
            # brighter it is (never above the usual cap: a roll-off only brings highlights down)
            down = cap + ROLLOFF_SLOPE * (neutral - ROLLOFF_FROM)
        lo, hi = max(lo, neutral - down), min(hi, neutral + cap)
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


def bound(prog, start=None):
    """Return (a copy of prog with every tunable value inside the tuner's bounds, what moved). The loop applies it
    to a model's program before measuring it: in testing a 7B's wheels(gamma: [0.1, 0, -0.1]) met every check and
    turned a blue wall red. start: the program the loop began from (a look). A value the look itself holds beyond
    the bounds (a vivid look's saturation 1.8) may stay anywhere between the bound and the look's value, not further
    out: the loop can soften a look but not push it past where it began."""
    prog = copy.deepcopy(prog)
    knobs = _knobs(prog)
    before = {k: _get(prog, k) for k in knobs}
    _repair(prog, knobs)
    own = {k: _get(start, k) for k in knobs if start is not None and _same_knob(prog, start, k)}
    moved = []
    for k in knobs:
        r, b = before[k], _get(prog, k)
        if k in own:
            s = own[k]
            _put(prog, k, round(float(min(max(r, min(b, s)), max(b, s))), 4))
    for nid in {k[0] for k in own}:  # a restored value can sit above its neighbour: keep diagonal curves monotone
        call = prog.nodes[nid].call
        if call.name in DIAGONAL_OPS:
            run, fixed = -1e9, []
            for row in call.args["points"]:
                run = max(run, float(row[1]))
                fixed.append((row[0], run, *row[2:]))
            call.args["points"] = tuple(fixed)
    for k in knobs:
        if abs(_get(prog, k) - before[k]) > 1e-4:
            moved.append(f"{k[0]}.{k[1]}{'' if k[2] is None else f'[{k[2]}]'}: {before[k]:g} -> {_get(prog, k):g}")
    return prog, moved


def _same_knob(prog, other, knob):
    """Does `other` have this value at the same place (node id, op, parameter, index; points at the same input)?"""
    nid, key, i, _, _ = knob
    a, b = prog.nodes[nid].call, getattr(other.nodes.get(nid), "call", None)
    if b is None or a.name != b.name or key not in b.args or isinstance(b.args[key], A.Keys):
        return False
    if i is None:
        return True
    va, vb = a.args[key], b.args[key]
    if i >= len(vb):
        return False
    if not isinstance(va[0], tuple):
        return not isinstance(vb[i], A.Keys)  # one keyframed component of the look: not a plain number to keep
    return isinstance(vb[i], tuple) and float(va[i][0]) == float(vb[i][0])


def _set(prog, knob, x):
    lo, hi = _limits(prog, knob)
    _put(prog, knob, round(float(min(max(x, lo), hi)), 4))  # readable values; 1e-4 is below every tolerance


def _put(prog, knob, x):
    nid, key, i, _, _ = knob
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
         explain=True, damage_guard=None):
    """Return (tuned Program, info dict). image: the full frame in `input` (a preview is made here).
    Search: single-value moves; when none improves at a step size, moves of two values on different nodes together
    (the pairs that "warmer, highlights held" needs); then the step halves. explain: for each note still unmet, tune
    for that note alone; info["unmet"] says whether it conflicts with the other notes on this frame (met alone) or is
    out of reach within the bounds (not even alone).
    damage_guard (default on; CIERRA_TUNER_GUARD=0 turns it off): a step that lowers the cost is taken only if it adds
    no damage flag (cierra.qc: clipping, crushed / milky blacks, posterisation, banding) the starting program did not
    already have, measured on a 480 px preview as the scorer does; the search keeps the best undamaged step. A benchmark
    found a third of the graded cases damaged by the tuner's pushes (2026-11-10). Then flags the start already had (a node
    the model wrote) are eased: _ease_damage. damage_guard False (or CIERRA_TUNER_GUARD=0) restores the tuner of
    2026-11-09 exactly."""
    import os
    if damage_guard is None:
        damage_guard = os.environ.get("CIERRA_TUNER_GUARD", "1") != "0"
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

    # the damage guard measures where the scorer and the loop do: a 480 px preview
    from .qc import damage as _damage
    gimg = preview(np.asarray(image, dtype=np.float64), 480)

    def g_render(p):
        return render(p, gimg, input=input, px_scale=gimg.shape[-1] / p.reference_width)

    g_base = g_render(ungraded(prog))

    def kinds(p):
        return {f.split(":")[0] for f in _damage(g_base, g_render(p))["flags"]}

    allowed = kinds(prog) if damage_guard else set()
    info["guard"] = {"on": bool(damage_guard), "allowed": sorted(allowed), "refused_steps": 0}

    def safe(p):
        if not damage_guard:
            return True
        ok = kinds(p) <= allowed
        info["guard"]["refused_steps"] += not ok
        return ok

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
                if c < best_cost - 1e-9 and safe(trial):
                    best, best_cost, best_d, improved = trial, c, d, True
                    break
            if info["evals"] >= max_evals:
                break
        if not improved:
            improved = _pair_step(best, knobs, step, cost, x0, best_cost, max_evals, info, safe)
            if improved:
                best, best_cost, best_d = improved
        if not improved:
            step /= 2
    if damage_guard:
        best = _ease_damage(best, knobs, g_render, g_base, probes, info)
    tuned = load(format_program(best))  # re-checked: every value is in range by construction
    from .qc import damage
    qc = damage(render_fn(ungraded(tuned)), render_fn(tuned))  # meeting the notes is not enough
    dmg = qc["flags"]
    info.update({"damage": dmg, "needs_human": bool(dmg), "watch": qc["watch"]})
    info.update({"deficits": {k: round(v, 3) for k, v in best_d.items()}, "cost": round(best_cost, 4),
                 "final": format_program(tuned)})
    if explain:
        info["unmet"] = _explain_unmet(tuned, probes, best_d, image, input, width, penalty)
    if log:
        log(f"tuner: {info['evals']} renders, added {added or 'nothing'}, deficits {info['deficits']}"
            + (f", unmet {info['unmet']}" if info.get("unmet") else ""))
    return tuned, info


EASE = (0.25, 0.5, 0.75, 1.0)  # share of the way back to neutral, least first


def _ease_damage(prog, knobs, g_render, g_base, probes, info):
    """For each damage flag the program still carries (the guard allows the flags its start had: usually a node the
    model wrote), find the node causing it (each node switched off in turn: the one whose removal clears the flag),
    then ease that node's values toward neutral, least first (EASE). An easing is kept only when the flag clears, no
    check that was met is lost (the scorer's own verdicts, cierra.notes.report, on the 480 px preview), and no other
    flag appears. When no single node clears the flag, pairs of nodes are tried the same way (eased together).
    Returns the program; info["eased"] lists what changed and info["ease_kept"] what could not be eased."""
    from .notes import failing, measure_notes, regions, report
    from .qc import damage
    regs = regions(g_base)
    b = measure_notes(probes, g_base, regs)

    def state(p):
        out = g_render(p)
        flags = {f.split(":")[0] for f in damage(g_base, out)["flags"]}
        rows, _ = report(probes, b, b, measure_notes(probes, out, regs))
        bad = {r["note"] for r in failing(rows)}
        return flags, {r["note"] for r in rows} - bad

    flags, met = state(prog)
    info["eased"], info["ease_kept"] = [], []
    import itertools
    nodes = sorted({k[0] for k in knobs})
    for flag in sorted(flags):
        cause = None
        # one node, then pairs of nodes (two moves that together cause it: a flattening contrast plus a lift)
        for group in [(n,) for n in nodes] + list(itertools.combinations(nodes, 2)):
            trial = copy.deepcopy(prog)
            for nid in group:
                trial.nodes[nid].enabled = False
            if flag not in state(trial)[0]:
                cause = group
                break
        if cause is None:
            info["ease_kept"].append(f"{flag}: no node or pair of nodes causes it")
            continue
        mine = [k for k in knobs if k[0] in cause and _neutral(prog, k) is not None]
        done = False
        for f in EASE:
            trial = copy.deepcopy(prog)
            for k in mine:
                n = _neutral(prog, k)
                _set(trial, k, n + (1 - f) * (_get(prog, k) - n))
            fl, mt = state(trial)
            if flag not in fl and fl <= flags and met <= mt:
                prog, flags, met, done = trial, fl, mt, True
                info["eased"].append(f"{flag}: node(s) {', '.join(cause)} eased {int(f * 100)} % toward neutral")
                break
        if not done:
            info["ease_kept"].append(f"{flag}: easing node(s) {', '.join(cause)} would lose a check or add a flag")
    return prog


def _pair_step(best, knobs, step, cost, x0, best_cost, max_evals, info, safe=lambda p: True):
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
                    if c < best_cost - 1e-9 and safe(trial):
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
