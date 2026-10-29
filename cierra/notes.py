"""Note checks: turn the colourist's notes into measured probes, so each round can be told which notes moved the
right way and which moved the wrong way, against the ungraded frame and against the previous version.

Probes measure fixed regions chosen once on the UNGRADED frame (skin, foliage), so a grade cannot "pass" a note by
moving other pixels into the region (warming the whole frame must not count as warmer faces because the sky became
skin-hued). Deterministic keyword rules; a note no rule recognises is listed as "not measured", never guessed.
All values are display code values (sRGB), Y'CbCr as in cierra.analysis.
"""

import re

import numpy as np

from .analysis import SKIN_LINE_DEG


def _opp(d):
    y = 0.2126 * d[0] + 0.7152 * d[1] + 0.0722 * d[2]
    cb, cr = (d[2] - y) / 1.8556, (d[0] - y) / 1.5748
    return y, cb, cr, np.hypot(cb, cr), np.degrees(np.arctan2(cr, cb)) % 360


def regions(d):
    """Pixel sets from the ungraded display frame (boolean (H, W) masks)."""
    y, _, _, c, hue = _opp(d)
    dev = np.abs(((hue - SKIN_LINE_DEG + 180) % 360) - 180)
    skin = (dev < 20) & (c > 0.02) & (c < 0.25) & (y > 0.15) & (y < 0.9)
    foliage = (np.abs(((hue - 232 + 180) % 360) - 180) < 35) & (c > 0.015) & (y > 0.05)
    return {"skin": skin, "rest": ~skin, "foliage": foliage, "all": np.ones_like(skin)}


def _mean(v, m):
    return float(v[m].mean()) if m.any() else float("nan")


def warmth(d, m):
    """Red-minus-blue opponent axis (Cr - Cb) averaged over a region: higher = warmer."""
    _, cb, cr, _, _ = _opp(d)
    return _mean(cr - cb, m)


def chroma(d, m):
    return _mean(_opp(d)[3], m)


def luma_q(q):
    return lambda d, m: float(np.quantile(_opp(d)[0][m], q)) if m.any() else float("nan")


def contrast(d, m):
    """Mid-tone contrast: the spread of the middle half of luma (robust to large black or white areas)."""
    y = _opp(d)[0][m]
    return float(np.quantile(y, 0.75) - np.quantile(y, 0.25)) if y.size else float("nan")


def crushed(d, m):
    return float(100 * (d.max(0)[m] <= 0.005).mean()) if m.any() else float("nan")


def at_white(d, m):
    """Share of pixels at display white (as analysis clipped_white_pct): a p99.5 can hold while more of the frame
    clips."""
    return float(100 * (d.max(0)[m] >= 0.995).mean()) if m.any() else float("nan")


# (pattern, key, label, region, metric, unit, want, tolerance)
#   want +1 = should go up, -1 = should go down, 0 = should not get worse (hold; "worse" = up for these metrics)
PROBES = [
    (r"\b(warm|warmer|healthier)\s+(faces?|skin)", "skin_warmth", "warmer faces: warmth on the faces", "skin",
     warmth, "Cr-Cb", +1, 0.002),
    (r"\b(warm|warmer|healthier)\s+(faces?|skin)", "rest_warmth", "warmer faces only: warmth off the faces", "rest",
     warmth, "Cr-Cb", 0, 0.004),
    (r"\bwarm(er)?\b(?!\s+(faces?|skin))", "warmth", "warmer overall", "all", warmth, "Cr-Cb", +1, 0.002),
    (r"\b(cool|cooler|colder)\b", "coolness", "cooler overall", "all", warmth, "Cr-Cb", -1, 0.002),
    (r"\bless green\b", "foliage_chroma", "less green: colour of the foliage", "foliage", chroma, "chroma", -1, 0.001),
    (r"\bmore contrast\b|\bpunch", "contrast_up", "more contrast: mid-tone contrast", "all", contrast,
     "luma p75-p25", +1, 0.005),
    (r"\bless contrast\b|\bsofter\b|\bflatter\b", "contrast_down", "less contrast: mid-tone contrast", "all",
     contrast, "luma p75-p25", -1, 0.005),
    (r"\b(clean|keep|hold)\b[^.,;]*\bblacks?\b|\bblacks?\b[^.,;]*\bclean\b|\bno crushed\b", "blacks",
     "keep the blacks clean: crushed blacks", "all", crushed, "% pixels", 0, 0.3),
    (r"\bhighlights?\b[^.,;]*\b(soft|softer|gentle|roll)", "highlights",
     "highlights soft: brightest highlights", "all", luma_q(0.995), "luma p99.5", 0, 0.005),
    (r"\bhighlights?\b[^.,;]*\b(soft|softer|gentle|roll)", "highlights_white",
     "highlights soft (white): pixels at white", "all", at_white, "% pixels", 0, 0.3),
    (r"\b(bright|brighter|lift|lighter)\b", "brighter", "brighter: mid-tones", "all", luma_q(0.5), "luma p50",
     +1, 0.005),
    (r"\b(dark|darker|moodier)\b", "darker", "darker: mid-tones", "all", luma_q(0.5), "luma p50", -1, 0.005),
    (r"\bmore\s+(saturation|colou?r)\b|\bricher\b(?!\s+(faces?|skin))|\bvivid\b", "richer", "more colour",
     "all", chroma, "chroma", +1, 0.001),
    (r"\bless\s+(saturation|colou?r)\b|\bmuted\b|\bdesaturated?\b", "muted", "less colour", "all", chroma, "chroma",
     -1, 0.001),
]


# One plain phrase per probe: the vocabulary a free-form note is translated into (translate_notes).
PHRASES = {"skin_warmth": "warmer faces", "warmth": "warmer", "coolness": "cooler", "foliage_chroma": "less green",
           "contrast_up": "more contrast", "contrast_down": "less contrast", "blacks": "keep the blacks clean",
           "highlights": "keep the highlights soft", "brighter": "brighter", "darker": "darker",
           "richer": "more colour", "muted": "less colour"}
TRANSLATE_MARK = "Translate the colourist's notes"
TRANSLATE_SYSTEM = "You turn a film colourist's notes into a short list of plain grading phrases. Reply with one line."

# "axes" translation: one direction per axis (the model must decide every axis, "same" included), then the phrases.
# (axis, what it means, {answer: probe key}). The first four take "slightly <answer>" for a direction the notes only
# hint at: it becomes "slightly <phrase>", which the tuner weights lower (cierra.tuner.note_weights).
GRADED_AXES = ("warmth", "brightness", "colour", "contrast")
AXES = [("warmth", "colour temperature of the whole picture", {"warmer": "warmth", "cooler": "coolness"}),
        ("brightness", "mid-tone level", {"brighter": "brighter", "darker": "darker"}),
        ("colour", "saturation of the whole picture", {"more": "richer", "less": "muted"}),
        ("contrast", "mid-tone contrast", {"more": "contrast_up", "less": "contrast_down"}),
        ("blacks", "whether the notes ask to protect shadow detail", {"keep clean": "blacks"}),
        ("highlights", "whether the notes ask to protect bright detail", {"keep soft": "highlights"}),
        ("faces", "skin tones only", {"warmer": "skin_warmth"}),
        ("greens", "foliage only", {"less": "foliage_chroma"})]
# Worked examples for the axes prompt. Written for this prompt and kept apart from the held-out sets: no held-out note,
# frame or wording is used here (tests/test_heldout_freeform.py checks the wording).
AXES_EXAMPLES = [
    ("Underwater: quiet, distant, the world far away.",
     dict(warmth="cooler", brightness="slightly darker", colour="less", contrast="less")),
    ("Candle-lit dinner, intimate, but I still want to see into the shadows.",
     dict(warmth="warmer", brightness="darker", blacks="keep clean")),
    ("Comic-book energy, everything popping off the screen.", dict(colour="more", contrast="more",
                                                                  brightness="slightly brighter")),
]


def _axes_prompt(notes):
    lines = [f"{TRANSLATE_MARK} into one direction per axis.", "Axes and the answers allowed:"]
    for axis, doc, opts in AXES:
        opts = list(opts) + ([f"slightly {o}" for o in opts] if axis in GRADED_AXES else [])
        lines.append(f"- {axis} ({doc}): {' / '.join(opts)} / same")
    lines += ["Answer every axis. Use 'same' unless the notes imply a direction; think about what the "
              "image would look like (a mood or a place implies light, colour and contrast). Use 'slightly' when "
              "the notes only hint at a direction, the plain answer when they clearly ask for it.",
              "Reply with exactly one line per axis, 'axis: answer', nothing else.", ""]
    for ex_notes, ans in AXES_EXAMPLES:
        lines.append(f"Example notes: {ex_notes}")
        lines += [f"{axis}: {ans.get(axis, 'same')}" for axis, _, _ in AXES]
        lines.append("")
    lines.append(f"Notes from the colourist: {notes}")
    return "\n".join(lines)


def _parse_axes(raw):
    """[(probe key, slight)] from 'axis: answer' lines."""
    keys = []
    for axis, _, opts in AXES:
        m = re.search(rf"^\W*{axis}\W*:\s*(.+)$", raw, re.M | re.I)
        if not m:
            continue
        val = m.group(1).strip().lower()
        slight = val.startswith("slight") and axis in GRADED_AXES
        val = re.sub(r"^slight(ly)?\s+", "", val)
        keys += [(key, slight) for ans, key in opts.items() if val.startswith(ans)]
    return keys


def translate_notes(notes, provider, style="axes"):
    """Ask the provider to restate free-form notes ("like a 70s snapshot") in the note-check phrases. Returns
    (phrases joined by ', ', raw reply). Only phrases the checks recognise are kept; nothing is invented here.
    style "axes": a direction per axis with worked examples; "phrases": pick from the phrase list (first version).
    A reply without axis lines (the rules driver echoes the notes) is read as phrases."""
    from .llm import _ask
    if style == "axes":
        user = _axes_prompt(notes)
    else:
        user = (f"{TRANSLATE_MARK} into these phrases only: {', '.join(PHRASES.values())}.\n"
                "Use each phrase at most once, only the ones the notes imply, comma-separated, nothing else.\n"
                f"Notes from the colourist: {notes}")
    raw = _ask(provider, TRANSLATE_SYSTEM if style != "axes" else "You read a film colourist's notes.", user)
    keys = _parse_axes(raw) if style == "axes" else []
    if not keys:
        keys = [(row[1], False) for row in probes_for(raw)]
    return ", ".join(("slightly " if slight else "") + PHRASES[k] for k, slight in keys if k in PHRASES), raw


OPPOSITE = {"warmth": "coolness", "coolness": "warmth", "brighter": "darker", "darker": "brighter",
            "richer": "muted", "muted": "richer", "contrast_up": "contrast_down", "contrast_down": "contrast_up"}


def compare_intents(got, want):
    """Score a translation against the intended checks (both note text). Returns hit / missed / reversed / extra
    probe keys: reversed = the opposite direction was named."""
    g = [row[1] for row in probes_for(got or "")]
    w = [row[1] for row in probes_for(want or "")]
    return {"hit": [k for k in w if k in g], "missed": [k for k in w if k not in g and OPPOSITE.get(k) not in g],
            "reversed": [k for k in w if OPPOSITE.get(k) in g], "extra": [k for k in g if k not in w
                                                                         and OPPOSITE.get(k) not in w]}


def probes_for(notes):
    """The probes the notes ask for, in table order, each once."""
    n = notes.lower()
    out, seen = [], set()
    for row in PROBES:
        if row[1] not in seen and re.search(row[0], n):
            seen.add(row[1])
            out.append(row)
    return out


def unmeasured(notes):
    """Comma / 'and' separated parts of the notes no probe recognises (listed, never guessed)."""
    parts = [p.strip() for p in re.split(r"[,.;]|\band\b", notes) if p.strip()]
    return [p for p in parts if not any(re.search(row[0], p.lower()) for row in PROBES)]


def measure_notes(probes, d, regs):
    return {row[1]: row[4](d, regs[row[3]]) for row in probes}


def verdict(want, tol, before, after):
    if not (np.isfinite(before) and np.isfinite(after)):
        return "not measurable"
    delta = after - before
    if want == 0:
        return "WRONG WAY" if delta > tol else "ok"
    if abs(delta) <= tol:
        return "no change"
    return "right way" if np.sign(delta) == want else "WRONG WAY"


def report(probes, base, prev, now):
    """Rows {key, label, unit, ungraded, previous, now, vs_ungraded, vs_previous} and plain-English lines."""
    rows, lines = [], []
    for row in probes:
        key, label, unit, want, tol = row[1], row[2], row[5], row[6], row[7]
        r = {"note": label, "unit": unit, "ungraded": round(base[key], 4), "previous": round(prev[key], 4),
             "now": round(now[key], 4), "vs_ungraded": verdict(want, tol, base[key], now[key]),
             "vs_previous": verdict(want, tol, prev[key], now[key])}
        rows.append(r)
        goal = {1: "should go up", -1: "should go down", 0: "should not rise"}[want]
        lines.append(f"{label} ({unit}, {goal}): ungraded {r['ungraded']}, previous {r['previous']}, now {r['now']} "
                     f"-> {r['vs_ungraded']} vs ungraded, {r['vs_previous']} vs previous")
    return rows, lines


def failing(rows):
    """Rows whose note is not met against the ungraded frame (WRONG WAY or no change)."""
    return [r for r in rows if r["vs_ungraded"] in ("WRONG WAY", "no change")]


def attribute(prog, probes, render_fn, regs, now_vals):
    """For each probe that went the WRONG WAY, the node whose removal helps it most (each enabled node switched off
    in turn, one re-render each): {probe key: node id}. A note that did not move has no cause; it needs a new node."""
    import copy
    from .walk import graph_nodes
    out, gains = {}, {}
    for node in graph_nodes(prog, prog.graph):
        if not node.enabled:
            continue
        p2 = copy.deepcopy(prog)
        p2.nodes[node.id].enabled = False
        vals = measure_notes(probes, render_fn(p2), regs)
        for row in probes:
            key, want, tol = row[1], row[6], row[7]
            gain = (vals[key] - now_vals[key]) * want if want else now_vals[key] - vals[key]
            if gain > tol and gain > gains.get(key, 0.0):
                gains[key], out[key] = gain, node.id
    return out


def revision_text(probes, rows, causes):
    """The opening of a revision turn: the notes not met, what causes them, what the next version must change."""
    by_label = {row[2]: row for row in probes}
    lines, must_change, must_add = [], [], []
    for r in failing(rows):
        row = by_label[r["note"]]
        if r["vs_ungraded"] == "WRONG WAY":
            cause = causes.get(row[1])
            if cause:
                lines.append(f"- {r['note']}: WRONG WAY ({r['ungraded']} -> {r['now']}, {row[5]}). Caused by node "
                             f"{cause}: change or remove it.")
                must_change.append(cause)
            else:
                lines.append(f"- {r['note']}: WRONG WAY ({r['ungraded']} -> {r['now']}, {row[5]}). Change the nodes "
                             "that move it.")
        else:
            lines.append(f"- {r['note']}: no change ({r['now']}, {row[5]}). No node does this yet: add one.")
            must_add.append(r["note"].split(":")[0])
    if not lines:
        return "", [], []
    must_change = sorted(set(must_change))
    req = []
    if must_change:
        req.append("change node(s) " + ", ".join(must_change))
    if must_add:
        req.append("add node(s) for: " + "; ".join(must_add))
    text = "\n".join(["REVISE FIRST. These notes are not met yet; keep what already works:", *lines,
                      "The next version must " + " and ".join(req) + ". Sending the same program again is not allowed."])
    return text, must_change, must_add
