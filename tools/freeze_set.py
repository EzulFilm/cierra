"""Validate and freeze a blind note set WITHOUT showing its cases (the person or chat freezing it may later play the
model). Prints counts only.

    python tools/freeze_set.py <raw answer file> tests/data/heldout_freeform_x.json [--max-hold 1]

Checks: valid JSON with "cases"; unique ids; frames from the test frames; every check phrase recognised by
cierra.notes (each phrase maps to exactly one probe); no contradicting pair; no check phrase written inside its own
note (the note must imply, not say). Writes the set and prints its SHA-256; record that hash before anything is run on it.
"""
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cierra.notes import PHRASES, probes_for  # noqa: E402

FRAMES = {p.stem for p in (ROOT / "tests" / "data" / "frames").glob("*.npy")}
OPPOSITE = [("warmth", "coolness"), ("brighter", "darker"), ("contrast_up", "contrast_down"), ("richer", "muted")]


HOLD = {"blacks", "highlights"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"), ap.add_argument("out")
    ap.add_argument("--max-hold", type=int, default=None, help="at most this many hold checks per case, and no case of "
                    "hold checks only")
    a = ap.parse_args()
    raw, out = Path(a.raw).read_text(), Path(a.out)
    m = re.search(r"\{.*\}", raw, re.S)
    data = json.loads(m.group(0))
    cases, problems = data["cases"], []
    ids = Counter(c["id"] for c in cases)
    problems += [f"duplicate id ({n}x)" for i, n in ids.items() if n > 1]
    keys, frames, nchecks = Counter(), Counter(), []
    for k, c in enumerate(cases):
        if c["frame"] not in FRAMES:
            problems.append(f"case {k}: unknown frame")
        frames[c["frame"]] += 1
        phrases = [p.strip() for p in c["checks"].split(",") if p.strip()]
        if not all(p in PHRASES.values() for p in phrases):
            problems.append(f"case {k}: a check phrase outside the list")
        found = [row[1] for row in probes_for(c["checks"])]
        found = [f for f in found if f != "highlights_white" and f != "rest_warmth"]  # paired probes of one phrase
        if len(found) != len(phrases):
            problems.append(f"case {k}: {len(phrases)} phrases but {len(found)} probes")
        if any(a in found and b in found for a, b in OPPOSITE):
            problems.append(f"case {k}: contradicting checks")
        if any(re.search(r"\b" + re.escape(p) + r"\b", c["notes"].lower()) for p in phrases):
            problems.append(f"case {k}: a check phrase written in the note")
        if a.max_hold is not None:
            holds = [f for f in found if f in HOLD]
            if len(holds) > a.max_hold:
                problems.append(f"case {k}: {len(holds)} hold checks (max {a.max_hold})")
            if found and len(holds) == len(found):
                problems.append(f"case {k}: hold checks only")
        keys.update(found)
        nchecks.append(len(found))
    print(f"cases {len(cases)}, checks {sum(nchecks)} (per case {min(nchecks)}-{max(nchecks)}), frames {dict(frames)}")
    print(f"check kinds {dict(keys)}")
    print(f"problems {len(problems)}: {problems[:10]}")
    if problems:
        sys.exit(1)
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    out.write_text(text)
    print(f"frozen {out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
