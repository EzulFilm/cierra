"""docs/conformance.md stays true: every normative spec section has a row, and every test it names exists."""

import re

from conftest import ROOT


def _rows():
    rows = {}
    for line in (ROOT / "docs" / "conformance.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(\d+)\s*\|", line)
        if m:
            rows[int(m.group(1))] = re.findall(r"`(test_\w+\.py)::(test_\w+)`", line)
    return rows


def test_every_spec_section_has_a_row_with_tests():
    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    sections = {int(n) for n in re.findall(r"^## (\d+)\. ", spec, re.M)}
    rows = _rows()
    assert sections == set(rows), (sections, set(rows))
    assert all(rows[s] for s in sections)


def test_every_named_test_exists():
    for sec, refs in _rows().items():
        for fname, test in refs:
            src = (ROOT / "tests" / fname).read_text(encoding="utf-8")
            assert re.search(rf"^def {test}\(", src, re.M), (sec, fname, test)


def test_every_op_mask_and_blend_mode_is_in_the_spec():
    """A new op, mask or blend mode cannot ship without its row (ops, masks) or its formula (blend modes) in spec.md."""
    from cierra.ops import BLEND_MODES, MASKS, OPS
    spec = (ROOT / "docs" / "spec.md").read_text(encoding="utf-8")
    missing = [n for n in [*OPS, *MASKS] if not re.search(rf"^\| {n} \|", spec, re.M)]
    graph = spec[spec.index("## 7. Graph"):spec.index("## 8. ")]
    missing += [f"blend {m}" for m in BLEND_MODES if not re.search(rf"\b{m}\b", graph)]
    assert not missing, missing
