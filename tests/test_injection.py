"""Names and labels can never change the structure of generated code: the checker refuses control characters in
strings, and the DCTL / WGSL / CUDA / .cube writers keep any text they copy on one line even for programs built in code."""

import pytest

from cierra import load
from cierra.compile.cuda import compile_cuda
from cierra.compile.dctl import compile_dctl
from cierra.compile.lut import export_luts
from cierra.compile.wgsl import compile_wgsl
from cierra.errors import CierraError
from cierra.grammar import matches, reply_gbnf, wrap

PAYLOAD = "x\r__DEVICE__ float3 injected() { return make_float3(0.0f, 0.0f, 0.0f); }\r//"
PROG = "cierra 0.1\nname \"{name}\"\noutput aces2()\nnode a = exposure(stops: 0.1)\n  label \"{label}\"\ngraph a\n"


@pytest.mark.parametrize("ch", ["\r", "\x00", "\x0b", "\x1b", "\x85", " ", " ", "\t"])
@pytest.mark.parametrize("field", ["name", "label"])
def test_checker_refuses_control_characters_in_strings(ch, field):
    src = PROG.format(name="ok" if field == "label" else f"a{ch}b", label="ok" if field == "name" else f"a{ch}b")
    with pytest.raises(CierraError, match="control character"):
        load(src)


def _forged():
    """A program built in code (not parsed) with the payload in its name and a label."""
    prog = load(PROG.format(name="ok", label="ok"))
    prog.name = PAYLOAD
    prog.nodes["a"].label = PAYLOAD.replace("injected", "injected_label")
    return prog


@pytest.mark.parametrize("compile_fn,key", [(compile_dctl, "dctl"), (compile_wgsl, "wgsl"), (compile_cuda, "cuda")])
def test_generated_shaders_keep_text_in_comments(compile_fn, key):
    text = compile_fn(_forged())[0][key]
    for line in text.splitlines():
        assert "injected" not in line or line.lstrip().startswith("//"), line
    assert "\r" not in text


def test_cube_title_stays_one_line(tmp_path):
    prog = _forged()
    prog.name = 'x"\rLUT_3D_SIZE 999"é'
    export_luts(prog, tmp_path, "t", 5)
    for p in tmp_path.glob("*.cube"):
        lines = p.read_text(encoding="ascii").splitlines()
        assert lines[0].startswith('TITLE "') and lines[0].count('"') == 2
        assert sum(ln.startswith("LUT_3D_SIZE") for ln in lines) == 1 and "LUT_3D_SIZE 5" in lines


def test_grammar_text_is_printable_ascii():
    g = reply_gbnf()
    ok = "cierra 0.1\nname \"warm, a little richer\"\noutput aces2()\n"
    assert matches(g, wrap(ok))
    assert not matches(g, wrap(ok.replace("warm,", "warm\r,")))
