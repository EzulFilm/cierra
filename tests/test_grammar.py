"""cierra.grammar: the GBNF the open-weight driver samples program replies under, checked with the reference
recogniser (no server). Every reply it allows must be ordinary Cierra; what it leaves out is listed here on purpose."""

import pytest

from cierra.checker import check, load
from cierra.examples import VALID
from cierra.format import format_program
from cierra.grammar import loop_grammar, matches, reply_gbnf, wrap
from cierra.llm import NEUTRAL, RulesProvider
from cierra.parser import parse

G = reply_gbnf()
OUTSIDE = ("keyframes", "display-referred photo", "depth and segment",  # keys(), photo_graph, requires,
           "key routing")  # key statements and key(...)


def _canon(src):
    return format_program(check(parse(src))[0])


@pytest.mark.parametrize("title,src", VALID, ids=[t[:30] for t, _ in VALID])
def test_valid_examples_in_the_subset_match(title, src):
    assert matches(G, wrap(_canon(src))) == (not title.startswith(OUTSIDE))


def test_neutral_start_and_an_output_only_program_match():
    assert matches(G, wrap(_canon(NEUTRAL.format(input="linear_rec709"))))
    assert loop_grammar(load(NEUTRAL.format(input="acescg")), "short") is not None


@pytest.mark.parametrize("bad", [
    "Sure! Here it is:\n" + wrap("cierra 0.1\noutput aces2()\n"),            # chatter before the block
    "```cierra\ncierra 0.1\noutput aces2()\n```\n",                           # no message line
    wrap("cierra 0.1\noutput aces2()\nnode a = sharpen_more(amount: 1)\ngraph a\n"),     # unknown op
    wrap("cierra 0.1\noutput aces2()\nnode a = exposure(amount: 1)\ngraph a\n"),         # another op's parameter
    wrap("cierra 0.1\noutput aces2()\nnode a = wheels(lift: [0.01, 0])\ngraph a\n"),     # rgb with two values
    wrap("cierra 0.1\noutput aces2()\nnode a = saturation(amount: 1.1)\n  mask face()\ngraph a\n"),  # unknown mask
    wrap("cierra 0.1\noutput aces2(display: p3)\n"),                                     # not a choice
    wrap("cierra 0.1\noutput aces2()\nnode a = exposure(stops: .5)\ngraph a\n"),         # no leading-dot numbers
])
def test_wrong_shapes_are_not_in_the_grammar(bad):
    assert not matches(G, bad)


def test_rules_driver_replies_match():
    p = RulesProvider()
    reply = p.complete("", "Notes from the colourist: warmer, more contrast, less colour\nCurrent program:\n```cierra\n"
                       + _canon(NEUTRAL.format(input="linear_rec709")) + "```\n")
    src = reply.split("```cierra\n")[1].split("```")[0]
    assert matches(G, wrap(_canon(src)))


def test_loop_grammar_follows_the_profile_and_keeps_the_current_ops():
    glow = load("cierra 0.1\noutput aces2()\nnode b = glow(threshold: 0.6)\n  mask window()\ngraph b\n")
    short = loop_grammar(glow, "short")
    assert short is not None and "op-glow" in short and "mask-window" in short and "op-grain" not in short
    assert "op-grain" in loop_grammar(glow, "full")


def test_programs_the_grammar_cannot_send_back_get_no_grammar():
    keyed = load("cierra 0.1\noutput aces2()\nnode f = exposure(stops: keys(smooth, 0: -3, 24: 0))\ngraph f\n")
    assert loop_grammar(keyed, "short") is None
