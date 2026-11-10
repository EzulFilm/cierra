"""The checked route (cierra/assist.py): each gate on a small case, with the simulated host (cierra/hosts/sim.py:
the package's DCTL through the CPU harness, the output transform, a 16-bit still; no grading application)."""

import json
import shutil

import numpy as np
import pytest

from cierra import load
from cierra.assist import host_check, prepare, verify_record
from conftest import small

needs_gcc = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc (the DCTL CPU harness)")

WARM = ("cierra 0.1\ninput linear_rec709\noutput aces2(display: srgb)\n"
        "node w = white_balance(temperature: 0.2)\nnode s = saturation(amount: 1.15)\ngraph w -> s\n")


def test_checker_rejects_before_anything_else(tmp_path, chart):
    rec = prepare(WARM.replace("0.2", "9"), small(chart), "warmer", tmp_path)
    assert rec["status"] == "rejected" and not rec["gates"]["G1_checker"]["ok"]
    assert not (tmp_path / "apply_in_host.py").exists()


def test_a_program_the_dctl_cannot_hold_is_refused_with_the_reason(tmp_path, chart):
    src = WARM.replace("graph w -> s", "node g = glow(threshold: 0.5, radius_px: 80, strength: 0.4)\ngraph w -> s -> g")
    rec = prepare(src, small(chart), "warmer", tmp_path)
    assert rec["status"] == "refused" and "glow" in rec["reason"]


def test_only_acescct_projects(tmp_path, chart):
    with pytest.raises(ValueError):
        prepare(WARM, small(chart), "warmer", tmp_path, project="rec1886_photo")


@needs_gcc
def test_route_honest_host_verified_and_faults_caught(tmp_path, chart):
    from cierra.hosts.sim import host
    img = small(chart)
    rec = prepare(WARM, img, "warmer, more colour", tmp_path, stem="warm")
    assert rec["status"] == "ready", rec
    assert rec["gates"]["G3_notes"]["met"] == 2 and rec["gates"]["G5_preflight"]["ok"]
    prog = load((tmp_path / "warm.cierra").read_text())
    assert host_check(tmp_path, *host(tmp_path, img, prog))["host_status"] == "verified"
    for fault, want in (("not_applied", "contradiction"), ("wrong_output", "contradiction"),
                        ("no_readback", "unverified"), ("half_res", "unverified")):
        assert host_check(tmp_path, *host(tmp_path, img, prog, fault))["host_status"] == want, fault
    ok, bad = verify_record(tmp_path)
    assert ok, bad
    (tmp_path / "warm.dctl").write_text((tmp_path / "warm.dctl").read_text() + " ")
    ok, bad = verify_record(tmp_path)
    assert not ok and any("warm.dctl" in b for b in bad)


@needs_gcc
def test_damage_flag_is_needs_human_but_still_packaged(tmp_path, chart):
    src = WARM.replace("graph w -> s", "node crush = contrast(amount: 2.5, pivot: 0.5)\ngraph w -> s -> crush")
    rec = prepare(src, small(chart), "more contrast", tmp_path, stem="crush")
    assert rec["status"] == "needs_human" and rec["gates"]["G2_damage"]["flags"]
    assert json.loads((tmp_path / "record.json").read_text())["files"]["crush.dctl"]


@needs_gcc
def test_two_pass_pair_goes_to_two_nodes(tmp_path, chart):
    from cierra.hosts.sim import host
    src = WARM.replace("graph w -> s", "node hs = highlights_shadows(highlights: -1.2, radius_px: 40)\n"
                                       "graph w -> s -> hs")
    img = small(chart)
    rec = prepare(src, img, "warmer", tmp_path, stem="pair")
    assert rec["gates"]["G4_compile"]["nodes"] == 2 and rec["gates"]["G5_preflight"]["ok"], rec["gates"]
    assert rec["dctl_files"] == ["pair_node1.dctl", "pair_node2.dctl"] and "host_note" in rec
    prog = load((tmp_path / "pair.cierra").read_text())
    assert host_check(tmp_path, *host(tmp_path, img, prog))["host_status"] == "verified"
