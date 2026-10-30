"""Security hardening (2026-10-29): comment splices, invisible and invalid characters, oversized LUT files, hanging
subprocesses, network-share paths, history file names, model server URLs and command-line bounds. Each case failed
before its fix."""

import shutil
import subprocess

import numpy as np
import pytest

from cierra import load
from cierra.errors import CierraError


def _labelled(label):
    from cierra import ast as A
    prog = load("cierra 0.1\nnode warm = white_balance(temperature: 0.1)\ngraph warm\n")
    prog.nodes["warm"].label = label  # built in code: the checker's string rules are not in the way
    assert isinstance(prog.nodes["warm"], A.Node)
    return prog


@pytest.mark.parametrize("label", ["ends in a backslash \\", "a trigraph ??/", "many ????/ and \\\\"])
def test_l1_a_label_cannot_splice_the_next_line_into_its_comment(label):
    from cierra.compile.dctl import compile_dctl
    text = compile_dctl(_labelled(label))[0]["dctl"]
    for line in text.splitlines():
        if line.lstrip().startswith("//"):
            assert "\\" not in line and "??" not in line, line
    if shutil.which("gcc"):  # the harness compiles with -std=c99, which reads trigraphs
        from cierra.compile.dctl_cpu import run_dctl
        out = run_dctl(text, np.full((3, 2, 2), 0.4))
        assert np.isfinite(out).all()


@pytest.mark.parametrize("ch", ["\u202e", "\u2066", "\ufffe", "\uffff", "\ufdd0", "\U0001fffe"])
def test_l2_bidi_overrides_and_noncharacters_are_refused_in_strings(ch):
    with pytest.raises(CierraError):
        load(f'cierra 0.1\nname "grade {ch} name"\n')
    from cierra.compile.dctl import one_line
    assert ch not in one_line(f"label {ch} text")


def test_l3_an_oversized_cube_is_refused_before_it_is_read(tmp_path, monkeypatch):
    from cierra import fit_lut as F
    big = tmp_path / "big.cube"
    big.write_text("0 0 0\n" * 2000)
    monkeypatch.setattr(F, "MAX_CUBE_BYTES", 1000)
    with pytest.raises(F.LUTFitError, match="bytes"):
        F.read_cube(big)


def test_l3_rows_before_the_size_line_are_capped(monkeypatch):
    from cierra import fit_lut as F
    monkeypatch.setattr(F, "MAX_CUBE", 3)
    with pytest.raises(F.LUTFitError, match="more rows"):
        F._parse_cube("0 0 0\n" * 28 + "LUT_3D_SIZE 3\n")


def test_l4_a_hanging_compiler_or_harness_times_out(monkeypatch):
    from cierra.compile import dctl_cpu
    if not shutil.which("gcc"):
        pytest.skip("no gcc")
    seen = []
    real = subprocess.run

    def run(*a, **k):
        seen.append(k.get("timeout"))
        return real(*a, **k)
    monkeypatch.setattr(dctl_cpu.subprocess, "run", run)
    from cierra.compile.dctl import compile_dctl
    dctl_cpu.run_dctl(compile_dctl(_labelled("x"))[0]["dctl"], np.full((3, 1, 1), 0.4))
    assert seen and all(t == dctl_cpu.TIMEOUT for t in seen)


@pytest.mark.parametrize("path", ["//server/share/a.cube", "\\\\server\\share\\a.cube", "/\\server/a.cube"])
def test_l5_unc_paths_are_not_handed_to_the_grading_application(path):
    from cierra.hosts.grading_app import HostError, _abs_path
    with pytest.raises(HostError):
        _abs_path(path, "LUT", ".cube")
    assert _abs_path("E:/luts/a.cube", "LUT", ".cube") and _abs_path("/home/user/a.cube", "LUT", ".cube")


def test_l6_a_history_log_cannot_point_outside_its_folder(tmp_path):
    from cierra.history import History
    h = History(tmp_path / "h")
    entry = h.commit(load("cierra 0.1\nnode a = exposure(stops: 0.1)\ngraph a\n"), "first")
    assert h.load(entry).nodes
    (tmp_path / "secret.cierra").write_text("cierra 0.1\n")
    for name in ("../secret.cierra", "/etc/passwd", "v0001_abc.cierra/../../secret.cierra", None):
        with pytest.raises(ValueError):
            h.load(dict(entry, file=name))


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://192.0.2.7:8000/v1", "ftp://localhost/v1", "https:///v1"])
def test_l7_the_model_server_url_is_https_or_local_http(url):
    from cierra.llm import OpenWeightProvider
    with pytest.raises(ValueError):
        OpenWeightProvider(url=url, model="m")
    for ok in ("http://127.0.0.1:8090/v1", "http://localhost:8000/v1", "https://pod.example.com/v1"):
        assert OpenWeightProvider(url=ok, model="m").url == ok


@pytest.mark.parametrize("argv", [["lut", "p.cierra", "out", "--size", "100000"], ["lut", "p.cierra", "out", "--size", "1"],
                                  ["dctl", "p.cierra", "out", "--frame", "-1"],
                                  ["render", "p.cierra", "i.npy", "o.png", "--frame", "10000001"],
                                  ["loop", "notes", "i.npy", "h", "--iters", "100000"]])
def test_l8_cli_sizes_and_frames_are_bounded(argv, capsys):
    from cierra.__main__ import main
    with pytest.raises(SystemExit):
        main(argv)
    assert "outside" in capsys.readouterr().err
