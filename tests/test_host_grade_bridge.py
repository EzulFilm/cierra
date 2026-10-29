"""The grade both ways through the grading application's scripting API (cierra/hosts/grading_app.py), offline, with
fake hosts in the pattern of test_host_grading_app.py: the read script and import_grade (application -> Cierra), and
node_parts / export_nodes_script / verify_nodes (Cierra -> application), rollback included. The live check in the
application is a manual step in a test project."""

import json
from pathlib import Path

import numpy as np
import pytest

from cierra import load
from cierra.compile.lut import bake, write_cube
from cierra.hosts.grading_app import (HostError, export_nodes_script, import_grade, node_parts, read_grade_script,
                                      verify_nodes, version_name)

PROG = load("cierra 0.1\nname \"two nodes\"\n"
            "node balance = cdl(slope: [1.05, 1, 0.95], offset: [0.01, 0, -0.01], sat: 0.9)\n  label \"balance\"\n"
            "node skin = saturation(amount: 1.3)\n  label \"skin\"\n"
            "  mask qualifier(hue: 120..165, sat: 0.02..0.25, lum: 0.2..0.75)\n"
            "node tone = curve(points: [[0.1, 0.09], [0.4136, 0.42], [0.8, 0.82]])\n"
            "graph balance -> skin -> tone\n")


class _Graph:
    def __init__(self, host, nodes):
        self.host, self.nodes = host, nodes

    def GetNumNodes(self):
        return len(self.nodes)

    def GetNodeLabel(self, i):
        return self.nodes[i - 1]["label"]

    def GetToolsInNode(self, i):
        return self.nodes[i - 1]["tools"]

    def GetLUT(self, i):
        if self.host is not None and self.host.readback is not None:
            return self.host.readback
        return self.nodes[i - 1].get("lut", "")

    def SetLUT(self, i, path):
        self.host.calls.append(("SetLUT", i, path))
        self.nodes[i - 1]["lut"] = path
        return True

    def SetNodeEnabled(self, i, on):
        self.host.calls.append(("SetNodeEnabled", i, on))
        self.nodes[i - 1]["enabled"] = on
        return True

    def SetNodeLabel(self, i, label):
        if not self.host.labels_ok:
            raise AttributeError("SetNodeLabel")
        self.nodes[i - 1]["label"] = label
        return True


class _Group:
    def GetName(self):
        return "night"

    def GetPreClipNodeGraph(self):
        return _Graph(None, [{"label": "pre", "tools": ["Primaries"]}])

    def GetPostClipNodeGraph(self):
        return _Graph(None, [])


class _Item:
    def __init__(self, host):
        self.host, self.versions, self.current = host, ["Version 1"], "Version 1"

    def GetName(self):
        return "A001_C002"

    def GetCurrentVersion(self):
        return {"versionName": self.current, "versionType": 0}

    def GetVersionNameList(self, t):
        return list(self.versions)

    def AddVersion(self, name, t):
        self.versions.append(name)
        return True

    def LoadVersionByName(self, name, t):
        self.current = name
        return name in self.versions

    def DeleteVersionByName(self, name, t):
        self.versions.remove(name)
        return True

    def GetNodeGraph(self):
        return _Graph(self.host, self.host.nodes)

    def GetColorGroup(self):
        return _Group() if self.host.group else None

    def SetCDL(self, cdl):
        self.host.calls.append(("SetCDL", cdl))
        return self.host.cdl_ok

    def ExportLUT(self, kind, path):
        self.host.calls.append(("ExportLUT", kind, path))
        if self.host.lut is None:
            return False
        write_cube(Path(path), self.host.lut, "clip")
        return True


class _Host:
    EXPORT_LUT_17PTCUBE, EXPORT_LUT_33PTCUBE, EXPORT_LUT_65PTCUBE = 0, 1, 2

    def __init__(self, nodes, lut=None, project="cierra-test", group=False, cdl_ok=True, readback=None,
                 labels_ok=True):
        self.nodes, self.lut, self.name, self.group = nodes, lut, project, group
        self.cdl_ok, self.readback, self.labels_ok, self.calls = cdl_ok, readback, labels_ok, []
        self.item = _Item(self)

    def GetProjectManager(self):
        return self

    def GetCurrentProject(self):
        return self

    def GetName(self):
        return self.name

    def GetCurrentTimeline(self):
        return self

    def GetCurrentVideoItem(self):
        return self.item

    def RefreshLUTList(self):
        return True


def _run(script, host):
    ns = {"resolve": host}
    exec(compile(script, "host", "exec"), ns)  # noqa: S102  the fixed template, against a fake host
    return ns["result"]


def _nodes(n):
    return [{"label": f"n{i}", "tools": [], "lut": ""} for i in range(1, n + 1)]


# ---------------------------------------------------------------- application -> Cierra

def test_read_grade_and_import(tmp_path):
    src = load("cierra 0.1\nnode a = contrast(amount: 1.1, pivot: 0.4)\nnode b = saturation(amount: 0.85)\n"
               "graph a -> b\n")
    nodes = [{"label": "contrast", "tools": ["Primaries"], "lut": ""},
             {"label": "skin", "tools": ["Qualifier", "Primaries"], "lut": ""},
             {"label": "edges", "tools": ["Window", "Blur"], "lut": "Cierra/x.dctl"}]
    host = _Host(nodes, lut=bake(src, 17), group=True)
    read = _run(read_grade_script("cierra-test", str(tmp_path), lut_size=17), host)
    assert read["ok"] and read["node_count"] == 3 and read["nodes"][1]["tools"] == ["Qualifier", "Primaries"]
    assert read["nodes"][2]["lut"] == "Cierra/x.dctl" and read["clip_lut"].endswith("cierra_clip_17.cube")
    assert read["group"]["pre_clip"][0]["label"] == "pre"
    assert [c[0] for c in host.calls] == ["ExportLUT"] and host.calls[0][1] == host.EXPORT_LUT_17PTCUBE
    assert host.item.versions == ["Version 1"]  # nothing changed in the project
    prog, rep = import_grade(read)
    assert prog.name == "A001_C002" and "1 contrast [Primaries]; 2 skin [Qualifier, Primaries]" in prog.about
    assert rep["lattice"]["max"] < 0.1 and [n.call.name for n in prog.nodes.values()][0] == "contrast"
    assert any("Window" in w and "Blur" in w for w in rep["warnings"])
    assert any("colour group 'night'" in w for w in rep["warnings"])


def test_read_grade_refuses_and_reports():
    host = _Host(_nodes(1), lut=None)
    read = _run(read_grade_script("cierra-test", "C:\\grades"), host)
    assert not read["ok"] and "ExportLUT did not write C:\\grades\\cierra_clip_65.cube" in read["error"]
    with pytest.raises(HostError, match="host read failed"):
        import_grade(read)
    other = _run(read_grade_script("cierra-test", "/tmp/grades"), _Host(_nodes(1), project="another"))
    assert not other["ok"] and "expected cierra-test" in other["error"]
    for bad in (dict(out_dir="grades"), dict(out_dir="C:\\x\\..\\y"), dict(out_dir="C:\\x;rm"), dict(lut_size=32)):
        args = dict(project="cierra-test", out_dir="C:\\grades")
        args.update(bad)
        with pytest.raises(HostError):
            read_grade_script(**args)


# ---------------------------------------------------------------- Cierra -> application

def test_node_parts_cdl_dctl_and_refusal(tmp_path):
    parts = node_parts(PROG, tmp_path, "look")
    assert [(p["index"], p["node"], p["kind"]) for p in parts] == [(1, "balance", "cdl"), (2, "skin", "dctl"),
                                                                 (3, "tone", "dctl")]
    assert np.allclose(parts[0]["cdl"]["slope"], [1.05, 1, 0.95]) and parts[0]["cdl"]["sat"] == pytest.approx(0.9)
    text = (tmp_path / parts[1]["file"]).read_text()
    assert "transform" in text and "1.3" in text  # the skin node alone, its qualifier inside
    assert [p["kind"] for p in node_parts(PROG, tmp_path, "look", prefer="dctl")] == ["dctl"] * 3
    glow = load("cierra 0.1\nnode a = exposure(stops: 0.2)\nnode b = glow()\ngraph a -> b\n")
    parts = node_parts(glow, tmp_path, "g")
    assert parts[1]["kind"] == "refused" and "spatial" in parts[1]["reason"]
    with pytest.raises(HostError, match="cannot go node for node: node b"):
        export_nodes_script("cierra-test", "cierra-abc", parts)
    with pytest.raises(HostError, match="serial graph"):
        node_parts(load("cierra 0.1\nnode a = exposure()\nnode b = contrast()\ngraph parallel(a, b)\n"), tmp_path, "p")


def test_export_nodes_sets_each_node_and_verifies(tmp_path):
    parts = node_parts(PROG, tmp_path, "look")
    v = version_name(PROG)
    cube = tmp_path / "readback.cube"
    script = export_nodes_script("cierra-test", v, parts, verify_lut=str(cube), disable_rest=True)
    host = _Host(_nodes(4), lut=bake(PROG, 17, fit_iters=0))  # the application graded exactly as the program
    res = _run(script, host)
    assert res["ok"] and res["steps"] == ["version", "nodes", "rest disabled", "readback"]
    assert host.item.current == v and res["previous_version"] == "Version 1" and res["nodes_available"] == 4
    cdl = [c[1] for c in host.calls if c[0] == "SetCDL"]
    assert cdl == [{"NodeIndex": "1", "Slope": "1.050000 1.000000 0.950000", "Offset": "0.010000 0.000000 -0.010000",
                    "Power": "1.000000 1.000000 1.000000", "Saturation": "0.900000"}]
    assert [c[1:] for c in host.calls if c[0] == "SetLUT"] == [(2, "Cierra/" + parts[1]["file"]),
                                                              (3, "Cierra/" + parts[2]["file"])]
    assert host.nodes[3]["enabled"] is False and host.nodes[0]["label"] == "balance"
    assert verify_nodes(res, PROG)["status"] == "verified"
    # a clip whose grade is not the program (another correction left on a node): a contradiction, not a pass
    write_cube(cube, bake(load("cierra 0.1\nnode a = exposure(stops: 0.3)\ngraph a\n"), 17, fit_iters=0), "other")
    assert verify_nodes(res, PROG)["status"] == "contradiction"
    assert verify_nodes({"ok": True}, PROG)["status"] == "unverified"


def test_export_nodes_without_label_api_still_applies(tmp_path):
    parts = node_parts(PROG, tmp_path, "look")
    res = _run(export_nodes_script("cierra-test", version_name(PROG), parts), _Host(_nodes(3), labels_ok=False))
    assert res["ok"] and [s["labelled"] for s in res["set"]] == [False] * 3


@pytest.mark.parametrize("fault, why", [
    (dict(nodes=2), "the clip has 2 nodes and this grade needs 3 serial nodes"),
    (dict(cdl_ok=False), "SetCDL on node 1 returned False"),
    (dict(readback="Cierra/other.dctl"), "node 2 reads back Cierra/other.dctl"),
    (dict(lut=None), "ExportLUT for the readback did not work"),
])
def test_export_nodes_rolls_back(tmp_path, fault, why):
    parts = node_parts(PROG, tmp_path, "look")
    v = version_name(PROG)
    n = fault.pop("nodes", 3)
    host = _Host(_nodes(n), **fault) if "lut" in fault else _Host(_nodes(n), lut=bake(PROG, 5, fit_iters=0), **fault)
    res = _run(export_nodes_script("cierra-test", v, parts, verify_lut=str(tmp_path / "r.cube")), host)
    assert not res["ok"] and why in res["error"] and res["rolled_back"] is True
    assert host.item.current == "Version 1" and host.item.versions == ["Version 1"]


def test_export_nodes_refuses_another_project_and_unsafe_data(tmp_path):
    parts = node_parts(PROG, tmp_path, "look")
    host = _Host(_nodes(3), project="another project")
    res = _run(export_nodes_script("cierra-test", "cierra-abc", parts), host)
    assert not res["ok"] and "expected cierra-test" in res["error"] and host.calls == []
    assert host.item.versions == ["Version 1"]
    bad = [dict(version="x'); import os; ('"), dict(lut_dir="../LUT"), dict(verify_lut="relative.cube"),
           dict(verify_lut="C:\\x\\r.txt"), dict(parts=[dict(parts[1], file="../x.dctl")]),
           dict(parts=[dict(parts[0], index=2)]), dict(parts=[dict(parts[0], cdl=dict(parts[0]["cdl"], sat=np.nan))])]
    for b in bad:
        args = dict(project="cierra-test", version="cierra-abc", parts=parts)
        args.update(b)
        with pytest.raises(HostError):
            export_nodes_script(**args)
    # the label and the project are data, never code
    script = export_nodes_script('evil"); import os #', "cierra-abc", [dict(parts[0], label='x")\nimport os')])
    data = json.loads(eval(script.split("json.loads(")[1].split(")\nout")[0]))
    assert data["project"] == 'evil"); import os #' and data["parts"][0]["label"] == 'x") import os'
