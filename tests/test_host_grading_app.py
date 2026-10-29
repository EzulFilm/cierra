"""The grading-application adapter (cierra/hosts/grading_app.py), offline: frame identity, the fixed apply script run
against a fake host, verification statuses and the offline bundle. The live check in the application is manual and
is a manual step in a test project."""

import json

import numpy as np
import pytest

from cierra import load
from cierra.checker import load_file
from cierra.hosts.grading_app import (HostError, apply_script, export_bundle, frame_identity, frames_to_tc, identity_from_read,
                                  read_script, tc_to_frames,
                                  verify, version_name)
from conftest import LOOKS


def test_timecode_and_frame_identity():
    assert tc_to_frames("01:00:00:00", 24) == 86400 and frames_to_tc(86400 + 23, 24) == "01:00:00:23"
    with pytest.raises(HostError, match="drop-frame"):
        tc_to_frames("01:00:00;02", 30)
    ident = frame_identity(record_frame=86410, item_start=86400, source_start_frame=1000, fps=24, item_end=86500)
    assert ident == {"record_frame": 86410, "record_tc": "01:00:00:10", "clip_frame": 10, "cierra_frame": 10,
                     "source_frame": 1010, "source_tc": "00:00:42:02"}
    # a 59.94 fps clip on a 24 fps timeline steps 2.4975 source frames per record frame (measured in the app)
    steps = [frame_identity(86400 + k, 86400, 5994, 24, source_fps=59.94)["source_frame"] for k in range(3)]
    assert steps == [5994, 5996, 5998]
    assert frame_identity(86400, 86400, 5994, 24, source_fps=59.94)["source_tc"] is None
    with pytest.raises(HostError, match="outside"):
        frame_identity(86600, 86400, 1000, 24, item_end=86500)


class _Graph:
    def __init__(self, host):
        self.host = host

    def GetNumNodes(self):
        return 2

    def SetLUT(self, node, path):
        self.host.lut[node] = path
        return True

    def GetLUT(self, node):
        return self.host.readback if self.host.readback is not None else self.host.lut.get(node, "")


class _Item:
    def __init__(self, host):
        self.host, self.versions, self.current = host, ["Version 1"], "Version 1"

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
        if not self.host.delete_ok:
            return False
        self.versions.remove(name)
        return True

    def GetNodeGraph(self):
        return _Graph(self.host)


class _Host:
    def __init__(self, project="cierra-test", readback=None, still_ok=True, playhead_ok=True, delete_ok=True):
        self.name, self.readback, self.lut, self.stills = project, readback, {}, []
        self.still_ok, self.playhead_ok, self.delete_ok = still_ok, playhead_ok, delete_ok
        self.item = _Item(self)

    # resolve -> project manager -> project -> timeline: all the same fake object
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

    def SetCurrentTimecode(self, tc):
        self.tc = tc
        return True

    def GetCurrentTimecode(self):
        return self.tc if self.playhead_ok else "01:00:00:00"

    def ExportCurrentFrameAsStill(self, path):
        self.stills.append((self.tc, path))
        return self.still_ok


def _run(script, host):
    ns = {"resolve": host}
    exec(compile(script, "apply", "exec"), ns)  # noqa: S102  the fixed template, against a fake host
    return ns["result"]


PROG = load("cierra 0.1\nnode a = exposure(stops: 0.3)\ngraph a\n")


def test_apply_script_success_readback_and_stills():
    v = version_name(PROG, parent="abc123def456")
    script = apply_script("cierra-test", v, "Cierra/cierra_x.dctl", [("01:00:00:10", "C:\\grades\\s1.tif")])
    host = _Host()
    res = _run(script, host)
    assert res["ok"] and res["readback"] == "Cierra/cierra_x.dctl" and host.item.current == v
    assert res["previous_version"] == "Version 1" and host.stills == [("01:00:00:10", "C:\\grades\\s1.tif")]


def test_apply_script_rolls_back_on_readback_mismatch_and_refuses_another_project():
    v = version_name(PROG)
    script = apply_script("cierra-test", v, "Cierra/cierra_x.dctl", [])
    host = _Host(readback="Cierra/something_else.dctl")
    res = _run(script, host)
    assert not res["ok"] and res["rolled_back"] and host.item.current == "Version 1" and v not in host.item.versions
    other = _Host(project="other-project")
    res = _run(script, other)
    assert not res["ok"] and "expected cierra-test" in res["error"] and other.item.versions == ["Version 1"]


@pytest.mark.parametrize("bad", [dict(version="x'); import os; ('"), dict(lut="../../etc/x.dctl"),
                                 dict(lut="Cierra/x.py"), dict(stills=[("01:00:00:00", "C:\\x\\..\\y.tif")]),
                                 dict(stills=[("01:00:00;00", "C:\\x\\y.tif")]), dict(node=0)])
def test_apply_script_rejects_unsafe_data(bad):
    args = dict(project="cierra-test", version="cierra-abc", lut="Cierra/cierra_x.dctl", stills=[], node=1)
    args.update(bad)
    with pytest.raises(HostError):
        apply_script(**args)


def test_project_name_is_data_not_code():
    script = apply_script('evil"); import os #', "cierra-abc", "Cierra/x.dctl", [])
    assert json.loads(eval(script.split("json.loads(")[1].split(")\nout")[0]))["project"] == 'evil"); import os #'


def test_verify_statuses():
    ref = np.full((3, 4, 4), 0.5)
    ok = {"ok": True, "readback": "Cierra/x.dctl", "stills": [{"ok": True}]}
    assert verify(ok, [ref + 0.0005], [ref])["status"] == "verified"
    assert verify(ok, [ref + 0.1], [ref])["status"] == "contradiction"
    assert verify({"ok": True, "readback": "", "stills": []}, [ref], [ref])["status"] == "unverified"
    assert verify(ok, [], [ref])["status"] == "unverified"
    assert "resolution" in verify(ok, [np.zeros((3, 2, 2))], [ref])["reason"]


def test_offline_bundle(tmp_path):
    m = export_bundle(load_file(LOOKS / "cinematic_modern.cierra"), tmp_path, "cinematic", parent="p0")
    assert m["parent"] == "p0" and m["version"].startswith("cierra-") and "cinematic.dctl" in m["files"]
    assert "cdl" in m["refused"]  # a curve is not a CDL: refused with the reason, the rest still exported
    assert json.loads((tmp_path / "manifest.json").read_text())["program"] == m["program"]


@pytest.mark.parametrize("fault", [dict(still_ok=False), dict(playhead_ok=False)])
def test_a_failed_still_or_playhead_rolls_the_grade_back(fault):
    v = version_name(PROG)
    script = apply_script("cierra-test", v, "Cierra/cierra_x.dctl", [("01:00:00:02", r"C:\grades\s.tif")], wait=0)
    host = _Host(**fault)
    res = _run(script, host)
    assert not res["ok"] and res["rolled_back"] is True and "still at 01:00:00:02 failed" in res["error"]
    assert host.item.current == "Version 1" and host.item.versions == ["Version 1"]


def test_a_rollback_that_did_not_happen_is_reported():
    v = version_name(PROG)
    script = apply_script("cierra-test", v, "Cierra/cierra_x.dctl", [], wait=0)
    host = _Host(readback="Cierra/other.dctl", delete_ok=False)
    res = _run(script, host)
    assert not res["ok"] and res["rolled_back"] is False
    assert res["rollback_failed"] == {"restored": True, "deleted": False, "current": "Version 1",
                                      "version_left": True}


def test_verify_catches_a_single_bad_pixel_and_a_missing_still():
    ref = np.full((3, 40, 25), 0.5)
    ok = {"ok": True, "readback": "Cierra/x.dctl", "stills": [{"ok": True}]}
    one = ref.copy()
    one[:, 3, 4] = [0.9, 0.1, 0.1]  # one pixel in 1000, p99 still 0
    v = verify(ok, [one], [ref])
    assert v["status"] == "contradiction" and v["frames"][0]["p99"] == 0 and "outside clipped white" in v["reason"]
    white = ref.copy()
    white[:, 3, 4] = [0.999, 0.9, 0.6]  # a blown highlight tinted differently: counted, allowed when rare
    assert verify(ok, [white] * 1, [ref])["frames"][0]["clipped_over_1"] == 1
    two = {"ok": True, "readback": "Cierra/x.dctl", "stills": [{"ok": True}, {"ok": True}]}
    assert verify(two, [ref], [ref])["status"] == "unverified"  # 2 reported, 1 image
    assert verify(ok, [ref], [ref], requested=2)["status"] == "unverified"


class _ReadHost:
    """resolve -> project -> timeline -> item -> media pool item, as measured in a live test project."""

    def __init__(self, project="cierra-test", clip_fps="59.94", drop="0"):
        self.project, self.clip_fps, self.drop, self.calls = project, clip_fps, drop, []

    def __getattr__(self, name):  # record every call: the read script must only read
        values = {"GetProjectManager": self, "GetCurrentProject": self, "GetCurrentTimeline": self,
                  "GetCurrentVideoItem": self, "GetMediaPoolItem": self, "GetName": self.project,
                  "GetCurrentTimecode": "01:00:00:02", "GetStart": 86400, "GetEnd": 86500,
                  "GetSourceStartFrame": 5994}
        if name == "GetSetting":
            return lambda key: {"timelineFrameRate": "24", "timelineDropFrameTimecode": self.drop}[key]
        if name == "GetClipProperty":
            return lambda key: {"FPS": self.clip_fps}[key]
        if name in values:
            return lambda *a: (self.calls.append(name), values[name])[1]
        raise AttributeError(name)


def test_read_script_gives_frame_identity_with_the_clips_own_rate():
    host = _ReadHost()
    read = _run(read_script("cierra-test"), host)
    assert read["ok"] and read["clip_fps"] == "59.94" and read["timeline_fps"] == "24"
    assert all(c.startswith("Get") for c in host.calls)  # read-only
    assert identity_from_read(read, 86402)["source_frame"] == 5998  # record +2 -> source +4, as measured live
    assert identity_from_read(_run(read_script("cierra-test"), _ReadHost(clip_fps="24")), 86402)["source_frame"] == 5996


def test_read_script_refuses_another_project_and_drop_frame():
    read = _run(read_script("cierra-test"), _ReadHost(project="another project"))
    assert not read["ok"] and "expected cierra-test" in read["error"]
    with pytest.raises(HostError, match="host read failed"):
        identity_from_read(read, 86402)
    with pytest.raises(HostError, match="drop-frame"):
        identity_from_read(_run(read_script("cierra-test"), _ReadHost(drop="1")), 86402)
