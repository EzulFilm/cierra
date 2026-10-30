"""A narrow host adapter for the grading application's scripting API (version 21.1 names below). Cierra talks to
it only through open formats (a DCTL file in, display stills out) and a few fixed scripts; it never runs code an LLM
wrote, never edits the application's database, and keeps nothing remote.

- Frame identity (frame_identity): timeline record frame, clip-relative frame (= the Cierra keyframe index), source
  frame and source timecode, all from the item's own numbers; drop-frame timecode is refused, not guessed.
- Transactional apply (apply_script): one fixed script template, data passed as JSON. It remembers the current
  colour version, adds a namespaced version named after the program hash and its parent, loads it, sets the DCTL on
  one node, reads the LUT path back, exports a still at each verification frame, and rolls back (re-loads the old
  version, deletes the new one, then reads the versions back and reports a rollback that did not happen) on any
  failure, a failed playhead move or still included: a grade that could not be checked is not left applied. Before each still it waits (the viewer renders after the playhead
  call returns: measured, an immediate still showed the previous frame) and checks the timecode. The script prints one JSON result.
- Verification (verify): the exported stills against the reference engine's render of the same frames: "verified"
  (every frame within the DCTL parity limit, spec §11), "contradiction" (a frame is not), "unverified" (no readback or
  no stills: an API call returning True is not evidence).
- Offline fallback (export_bundle): DCTL, LUTs and CDL where the program allows, plus a manifest with the hashes, so
  the grade can be applied by hand when no scripting host is available.
- The grade back (read_grade_script, import_grade): a read-only script lists the clip's nodes (label, tools used,
  LUT path) and exports the whole grade as a .cube; import_grade fits a readable program to that LUT
  (cierra.fit_lut), with the node list as hints. Wheel, curve, window and qualifier values cannot be read through
  the API, and spatial effects are not in a LUT.
- Node for node (node_parts, export_nodes_script, verify_nodes): each Cierra node becomes a CDL (SetCDL) or its own
  DCTL (SetLUT) on the clip's node of the same number, in a new colour version with rollback; the clip must already
  have the nodes (the API cannot add any), and an exported LUT of the result is checked against the program.
"""

import json
import re
from pathlib import Path

import numpy as np

SAFE = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")
VERIFY_P99 = 0.5  # dE00, the parity requirement of spec §11
VERIFY_LIMITS = {"share_over_0.5": 0.001, "max_unclipped": 1.0, "clipped_share": 0.0001}


class HostError(ValueError):
    pass


# ---------------------------------------------------------------- frame identity

def tc_to_frames(tc, fps):
    """'HH:MM:SS:FF' (non-drop) -> frame count. Drop-frame (';') is refused."""
    if ";" in tc:
        raise HostError(f"drop-frame timecode {tc!r} is not supported; use non-drop or frame numbers")
    m = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})", tc)
    if not m or int(fps) != fps:
        raise HostError(f"bad timecode {tc!r} at {fps} fps (integer fps, HH:MM:SS:FF)")
    h, mi, s, f = (int(x) for x in m.groups())
    if f >= fps:
        raise HostError(f"frame field {f} >= fps {fps} in {tc!r}")
    return ((h * 60 + mi) * 60 + s) * int(fps) + f


def frames_to_tc(n, fps):
    if int(fps) != fps or n < 0:
        raise HostError("integer fps and a non-negative frame count only")
    fps = int(fps)
    f, s = n % fps, n // fps
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}:{f:02d}"


def frame_identity(record_frame, item_start, source_start_frame, fps, item_end=None, source_fps=None):
    """Every name of one frame of a timeline item. record_frame: timeline frame (Timeline position);
    item_start: TimelineItem.GetStart(); source_start_frame: TimelineItem.GetSourceStartFrame(); fps: the
    timeline's; source_fps: the clip's own (media pool "FPS"). A clip at another rate steps source_fps / fps source
    frames per timeline frame, rounded down (measured: a 59.94 fps EXR sequence on a 24 fps timeline shows
    source +4 at record +2, not +2). source_tc only for an integer source rate."""
    clip = record_frame - item_start
    if clip < 0 or (item_end is not None and record_frame >= item_end):
        raise HostError(f"record frame {record_frame} is outside the item ({item_start}..{item_end})")
    ratio = (source_fps or fps) / fps
    source = source_start_frame + int(clip * ratio + 1e-6)
    sfps = source_fps or fps
    return {"record_frame": record_frame, "record_tc": frames_to_tc(record_frame, fps), "clip_frame": clip,
            "cierra_frame": clip, "source_frame": source,
            "source_tc": frames_to_tc(source, sfps) if float(sfps).is_integer() else None}


# ---------------------------------------------------------------- apply

def version_name(prog, parent=None):
    from ..format import program_hash
    name = f"cierra-{program_hash(prog)}" + (f"-from-{parent}" if parent else "")
    if not SAFE.match(name):
        raise HostError(f"unsafe version name {name!r}")
    return name


_APPLY = r'''
import json
import time
ARGS = json.loads(__ARGS__)
out = {"ok": False, "steps": [], "stills": []}


def norm(p):
    return p.replace("\\", "/").lower()


def run():
    proj = resolve.GetProjectManager().GetCurrentProject()
    if proj.GetName() != ARGS["project"]:
        out["error"] = "current project is " + proj.GetName() + ", expected " + ARGS["project"]
        return
    tl = proj.GetCurrentTimeline()
    item = tl.GetCurrentVideoItem()
    prev = item.GetCurrentVersion()["versionName"]
    out["previous_version"] = prev

    def rollback(reason):
        # both steps are checked, and the result is read back: a rollback that did not happen is reported as such
        restored = bool(item.LoadVersionByName(prev, 0))
        deleted = ARGS["version"] not in item.GetVersionNameList(0) or bool(item.DeleteVersionByName(ARGS["version"], 0))
        now = item.GetCurrentVersion()["versionName"]
        left = ARGS["version"] in item.GetVersionNameList(0)
        out["error"] = reason
        out["rolled_back"] = restored and deleted and now == prev and not left
        if not out["rolled_back"]:
            out["rollback_failed"] = {"restored": restored, "deleted": deleted, "current": now, "version_left": left}

    if ARGS["version"] in item.GetVersionNameList(0):
        out["error"] = "version " + ARGS["version"] + " already exists"
        return
    if not item.AddVersion(ARGS["version"], 0):
        out["error"] = "could not add the version"
        return
    if not item.LoadVersionByName(ARGS["version"], 0):
        return rollback("could not load the new version")
    out["steps"].append("version")
    proj.RefreshLUTList()
    graph = item.GetNodeGraph()
    if graph.GetNumNodes() < ARGS["node"]:
        return rollback("node %d does not exist" % ARGS["node"])
    if not graph.SetLUT(ARGS["node"], ARGS["lut"]):
        return rollback("SetLUT returned False")
    back = graph.GetLUT(ARGS["node"]) or ""
    out["readback"] = back
    if norm(back) != norm(ARGS["lut"]):
        return rollback("readback " + back + " is not " + ARGS["lut"])
    out["steps"].append("lut")
    for tc, path in ARGS["stills"]:
        ok = tl.SetCurrentTimecode(tc)
        time.sleep(ARGS["wait"])  # the viewer renders after the call returns; an immediate still is the old frame
        at = tl.GetCurrentTimecode()
        ok = ok and at == tc and proj.ExportCurrentFrameAsStill(path)
        out["stills"].append({"tc": tc, "at": at, "path": path, "ok": bool(ok)})
        if not ok:  # a grade that cannot be checked is not left applied
            return rollback("still at " + tc + " failed (playhead at " + str(at) + ")")
    out["steps"].append("stills")
    out["ok"] = True


run()
result = out
'''


_READ = r'''
import json
ARGS = json.loads(__ARGS__)
out = {"ok": False}


def run():
    proj = resolve.GetProjectManager().GetCurrentProject()
    if proj.GetName() != ARGS["project"]:
        out["error"] = "current project is " + proj.GetName() + ", expected " + ARGS["project"]
        return
    tl = proj.GetCurrentTimeline()
    item = tl.GetCurrentVideoItem() if tl else None
    if item is None:
        out["error"] = "no current timeline item"
        return
    mpi = item.GetMediaPoolItem()
    out.update({"timeline": tl.GetName(), "timeline_fps": tl.GetSetting("timelineFrameRate"),
                "drop_frame": tl.GetSetting("timelineDropFrameTimecode"), "timecode": tl.GetCurrentTimecode(),
                "item": item.GetName(), "item_start": item.GetStart(), "item_end": item.GetEnd(),
                "source_start_frame": item.GetSourceStartFrame(),
                "clip_fps": mpi.GetClipProperty("FPS") if mpi else None})
    out["ok"] = True


run()
result = out
'''


def read_script(project):
    """The fixed read-only script: the current item's numbers that frame identity needs (timeline and clip FPS,
    item start / end, source start frame, current timecode). It changes nothing in the host."""
    return _READ.replace("__ARGS__", repr(json.dumps({"project": project})))


def identity_from_read(read, record_frame):
    """frame_identity from read_script's result, with the clip's own rate (no FPS passed by hand)."""
    if not read.get("ok"):
        raise HostError("host read failed: " + str(read.get("error")))
    if str(read.get("drop_frame")).lower() in ("1", "true"):
        raise HostError("drop-frame timeline: not supported; use non-drop or frame numbers")
    try:
        fps, clip_fps = float(read["timeline_fps"]), float(read["clip_fps"] or read["timeline_fps"])
    except (TypeError, ValueError):
        raise HostError(f"unreadable frame rates {read.get('timeline_fps')!r} / {read.get('clip_fps')!r}")
    return frame_identity(record_frame, int(read["item_start"]), int(read["source_start_frame"]), fps,
                          item_end=int(read["item_end"]), source_fps=clip_fps)


def apply_script(project, version, lut, stills, node=1, wait=2.0):
    """The fixed apply script for the scripting host, with this data. project: the (test) project that must be
    current; version: version_name(); lut: the DCTL's path relative to the LUT folder (e.g. 'Cierra/cierra_x.dctl');
    stills: [(record timecode, absolute output path)]. Data is validated and passed as JSON, never as code."""
    if not SAFE.match(version):
        raise HostError(f"unsafe version name {version!r}")
    if not re.fullmatch(r"[A-Za-z0-9_./ -]{1,200}\.(dctl|cube|clf)", lut) or ".." in lut:
        raise HostError(f"unsafe LUT path {lut!r}")
    if not isinstance(node, int) or not 1 <= node <= 64:
        raise HostError("node index 1..64")
    for tc, path in stills:
        tc_to_frames(tc, 24)  # shape check only (HH:MM:SS:FF, non-drop)
        if not re.fullmatch(r"[A-Za-z]:[\\/][A-Za-z0-9_.\\/ -]{1,240}\.(tif|tiff|exr|png|dpx)", path) or ".." in path:
            raise HostError(f"unsafe still path {path!r}")
    if not 0 <= wait <= 10:
        raise HostError("wait 0..10 s")
    args = json.dumps({"project": project, "version": version, "lut": lut, "stills": stills, "node": node,
                       "wait": float(wait)})
    return _APPLY.replace("__ARGS__", repr(args))


# ---------------------------------------------------------------- verify

def verify(result, stills, references, eotf="srgb", requested=None):
    """result: the apply script's JSON; stills / references: lists of display images (3, H, W), same order as the
    stills requested (requested: how many were asked for; default the script's own list). Returns {"status":
    verified | contradiction | unverified, "frames": [...], "reason": ...}.

    A host still carries the host's own output transform and 16-bit quantisation, not only the DCTL, so it is held to
    the spec's p99 <= 0.5 plus two per-pixel limits (VERIFY_LIMITS): at most 0.1 % of pixels above 0.5, and no pixel
    above 1.0 except pixels the still has clipped at display white (a channel >= 0.998), where two ACES 2.0
    implementations may tint a blown highlight differently (measured: one spark pixel per frame, ΔE00 5-23);
    those are counted, reported and limited to 0.01 % of the frame. One bad pixel anywhere else fails the frame."""
    from ..colour import delta_e00_display
    if not result.get("ok") or not result.get("readback"):
        return {"status": "unverified", "frames": [], "reason": result.get("error") or "no LUT readback"}
    meta = result.get("stills", [])
    n = len(meta) if requested is None else requested
    if not stills or len(meta) != n or len(stills) != n or len(references) != n or not all(s.get("ok") for s in meta):
        return {"status": "unverified", "frames": [],
                "reason": f"missing or failed stills ({len(meta)} reported, {len(stills)} images, {n} requested)"}
    lim, frames, bad = VERIFY_LIMITS, [], []
    for i, (img, ref) in enumerate(zip(stills, references)):
        if img.shape != ref.shape:
            return {"status": "unverified", "frames": frames,
                    "reason": f"still is {img.shape}, reference {ref.shape}: resolution or crop differs"}
        img = np.clip(img, 0, 1)
        de = delta_e00_display(img, np.clip(ref, 0, 1), eotf)
        white = img.max(0) >= 0.998
        rest = de[~white]
        f = {"mean": round(float(de.mean()), 4), "p99": round(float(np.quantile(de, 0.99)), 4),
             "max": round(float(de.max()), 4), "max_unclipped": round(float(rest.max()) if rest.size else 0.0, 4),
             "share_over_0.5": round(float((de > 0.5).mean()), 6),
             "clipped_over_1": int((de[white] > 1.0).sum())}
        frames.append(f)
        why = [w for w, failed in (
            (f"p99 {f['p99']} > {VERIFY_P99}", f["p99"] > VERIFY_P99),
            (f"{100 * f['share_over_0.5']:.3f} % of pixels > 0.5", f["share_over_0.5"] > lim["share_over_0.5"]),
            (f"a pixel at {f['max_unclipped']} outside clipped white", f["max_unclipped"] > lim["max_unclipped"]),
            (f"{f['clipped_over_1']} clipped-white pixels > 1.0",
             f["clipped_over_1"] > lim["clipped_share"] * de.size)) if failed]
        if why:
            bad.append(f"frame {i}: " + ", ".join(why))
    if bad:
        return {"status": "contradiction", "frames": frames,
                "reason": "; ".join(bad) + " (check the output transform and the LUT on the node)"}
    return {"status": "verified", "frames": frames, "reason": ""}


# ---------------------------------------------------------------- offline fallback

def export_bundle(prog, folder, stem, parent=None, frames=None):
    """DCTL (+ LUTs and CDL when the program allows them) and manifest.json with the program hash, its parent and the
    frame identities the grade was checked on. Returns the manifest."""
    from ..compile.cdl import CDLError, export_cdl
    from ..compile.dctl import DCTLError, compile_dctl
    from ..compile.lut import LUTError, export_luts
    from ..format import format_program, program_hash
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    files, refused = {}, {}
    try:
        for part in compile_dctl(prog, split=False):
            (folder / f"{stem}.dctl").write_text(part["dctl"])
            files[f"{stem}.dctl"] = "DCTL: ACEScct in / out"
    except DCTLError as e:
        refused["dctl"] = str(e)
    for name, fn in (("lut", lambda: export_luts(prog, folder, stem)), ("cdl", lambda: export_cdl(prog, folder, stem))):
        try:
            files.update(fn())
        except (LUTError, CDLError) as e:
            refused[name] = str(e)
    (folder / f"{stem}.cierra").write_text(format_program(prog))
    files[f"{stem}.cierra"] = "the program (canonical)"
    manifest = {"program": program_hash(prog), "parent": parent, "version": version_name(prog, parent),
                "files": files, "refused": refused, "frames": frames or []}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ---------------------------------------------------------------- the grade, both ways
# What the scripting API allows, and nothing more: per node it reads the label, the names of the tools used (no
# values) and a LUT / DCTL path; per clip it exports the whole grade as a 17 / 33 / 65-point .cube. It writes a LUT or
# DCTL per node (SetLUT), a CDL per node (SetCDL) and node enable. It cannot add nodes, and it cannot read or set
# wheel values, curves, windows or qualifier ranges.

# a drive or POSIX path, never UNC (//server/share, \\server\share): opening one sends Windows credentials
ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/(?![\\/]))[A-Za-z0-9_.\\/ -]{0,230}")
LUT_SIZES = (17, 33, 65)
SPATIAL_TOOLS = ("blur", "sharpen", "window", "noise", "glow", "halation", "grain", "vignette", "mist", "tracker",
                 "depth", "mask", "lens", "defocus", "relight", "face", "warp", "aperture")


def _abs_path(path, what, ext=None):
    if not isinstance(path, str) or not ABS_PATH.fullmatch(path) or ".." in path or (ext and not path.endswith(ext)):
        raise HostError(f"unsafe {what} {path!r} (an absolute path of letters, digits, _ . - / \\ and spaces"
                        + (f", ending {ext}" if ext else "") + ")")
    return path


_READ_GRADE = r'''
import json
import os
ARGS = json.loads(__ARGS__)
out = {"ok": False, "nodes": []}


def graph_nodes(graph):
    nodes = []
    for i in range(1, graph.GetNumNodes() + 1):
        nodes.append({"index": i, "label": graph.GetNodeLabel(i) or "", "tools": list(graph.GetToolsInNode(i) or []),
                      "lut": graph.GetLUT(i) or ""})
    return nodes


def run():
    proj = resolve.GetProjectManager().GetCurrentProject()
    if proj.GetName() != ARGS["project"]:
        out["error"] = "current project is " + proj.GetName() + ", expected " + ARGS["project"]
        return
    tl = proj.GetCurrentTimeline()
    item = tl.GetCurrentVideoItem() if tl else None
    if item is None:
        out["error"] = "no current timeline item"
        return
    graph = item.GetNodeGraph()
    if graph is None:
        out["error"] = "the item has no node graph"
        return
    out["item"] = item.GetName()
    out["version"] = item.GetCurrentVersion()["versionName"]
    out["node_count"] = graph.GetNumNodes()
    out["nodes"] = graph_nodes(graph)
    try:
        group = item.GetColorGroup()
        if group:
            out["group"] = {"name": group.GetName(), "pre_clip": graph_nodes(group.GetPreClipNodeGraph()),
                            "post_clip": graph_nodes(group.GetPostClipNodeGraph())}
    except Exception as e:  # a group that cannot be read is reported, not guessed
        out["group_error"] = str(e)
    kind = {17: resolve.EXPORT_LUT_17PTCUBE, 33: resolve.EXPORT_LUT_33PTCUBE, 65: resolve.EXPORT_LUT_65PTCUBE}
    if not item.ExportLUT(kind[ARGS["size"]], ARGS["path"]) or not os.path.isfile(ARGS["path"]):
        out["error"] = "ExportLUT did not write " + ARGS["path"]
        return
    out["clip_lut"] = ARGS["path"]
    out["lut_size"] = ARGS["size"]
    out["ok"] = True


run()
result = out
'''


def read_grade_script(project, out_dir, lut_size=65):
    """The fixed script that reads the current clip's grade: node count, per node its label, the tools it uses and
    its LUT / DCTL path, the colour group's pre-clip / post-clip nodes when the clip is in a group, and the whole
    grade exported as a lut_size-point .cube into out_dir (cierra_clip_<size>.cube). It changes nothing in the
    project; the only thing it writes is that file."""
    if lut_size not in LUT_SIZES:
        raise HostError(f"lut_size one of {LUT_SIZES}")
    out_dir = _abs_path(out_dir, "output folder").rstrip("\\/")
    sep = "\\" if re.match(r"[A-Za-z]:", out_dir) else "/"
    path = _abs_path(f"{out_dir}{sep}cierra_clip_{lut_size}.cube", "LUT path", ".cube")
    return _READ_GRADE.replace("__ARGS__", repr(json.dumps({"project": project, "path": path, "size": lut_size})))


def import_grade(read, input="acescct", frames=None, lut=None):
    """(Program, report) from read_grade_script's result: the clip LUT fitted as a readable program (cierra.fit_lut),
    with the node list as hints (a node that uses a qualifier may add a keyed node) and in the program's `about`.
    lut: the exported .cube when it was moved since the read. report["warnings"]: what the LUT cannot hold."""
    from ..fit_lut import fit_lut
    if not read.get("ok"):
        raise HostError("host read failed: " + str(read.get("error")))
    nodes = read.get("nodes") or []
    desc = "; ".join(f"{n.get('index')} {n.get('label') or '(no label)'}"
                     + (f" [{', '.join(map(str, n.get('tools') or []))}]" if n.get("tools") else "") for n in nodes)
    about = (f"imported from the grading application: clip {read.get('item', '?')}, version "
             f"{read.get('version', '?')}, {len(nodes)} nodes: {desc}")
    prog, report = fit_lut(lut or read["clip_lut"], input=input, hints={"nodes": nodes}, frames=frames,
                           name=str(read.get("item") or "imported grade"), about=about)
    warn = []
    spatial = [f"node {n.get('index')} uses {t}" for n in nodes for t in (n.get("tools") or [])
               if any(k in str(t).lower() for k in SPATIAL_TOOLS)]
    if spatial:
        warn.append("spatial or position tools are not in an exported LUT, so not in the program: "
                    + "; ".join(spatial))
    group = read.get("group") or {}
    if group.get("pre_clip") or group.get("post_clip"):
        warn.append(f"the clip is in colour group {group.get('name')!r} with group nodes: whether the clip LUT holds "
                    "them depends on the application, check before relying on it")
    if read.get("group_error"):
        warn.append("the colour group could not be read: " + read["group_error"])
    report.update({"source_nodes": nodes, "warnings": warn})
    return prog, report


def node_parts(prog, folder, stem, frame=0, prefer="cdl"):
    """The program node for node, for a clip's serial nodes 1..N: per enabled node a CDL (when the node alone is one
    CDL and prefer="cdl") or its own DCTL written to folder (ACEScct in and out, mask and mix included), or refused
    with the reason (spatial ops, host planes). Returns [{"index", "node", "kind": cdl | dctl | refused, "label",
    "cdl" | "file" | "reason"}]."""
    import dataclasses

    from .. import ast as A
    from ..compile.cdl import CDLError, fold
    from ..compile.dctl import compile_graph
    from ..targets import dctl_node_reasons
    if prog.graph is None:
        raise HostError("the program has no graph")
    items = prog.graph.items if isinstance(prog.graph, A.Serial) else [prog.graph]
    items = [i for i in items if not isinstance(i, A.Pass)]  # `pass` in a chain changes nothing
    if not all(isinstance(i, A.Ref) for i in items):
        raise HostError("node for node needs a serial graph of nodes (no parallel or layer): the application's "
                        "scripting API cannot build a graph")
    if not SAFE.match(stem):
        raise HostError(f"unsafe file stem {stem!r}")
    folder = Path(folder)
    parts = []
    for ref in items:
        node = prog.nodes[ref.name]
        if not node.enabled:
            continue
        part = {"index": len(parts) + 1, "node": node.id, "label": node.label or node.id}
        sub = dataclasses.replace(prog, graph=A.Ref(node.id))
        why = dctl_node_reasons(node)
        try:
            if prefer != "cdl":
                raise CDLError("DCTL preferred")
            slope, offset, power, sat = fold(sub)
            part.update(kind="cdl", cdl={"slope": [float(v) for v in slope], "offset": [float(v) for v in offset],
                                         "power": [float(v) for v in power], "sat": float(sat)})
        except CDLError:
            if why:
                part.update(kind="refused", reason="; ".join(why))
            else:
                folder.mkdir(parents=True, exist_ok=True)
                name = f"{stem}_{part['index']:02d}_{node.id}.dctl"
                title = f"{prog.name or stem}: node {part['index']} {node.id}"
                (folder / name).write_text(compile_graph(sub, sub.graph, frame, title=title))
                part.update(kind="dctl", file=name)
        parts.append(part)
    return parts


_EXPORT = r'''
import json
ARGS = json.loads(__ARGS__)
out = {"ok": False, "steps": [], "set": []}


def norm(p):
    return p.replace("\\", "/").lower()


def run():
    proj = resolve.GetProjectManager().GetCurrentProject()
    if proj.GetName() != ARGS["project"]:
        out["error"] = "current project is " + proj.GetName() + ", expected " + ARGS["project"]
        return
    tl = proj.GetCurrentTimeline()
    item = tl.GetCurrentVideoItem()
    prev = item.GetCurrentVersion()["versionName"]
    out["previous_version"] = prev

    def rollback(reason):
        restored = bool(item.LoadVersionByName(prev, 0))
        deleted = ARGS["version"] not in item.GetVersionNameList(0) or bool(item.DeleteVersionByName(ARGS["version"], 0))
        now = item.GetCurrentVersion()["versionName"]
        left = ARGS["version"] in item.GetVersionNameList(0)
        out["error"] = reason
        out["rolled_back"] = restored and deleted and now == prev and not left
        if not out["rolled_back"]:
            out["rollback_failed"] = {"restored": restored, "deleted": deleted, "current": now, "version_left": left}

    if ARGS["version"] in item.GetVersionNameList(0):
        out["error"] = "version " + ARGS["version"] + " already exists"
        return
    if not item.AddVersion(ARGS["version"], 0):
        out["error"] = "could not add the version"
        return
    if not item.LoadVersionByName(ARGS["version"], 0):
        return rollback("could not load the new version")
    out["steps"].append("version")
    proj.RefreshLUTList()
    graph = item.GetNodeGraph()
    have, need = graph.GetNumNodes(), len(ARGS["parts"])
    out["nodes_available"] = have
    if have < need:
        return rollback("the clip has %d nodes and this grade needs %d serial nodes; the scripting API cannot add "
                        "nodes: add %d serial nodes by hand and run again" % (have, need, need - have))
    for p in ARGS["parts"]:
        i = p["index"]
        if p["kind"] == "cdl":
            c = p["cdl"]
            ok = item.SetCDL({"NodeIndex": str(i), "Slope": c["slope"], "Offset": c["offset"], "Power": c["power"],
                              "Saturation": c["sat"]})
            if not ok:
                return rollback("SetCDL on node %d returned False" % i)
        else:
            if not graph.SetLUT(i, p["lut"]):
                return rollback("SetLUT on node %d returned False" % i)
            back = graph.GetLUT(i) or ""
            if norm(back) != norm(p["lut"]):
                return rollback("node %d reads back %s, not %s" % (i, back, p["lut"]))
        if not graph.SetNodeEnabled(i, True):
            return rollback("could not enable node %d" % i)
        labelled = False
        try:  # labels where the API allows it; a label is not part of the grade
            labelled = bool(graph.SetNodeLabel(i, p["label"]))
        except Exception:
            labelled = False
        out["set"].append({"index": i, "kind": p["kind"], "labelled": labelled})
    out["steps"].append("nodes")
    if ARGS["disable_rest"]:
        for i in range(need + 1, have + 1):
            if not graph.SetNodeEnabled(i, False):
                return rollback("could not disable node %d" % i)
        out["steps"].append("rest disabled")
    if ARGS["verify_lut"]:
        if not item.ExportLUT(resolve.EXPORT_LUT_33PTCUBE, ARGS["verify_lut"]):
            return rollback("ExportLUT for the readback did not work")
        out["readback_lut"] = ARGS["verify_lut"]
        out["steps"].append("readback")
    out["ok"] = True


run()
result = out
'''


def _cdl_text(v):
    v = float(v)
    if not np.isfinite(v):
        raise HostError(f"non-finite CDL value {v}")
    return f"{v:.6f}"


def export_nodes_script(project, version, parts, lut_dir="Cierra", verify_lut=None, disable_rest=False):
    """The fixed script that puts a program on the current clip node for node, transactionally like apply_script:
    a new colour version (version_name()); node i gets the i-th part (SetCDL, or SetLUT with readback for a DCTL),
    a label where the API allows; with disable_rest the nodes after the last part are switched off; with verify_lut
    (an absolute .cube path) the clip's grade is exported for verify_nodes. Any failure loads the previous version
    back and deletes the new one. The clip must already have at least len(parts) serial nodes with no other
    correction on them: the API cannot add nodes, and cannot tell serial from parallel nodes (verify_lut can).
    parts: node_parts(); lut_dir: the DCTL files' folder relative to the application's LUT folder."""
    if not SAFE.match(version):
        raise HostError(f"unsafe version name {version!r}")
    refused = [f"node {p.get('node')}: {p.get('reason')}" for p in parts if p.get("kind") == "refused"]
    if refused:
        raise HostError("cannot go node for node: " + "; ".join(refused))
    if not parts or len(parts) > 64:
        raise HostError("1..64 parts")
    if not re.fullmatch(r"[A-Za-z0-9_ -]{1,60}(/[A-Za-z0-9_ -]{1,60}){0,3}", lut_dir):
        raise HostError(f"unsafe LUT folder {lut_dir!r}")
    data = []
    for k, p in enumerate(parts, 1):
        if p.get("index") != k:
            raise HostError("parts must be numbered 1..N in order")
        label = " ".join(re.sub(r"[^ -~]", " ", str(p.get("label") or "")).split())[:60]
        d = {"index": k, "kind": p.get("kind"), "label": label}
        if d["kind"] == "cdl":
            c = p["cdl"]
            d["cdl"] = {key: " ".join(_cdl_text(v) for v in c[key]) for key in ("slope", "offset", "power")}
            d["cdl"]["sat"] = _cdl_text(c["sat"])
        elif d["kind"] == "dctl":
            if not SAFE.match(str(p.get("file", ""))) or not p["file"].endswith(".dctl"):
                raise HostError(f"unsafe DCTL file name {p.get('file')!r}")
            d["lut"] = f"{lut_dir}/{p['file']}"
        else:
            raise HostError(f"unknown part kind {d['kind']!r}")
        data.append(d)
    if verify_lut is not None:
        _abs_path(verify_lut, "readback LUT path", ".cube")
    args = json.dumps({"project": project, "version": version, "parts": data, "verify_lut": verify_lut,
                       "disable_rest": bool(disable_rest)})
    return _EXPORT.replace("__ARGS__", repr(args))


def verify_nodes(result, prog, lut=None, input="acescct"):
    """The export script's result and the clip LUT it exported (lut: the file, when it was moved) against the
    program: "verified" when every lattice node of that LUT (exact samples, no interpolation) is within mean 0.1 /
    p95 0.5 ΔE00 of the program, "contradiction" when not (a node that is not serial, another correction on a node,
    a clamp), "unverified" without a readback LUT."""
    from ..fit_lut import read_cube, residual
    path = lut or result.get("readback_lut")
    if not result.get("ok") or not path:
        return {"status": "unverified", "reason": result.get("error") or "no readback LUT (pass verify_lut)"}
    cube, dmin, dmax = read_cube(path)
    de = residual(prog, cube, input, dmin, dmax)["lattice"]
    ok = de["mean"] <= 0.1 and de["p95"] <= 0.5
    return {"status": "verified" if ok else "contradiction", "lattice": de,
            "reason": "" if ok else f"the clip's LUT is {de['mean']} mean / {de['p95']} p95 ΔE00 from the program"}
