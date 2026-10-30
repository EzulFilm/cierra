"""Versions of a grade, like commits: each version is the canonical program text, its hash, its parent, a message
(why it changed), the measurements, and a diff a colourist can read ("node warm_skin: saturation amount 1.10 -> 1.16").

Layout of a history folder:
    <dir>/v0001_<hash>.cierra     canonical source
    <dir>/log.jsonl               one JSON line per version: id, hash, parent, message, measurements, diff
"""

import json
import re
from pathlib import Path

from .format import fcall, fgraph, fmask, fval, format_program, program_hash
from .ops import OPS, OUTPUTS

VERSION_FILE = re.compile(r"v\d{4,}_[0-9a-f]+\.cierra")  # what commit() writes; the log is not trusted


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return fval(v)


def diff(old, new):
    """Readable semantic diff between two checked programs: list of lines."""
    out = []
    if old is None:
        keys = f", {len(new.keys)} key statements" if new.keys else ""
        return [f"new program: {len(new.nodes)} nodes{keys}"]
    for attr in ("name", "input", "reference_width", "fps", "seed"):
        a, b = getattr(old, attr), getattr(new, attr)
        if a != b:
            out.append(f"{attr}: {_fmt(a)} -> {_fmt(b)}")
    if fcall(old.output, OUTPUTS[old.output.name]) != fcall(new.output, OUTPUTS[new.output.name]):
        out.append(f"output: {fcall(old.output, OUTPUTS[old.output.name])} -> {fcall(new.output, OUTPUTS[new.output.name])}")
    for kid in old.keys:
        if kid not in new.keys:
            out.append(f"- key {kid} removed ({fmask(old.keys[kid].mask)})")
    for kid, k in new.keys.items():
        if kid not in old.keys:
            out.append(f"+ key {kid} added: {fmask(k.mask)}")
        elif fmask(old.keys[kid].mask) != fmask(k.mask):
            out.append(f"~ key {kid}: {fmask(old.keys[kid].mask)} -> {fmask(k.mask)}")
    for nid in old.nodes:
        if nid not in new.nodes:
            out.append(f"- node {nid} removed ({fcall(old.nodes[nid].call, OPS[old.nodes[nid].call.name])})")
    for nid, n in new.nodes.items():
        if nid not in old.nodes:
            line = f"+ node {nid} added: {fcall(n.call, OPS[n.call.name])}"
            if n.mask is not None:
                line += f" where {fmask(n.mask)}"
            out.append(line)
            continue
        o = old.nodes[nid]
        if o.call.name != n.call.name:
            out.append(f"~ node {nid}: {fcall(o.call, OPS[o.call.name])} -> {fcall(n.call, OPS[n.call.name])}")
        else:
            for key in OPS[n.call.name].params:
                a, b = o.call.args.get(key), n.call.args.get(key)
                if _fmt(a) != _fmt(b):
                    out.append(f"~ node {nid}: {n.call.name} {key} {_fmt(a)} -> {_fmt(b)}")
        om = fmask(o.mask) if o.mask is not None else "none"
        nm = fmask(n.mask) if n.mask is not None else "none"
        if om != nm:
            out.append(f"~ node {nid}: mask {om} -> {nm}")
        if _fmt(o.mix) != _fmt(n.mix):
            out.append(f"~ node {nid}: mix {_fmt(o.mix)} -> {_fmt(n.mix)}")
        if o.enabled != n.enabled:
            out.append(f"~ node {nid}: {'on' if n.enabled else 'off'}")
    for g in ("photo_graph", "graph"):
        a = fgraph(getattr(old, g)) if getattr(old, g) is not None else "none"
        b = fgraph(getattr(new, g)) if getattr(new, g) is not None else "none"
        if a != b:
            out.append(f"{g}: {a}  ->  {b}")
    return out or ["no change"]


class History:
    def __init__(self, folder):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "log.jsonl"

    def entries(self):
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines() if line.strip()]

    def head(self):
        e = self.entries()
        return e[-1] if e else None

    def load(self, entry):
        from .checker import load
        name = entry.get("file")
        if not isinstance(name, str) or not VERSION_FILE.fullmatch(name):
            raise ValueError(f"history entry names {name!r}, not a version file in {self.dir}")
        return load((self.dir / name).read_text(encoding="utf-8"))

    def commit(self, prog, message, measurements=None, parent_prog=None):
        entries = self.entries()
        h = program_hash(prog)
        vid = len(entries) + 1
        fname = f"v{vid:04d}_{h}.cierra"
        (self.dir / fname).write_text(format_program(prog), encoding="utf-8")
        parent = entries[-1] if entries else None
        if parent_prog is None and parent is not None:
            parent_prog = self.load(parent)
        entry = {"id": vid, "hash": h, "parent": parent["hash"] if parent else None, "file": fname,
                 "message": message, "measurements": measurements or {}, "diff": diff(parent_prog, prog)}
        with self.log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

