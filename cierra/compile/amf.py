"""ACES Metadata File (AMF v2.0, urn:ampas:aces:amf:v2.0) next to the CLF export.

An AMF describes an ACES pipeline: input transform, looks, output transform, by Academy transform IDs or files. Here:
  - the look is a CLF in ACES2065-1 (AMF look files are ACES2065-1 in and out; only a CDL has a working-space
    wrapper): ACES2065-1 -> ACEScct, Cierra's fitted 65^3 lattice (the same one as `cierra lut`'s ACEScct CLF),
    ACEScct -> ACES2065-1, written by OpenColorIO from its ACES 2.0 studio config, inline (one file, no references);
  - the input transform: the camera's Academy CSC ID when the program's input is a camera log the config names
    (e.g. S-Log3), else none (scene-linear Rec.709 and display-referred inputs have no AMF input ID);
  - the output transform: the ACES 2.0 Output ID for the program's output (aces2, SDR Rec.709 on sRGB or BT.1886).
Only pointwise programs (what a LUT can hold); the rest is refused with the reason, as for the LUT targets.
The AMF validates against the Academy's schema (github.com/ampas/aces-amf, schema/acesMetadataFile.xsd).
"""

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from .. import colour as C
from ..format import program_hash

NS = "urn:ampas:aces:amf:v2.0"
TID = "urn:ampas:aces:transformId:v2.0:"
OUTPUTS = {"srgb": "Output.Academy.Rec709-D65_100nit_in_Rec709-D65_sRGB-Piecewise.a2.v1",
           "rec1886": "Output.Academy.Rec709-D65_100nit_in_Rec709-D65_BT1886.a2.v1"}
# the OpenColorIO display + view each Output ID means (the reader side, cierra.compile.amf.resolve)
DISPLAYS = {"Rec709-D65_sRGB-Piecewise": "sRGB - Display", "Rec709-D65_BT1886": "Rec.1886 Rec.709 - Display"}
VIEW = "ACES 2.0 - SDR 100 nits (Rec.709)"
INPUTS = {"sony_slog3_sgamut3": "CSC.Sony.SLog3_SGamut3_to_ACES.a2.v1",
          "sony_slog3_sgamut3cine": "CSC.Sony.SLog3_SGamut3Cine_to_ACES.a2.v1",
          "arri_logc3": "CSC.Arri.LogC3_to_ACES.a2.v1", "arri_logc4": "CSC.Arri.LogC4_to_ACES.a2.v1",
          "panasonic_vlog": "CSC.Panasonic.VLog_VGamut_to_ACES.a2.v1", "red_log3g10": "CSC.Red.Log3G10_RWG_to_ACES.a2.v1",
          "canon_log2": "CSC.Canon.CLog2_CGamut_to_ACES.a1.v1", "canon_log3": "CSC.Canon.CLog3_CGamut_to_ACES.a1.v1",
          "apple_log": "CSC.Apple.AppleLog_BT2020_to_ACES.a2.v1", "dji_dlog": "CSC.DJI.DLog_DGamut_to_ACES.a1.v1",
          "bmd_film_gen5": "CSC.Blackmagic.BMDFilm_WideGamut_Gen5_to_ACES.a2.v1",
          "acescct": "CSC.Academy.ACEScct_to_ACES.a2.v1"}


class AMFError(ValueError):
    pass


def _config():
    import PyOpenColorIO as ocio
    return ocio.Config.CreateFromBuiltinConfig(C.ACES2_CONFIG)


def write_look_clf(prog, path, size=65):
    """The look as an ACES2065-1 -> ACES2065-1 CLF (OpenColorIO writes it, inline)."""
    import tempfile

    import PyOpenColorIO as ocio

    from .lut import LUTError, bake, write_clf
    try:
        lmt = bake(prog, size)
    except LUTError as e:
        raise AMFError(f"an AMF look needs a LUT: {e}") from None
    cfg = _config()
    with tempfile.TemporaryDirectory() as tmp:
        inner = Path(tmp) / "lmt.clf"
        write_clf(inner, lmt, prog.name or "look", program_hash(prog), "ACEScct (AP1)", "ACEScct (AP1)")
        group = ocio.GroupTransform([
            ocio.ColorSpaceTransform(src="ACES2065-1", dst="ACEScct"),
            ocio.FileTransform(src=str(inner), interpolation=ocio.INTERP_TETRAHEDRAL),
            ocio.ColorSpaceTransform(src="ACEScct", dst="ACES2065-1")])
        flat = cfg.getProcessor(group).createGroupTransform()
        meta = flat.getFormatMetadata()
        meta.setName(f"{prog.name or 'Cierra look'} (Cierra {program_hash(prog)})")
        meta.setID(f"cierra-{program_hash(prog)}")
        meta.addChildElement("Description", "Cierra look, ACES2065-1 in and out: ACES2065-1 -> ACEScct, the "
                             "program's 65^3 lattice (tetrahedral), ACEScct -> ACES2065-1")
        meta.addChildElement("InputDescriptor", "ACES2065-1")
        meta.addChildElement("OutputDescriptor", "ACES2065-1")
        flat.write(formatName="Academy/ASC Common LUT Format", fileName=str(path), config=cfg)
    return path


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def amf_text(prog, clf_name, clf_sha256, input=None, when=None, ids=None):
    """The AMF v2.0 document for a program whose look is the CLF `clf_name`."""
    out = prog.output
    if out.name != "aces2":
        raise AMFError(f"output {out.name}(): an AMF output transform is an ACES Output; use aces2(display: srgb | "
                       "rec1886)")
    display = out.args.get("display", "srgb")
    if display not in OUTPUTS:
        raise AMFError(f"aces2(display: {display}) has no AMF Output ID here (srgb, rec1886)")
    kind = input or prog.input
    when = when or _now()
    ids = ids or [uuid.uuid4(), uuid.uuid4()]
    title = escape(prog.name or "Cierra look")
    dt = (f"<aces:dateTime><aces:creationDateTime>{when}</aces:creationDateTime>"
          f"<aces:modificationDateTime>{when}</aces:modificationDateTime></aces:dateTime>")
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             f'<aces:acesMetadataFile version="2.0" xmlns:aces="{NS}">',
             "  <aces:amfInfo>",
             f"    <aces:description>{title} (Cierra program {program_hash(prog)})</aces:description>",
             f"    {dt}",
             f"    <aces:uuid>urn:uuid:{ids[0]}</aces:uuid>",
             "  </aces:amfInfo>",
             "  <aces:pipeline>",
             "    <aces:pipelineInfo>",
             f"      <aces:description>{title}</aces:description>",
             f"      {dt}",
             f"      <aces:uuid>urn:uuid:{ids[1]}</aces:uuid>",
             "      <aces:systemVersion><aces:majorVersion>2</aces:majorVersion><aces:minorVersion>0"
             "</aces:minorVersion><aces:patchVersion>0</aces:patchVersion></aces:systemVersion>",
             "    </aces:pipelineInfo>"]
    if kind in INPUTS:
        lines += ['    <aces:inputTransform applied="false">',
                  f"      <aces:transformId>{TID}{INPUTS[kind]}</aces:transformId>",
                  "    </aces:inputTransform>"]
    lines += ['    <aces:lookTransform applied="false">',
              f"      <aces:description>{title}: Cierra look, ACES2065-1 in and out</aces:description>",
              f'      <aces:hash algorithm="http://www.w3.org/2001/04/xmlenc#sha256">{clf_sha256}</aces:hash>',
              f"      <aces:file>{escape(clf_name)}</aces:file>",
              "    </aces:lookTransform>",
              '    <aces:outputTransform applied="false">',
              f"      <aces:transformId>{TID}{OUTPUTS[display]}</aces:transformId>",
              "    </aces:outputTransform>",
              "  </aces:pipeline>",
              "</aces:acesMetadataFile>", ""]
    return "\n".join(lines)


def export_amf(prog, folder, stem, input=None):
    """Write <stem>_look.clf (ACES2065-1 in / out) and <stem>.amf; return {file: description}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    clf = folder / f"{stem}_look_ACES2065-1.clf"
    write_look_clf(prog, clf)
    sha = hashlib.sha256(clf.read_bytes()).hexdigest()
    (folder / f"{stem}.amf").write_text(amf_text(prog, clf.name, sha, input))
    return {clf.name: "the look as CLF, ACES2065-1 in and out (the AMF's look file)",
            f"{stem}.amf": "ACES Metadata File v2.0: input (camera CSC ID where known), the look file, ACES 2.0 output"}


def resolve(amf_path):
    """Read an AMF written here back into OpenColorIO terms: {"input": OCIO colour space or None, "looks": [CLF
    paths, checked against their hashes], "display", "view"}. The reader side of the round trip."""
    import xml.etree.ElementTree as ET
    ns = {"aces": NS}
    amf_path = Path(amf_path)
    root = ET.parse(amf_path).getroot()
    pipe = root.find("aces:pipeline", ns)
    res = {"input": None, "looks": [], "display": None, "view": None}
    it = pipe.find("aces:inputTransform/aces:transformId", ns)
    if it is not None:
        tid = it.text.strip().removeprefix(TID)
        res["input"] = next((C.CAMERA_INPUTS.get(k, "ACEScct") for k, v in INPUTS.items() if v == tid), None)
    for lt in pipe.findall("aces:lookTransform", ns):
        f = amf_path.parent / lt.find("aces:file", ns).text.strip()
        h = lt.find("aces:hash", ns)
        if h is not None and hashlib.sha256(f.read_bytes()).hexdigest() != h.text.strip():
            raise AMFError(f"{f.name}: hash differs from the AMF")
        res["looks"].append(f)
    ot = pipe.find("aces:outputTransform/aces:transformId", ns).text.strip().removeprefix(TID)
    enc = ot.split("_in_", 1)[1].rsplit(".a", 1)[0]
    if enc not in DISPLAYS:
        raise AMFError(f"output {ot}: no display mapped")
    res["display"], res["view"] = DISPLAYS[enc], VIEW
    return res


def render_amf(amf_path, aces2065):
    """Apply an AMF's looks and output to ACES2065-1 pixels (3, H, W) through OpenColorIO: display code values."""
    import PyOpenColorIO as ocio
    r = resolve(amf_path)
    cfg = _config()
    steps = [ocio.FileTransform(src=str(p), interpolation=ocio.INTERP_TETRAHEDRAL) for p in r["looks"]]
    steps.append(ocio.DisplayViewTransform(src="ACES2065-1", display=r["display"], view=r["view"]))
    proc = cfg.getProcessor(ocio.GroupTransform(steps)).getDefaultCPUProcessor()
    return C._ocio_apply(proc, aces2065)
