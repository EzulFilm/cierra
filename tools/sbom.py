"""Write docs/sbom.json: every installed distribution in Cierra's venv with version and licence, marked runtime /
llm / dev, and fail if a runtime or llm dependency (or anything they pull in) is GPL / AGPL / LGPL / non-commercial,
or carries a licence that is neither a standard permissive identifier nor reviewed by hand (REVIEWED).

    python tools/sbom.py            # check and rewrite docs/sbom.json
    python tools/sbom.py --check    # check only (CI)
"""

import json
import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = {"numpy", "scipy", "opencolorio"}
LLM = {"anthropic"}
IMAGES = {"pillow", "tifffile"}  # the `images` extra: command-line image I/O and evaluation sheets
RAW = {"rawpy"}  # the `raw` extra: tools/raw_to_video.py
CUDA = {"cupy-cuda12x"}  # the `cuda` extra: the reference engine on an NVIDIA GPU (cierra.backend)
# NVIDIA's CUDA runtime libraries, pulled in by cupy's `ctk` extra: proprietary (NVIDIA EULA), downloaded by the user
# from PyPI when they ask for GPU rendering, never shipped or bundled with Cierra; listed, not licence-checked.
NVIDIA = re.compile(r"^(nvidia-.*-cu\d+|cuda-toolkit)$")
BAD = re.compile(r"\b(A?GPL|LGPL|GNU (Affero )?General Public|Non[- ]?Commercial|CC[- ]BY[- ]NC)", re.I)
# Accepted without a manual review (permissive, Apache-2.0-compatible). Anything else on a runtime / llm dependency
# (UNKNOWN, a LicenseRef, free text) fails until it is reviewed and listed in REVIEWED.
APPROVED = re.compile(r"^(MIT( License)?|MIT-CMU|BSD( License)?|BSD-[23]-Clause|0BSD|ISC( License)?|Zlib|CC0-1\.0|"
                      r"Apache Software License|Apache-2\.0|PSF-2\.0|Python Software Foundation License|HPND|"
                      r"(The )?Unlicense)$", re.I)
# name -> (licence as reviewed, where it was checked): for packages whose metadata is not a standard identifier.
REVIEWED = {
    "opencolorio": ("BSD-3-Clause", "LICENSE in the OpenColorIO repository; PyPI metadata gives only a copyright line"),
    "wgpu": ("BSD-2-Clause", "LICENSE in the wgpu-py 0.32.0 wheel (metadata says 'BSD 2-Clause License'); the bundled "
                             "libwgpu_native is MIT OR Apache-2.0 (gfx-rs/wgpu-native). Dev / tests only"),
    "rendercanvas": ("BSD-2-Clause", "LICENSE in the rendercanvas 2.7.2 wheel (no licence field in its metadata); "
                                     "pulled in by wgpu. Dev / tests only"),
    "openexr": ("BSD-3-Clause", "LICENSE.md in the OpenEXR repository (AcademySoftwareFoundation/openexr); the wheel's "
                                "metadata gives none. Dev / tests only"),
    "rawpy": ("MIT (wrapper); bundles LibRaw, LGPL-2.1 or CDDL-1.0 at the user's choice",
              "LICENSE in the rawpy 0.27.1 wheel and LibRaw's COPYRIGHT; optional `raw` extra used only by "
              "tools/raw_to_video.py, never imported by the library"),
    "colour-science": ("BSD-3-Clause", "LICENSE in the colour repository (colour-science/colour); metadata says only "
                                       "'OSI Approved'. Dev / tests only"),
}
# Not Python distributions: what `make llm-setup` fetches (tools/setup_llm.sh). Nothing of it is shipped with Cierra.
EXTERNAL = [
    {"name": "FFmpeg", "version": "user-installed", "licence": "LGPL-2.1+ (GPL when built with x264 / x265)",
     "role": "external-program", "reviewed": "called as a separate process by tools/raw_to_video.py; not shipped or "
     "linked. ProRes-compatible output uses FFmpeg's prores_ks encoder"},
    {"name": "llama.cpp", "version": "b11580 (4413a19)", "licence": "MIT", "role": "llm-server",
     "reviewed": "LICENSE in ggml-org/llama.cpp; built from source by tools/setup_llm.sh"},
    {"name": "Qwen2.5-7B-Instruct-GGUF (Q4_K_M)", "version": "SHA-256 in tools/setup_llm.sh", "licence": "Apache-2.0",
     "role": "llm-weights", "reviewed": "model card of Qwen/Qwen2.5-7B-Instruct-GGUF; downloaded on request, not shipped"},
    {"name": "Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M)", "version": "SHA-256 in tools/setup_llm.sh", "licence": "Apache-2.0",
     "role": "llm-weights", "reviewed": "model card of Qwen/Qwen2.5-1.5B-Instruct-GGUF; downloaded on request, not shipped"},
]


def licence(dist):
    md = dist.metadata
    expr = md.get("License-Expression")
    if expr:
        return expr
    classifiers = [c.split("::")[-1].strip() for c in md.get_all("Classifier") or [] if c.startswith("License ::")]
    if classifiers:
        return "; ".join(classifiers)
    lic = (md.get("License") or "").strip().splitlines()
    return lic[0][:80] if lic else "UNKNOWN"


def approved(lic):
    parts = [x.strip(" ()") for x in re.split(r";|\bAND\b|\bOR\b", lic)]
    return all(APPROVED.match(x) for x in parts if x)


def closure(names, dists):
    seen, todo = set(), list(names)
    while todo:
        n = todo.pop()
        if n in seen or n not in dists:
            continue
        seen.add(n)
        for req in dists[n].requires or []:
            if "extra ==" in req:
                continue
            todo.append(re.split(r"[ ;<>=!~\[(]", req, 1)[0].lower().replace("_", "-"))
    return seen


def main():
    dists = {d.metadata["Name"].lower().replace("_", "-"): d for d in metadata.distributions()}
    run, llm, img, raw = closure(RUNTIME, dists), closure(LLM, dists), closure(IMAGES, dists), closure(RAW, dists)
    cuda = closure(CUDA, dists)
    rows, problems = [], []
    for name in sorted(dists):
        d = dists[name]
        role = ("runtime" if name in run else "llm" if name in llm else "images" if name in img else "raw" if name in raw
                else "cuda" if name in cuda else "cuda-runtime" if NVIDIA.match(name) else "dev")
        lic = licence(d)
        row = {"name": name, "version": d.version, "licence": lic, "role": role}
        if role == "cuda-runtime":
            row["reviewed"] = ("NVIDIA proprietary (CUDA Toolkit EULA); installed by the user with the optional `cuda` "
                               "extra, loaded by CuPy at run time, never shipped with Cierra")
        if name in REVIEWED and not BAD.search(lic):  # dev rows too, so the SBOM shows the reviewed licence
            row["licence"], row["reviewed"] = REVIEWED[name]
        if role not in ("dev", "cuda-runtime"):
            if BAD.search(lic):
                problems.append(f"{name} {d.version}: {lic} (not Apache-compatible)")
            elif name not in REVIEWED and not approved(lic):
                problems.append(f"{name} {d.version}: {lic!r} is not a reviewed licence (check the project's licence, "
                                "then add it to REVIEWED)")
        rows.append(row)
    if "--check" not in sys.argv:
        (ROOT / "docs" / "sbom.json").write_text(json.dumps({"format": "cierra-sbom-1", "packages": rows, "external": EXTERNAL}, indent=1)
                                                 + "\n")
    for r in rows:
        print(f"{r['role']:8s} {r['name']:28s} {r['version']:12s} {r['licence']}")
    if problems:
        print("LICENCE CHECK FAILED:", *problems, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
