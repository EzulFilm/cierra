# Changelog

## Unreleased

- Display-referred LUTs import (`import-lut --input rec709_display` or `srgb_display`): Rec.709 video LUTs, display
  codes in and out, come back as a program with a photo input and `display_encode` output (no tone scale). New input
  `rec1886_photo`: display-referred Rec.709 video decoded with gamma 2.4.
- TIFF input: 8-bit, 16-bit and float TIFFs load at full depth (tifffile, in the `images` extra).
- The grading loop keeps a model's values inside the tuner's bounds before measuring (`CIERRA_BOUND=0` turns it off);
  values it pulled in are recorded as "capped" and told to the model.

- Key routing, as in the grading app's node graph (spec §6.1):
  - `key sky = graduated(...)`: a named mask, made again wherever it is used;
  - `key(face)` in a mask: node face's key, carried from where face ran (made on face's input, before its grade), so
    `mask not key(face)` is the outside node, exact even after face changed the colours it keyed;
  - the key mixer: `+` (add), `-` (subtract), `and` (multiply), `or` (max), and `key(name, gain: g, offset: o)`;
  - compiled to DCTL / WGSL / CUDA (a carried key is made once and passed along) and to LUTs when pointwise; checked
    for unknown names, cycles, order (the keyed node runs once, before its readers) and expanded size.

- Looks re-synced to the current version of the engine they were first written for:
  - `exposure` and `white_balance` are linear-light gains (decode, multiply, encode), as before above the ACEScct toe
    and no longer lifting black below it. They no longer fold into an ASC CDL (use `printer_lights` / `cdl()`), and a
    LUT fit reads them back as gains;
  - the qualifier's hue key no longer fades out toward neutral; a pixel with chroma below 1e-5 has hue 0° on every
    target;
  - the reference renders the looks are tested against were rebuilt from that version.
- The reference engine runs on an NVIDIA GPU (`pip install -e ".[cuda]"`, `--device gpu` or `CIERRA_DEVICE=gpu`): the
  same float64 code through CuPy, tested against the CPU. OpenColorIO transforms use 4 threads (identical results).
  A 1080p S-Log3 frame with an 11-node grade: 14.9 s before, 7.4 s on the CPU now, 0.7 s on the GPU.
- CUDA target (`cierra cuda`): one self-contained kernel source (one thread per pixel, ACEScct float buffers in and
  out) from the generator DCTL and WGSL share; tested on the CPU by compiling it as C++ with g++, optional nvcc runner.
- DCTL / WGSL / CUDA grain clamp the grain size to 0.001 pixel as the engine and the spec do (only very small frames).
- New pointwise ops, compiled to every target: `rgb_mixer`, `sat_vs_lum`, `density` (subtractive saturation),
  `hue_rotate`, `vibrance`, `split_tone`, `gamut_compress` (the open ACES reference gamut compression).
- Layer blend modes screen, overlay, subtract and difference.
- Test frames, goldens and evaluation sets on CC0 camera RAW photos (raw.pixls.us, pinned by SHA-256); the Netflix
  Open Content frames are gone from the tree.
- README figures from four CC0 camera RAWs (raw.pixls.us, downloaded pinned by SHA-256) and five example grades,
  instead of the Netflix test frames.
- `pass` in a graph: the unchanged input, for a layer bottom or a parallel branch (`layer(pass, glow, mode: screen)`);
  `pass` is now a reserved word.
- `rgb_key` mask, and matte clip black / white (`clip_black`, `clip_white`) on every mask.
- `box` (rotated, rounded rectangle) and `polygon` windows, in every target.
- Camera log inputs through OpenColorIO (ARRI, Sony, Panasonic, Canon, RED, Blackmagic, Apple, DJI; linear Rec.2020 /
  P3-D65) and ACES 2.0 outputs for P3 and HDR (PQ 1000 / 2000 / 4000 nits, HLG); the loop keeps measuring SDR Rec.709.
- `cierra import-lut grade.cube out.cierra` and `cierra.fit_lut.fit_lut`: fit a short, readable program to a 3D LUT
  (ACEScct, ACEScg or linear Rec.709 domain) and report the residual in ΔE00 over the lattice and test frames.
- Grading application adapter: a read-only script that lists the clip's nodes and exports its grade as a .cube, and
  `import_grade` to fit a program to it; node-for-node export (a CDL or a DCTL per node) in a new colour version,
  with rollback and a check of the exported result. The API cannot add nodes or read wheel, curve, window or
  qualifier values, and the README says what that means.

## 0.1.0: first public release

The language
- Cierra 0.1: nodes, masks and a graph in plain text; 24 ops, 7 masks (`and` / `or` / `not`), keyframes, parallel
  branches and layer blending. The specification (docs/spec.md) is normative; docs/conformance.md maps each section
  to the tests that hold it.
- A checker with types, units and ranges for every parameter, limits on size, nesting, references and numbers, and
  every problem reported at once. Strings are one line of printable text.

Engine and targets
- NumPy reference engine (float64), ACES 2.0 output through OpenColorIO.
- Compilers to DCTL, WGSL (WebGPU), 3D LUTs (.cube and CLF v3) and ASC CDL, each tested against the engine; a target
  that cannot express an op refuses it with a reason, and DCTL / WGSL can split a graph around spatial ops.

Grading with notes
- `cierra loop`: notes in, checked versions out, each with a readable diff, measurements and a damage gate.
  Drivers: a deterministic rules driver, a local open-weight model (llama.cpp, with grammar-constrained replies) and
  the Anthropic API.
- Note checks (right way / no change / wrong way, and which node caused it), a numeric tuner, and two held-out
  evaluation sets.

Tools
- `cierra` command line: check, render, dctl, wgsl, lut, cdl, fit, loop.
- `tools/raw_to_video.py` and `make pipeline-demo`: camera RAW stills to a graded ProRes-compatible or H.264 movie.
- Makefile for setup, tests, builds, evaluations and a pinned, hash-checked local model.
- Example looks and commented example programs.
