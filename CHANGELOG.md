# Changelog

## Unreleased

- `cierra amf PROGRAM OUT_DIR [--in KIND]`: the AMF export from the command line (the look CLF and the `.amf`);
  a program a LUT cannot hold, or an output that is not ACES 2.0 SDR, exits 1 with the reason.
- Translation guide for shorthand notes: three rules, a glossary of trade words and two more examples in the
  translation prompt. On by default for local open-weight models, off for hosted ones; `CIERRA_TRANSLATE_GUIDE=1` / `0`
  forces it. On a blind set of hard notes a 7B went from 21 to 30 of 40 notes graded the right way without damage.
- In tuner mode the structure round's note checks come from the translation, not from words in the raw note.
- Web playground (`web/playground`, `tools/playground_server.py`): paste a program, pick or drop a frame, and the
  grade, its output transform and scopes run in the browser on WebGPU. `tools/build_playground_static.py` builds a
  version with no server, where Cierra's compiler runs in the browser through Pyodide. Measured against the reference
  engine in headless Chromium (`tools/playground_parity.py`): shader max 0.003 dE00, on screen p99.9 0.36.
- Tuner damage guard (on by default; `CIERRA_TUNER_GUARD=0` restores the old tuner): a step that adds a damage flag is
  refused and the best undamaged step kept; then a node, or a pair of nodes, that causes a flag the program started
  with is eased toward neutral when that loses no met check.
- `cierra/assist.py`: the checked route for an assistant (checker and bounds, damage gate, note checks, DCTL compile,
  a pre-flight of the compiled DCTL on the frame, a package with file hashes, and a check of the host's own stills).
  `cierra/hosts/sim.py` simulates a host for tests; no grading application has run it yet.
- `cierra.compile.amf.export_amf`: the look as a CLF in ACES2065-1 and an ACES Metadata File (AMF v2.0) with the
  camera input ID, the look file's hash and the ACES 2.0 output ID; read back through OpenColorIO.
- DCTL for Rec.709 projects: the ACES 2.0 output LUT is 97^3 and fitted for tetrahedral interpolation (was 65^3
  sampled); `--lut-size` default 97.
- `cierra.qc.intended_black_lift` and `damage(..., allow=)`: a look can state its own shadow lift, which the damage
  gate then allows on top of its threshold.
- Relay driver (`CIERRA_DRIVER=relay`, `CIERRA_RELAY_DIR`): a person or another program answers the loop's prompts
  through files. `tools/relay_eval.py` runs an evaluation that way; `tools/bench_notes.py` scores any run by outcome
  (right way and undamaged, damaged, missed); `tools/replay_tuner.py` re-tunes a stored run.
- Fixed: temporal `align` could move a still, noisy frame by a fraction of a pixel (the sub-pixel refinement is now
  kept only when it lowers the match error by a tenth).
- `cierra scopes clip.mov --from F0 --to F1 --step K`: numeric per-frame tone, histogram, waveform, vectorscope,
  neutral-cast, skin and clipping measurements, with source frame numbers and mean-luma flicker across the selection.
- `loop --viewing-width N`: render first, then box-downsample the output and reference to the viewing size for note
  checks, damage, side effects and reference error; the width is recorded in history. Contrast note tolerances now use
  5% of the frame's own mid-tone spread, capped by the previous fixed tolerance, so flat frames can register a move.
- Temporal `align` refines the whole-pixel pan estimate to 1/8 pixel and resamples each neighbour once, bilinearly;
  the patch-difference rejection of independent local motion is unchanged.
- Note checks report side effects: each loop round measures the qualities no note asked about (warmth, brightness,
  mid-tone contrast, colourfulness around the frame's mean tint) and flags any that moved more than about 1 dE00 at
  display mid grey ("also moved, no note asked: brightness 0.31 -> 0.38"), so a note met by moving something else is
  visible. In the round record (`side_effects`) and the log; the verdict and the model's feedback are unchanged.
- DCTL: blur, sharpen, glow, halation and diffusion compile to DCTL when the kernel radius (int(3 sigma)) is at most
  12 px at the reference width, one such node per DCTL. It reads its neighbours through the texture entry point, and
  the nodes before it run again for every sampled pixel, so the result equals the engine within float32 (sharpen on
  Rec.709 video: 0.0004 code value). The radius scales with the frame width, so a 4K frame takes four times the
  samples. `--split` starts a new DCTL before a second such node. Wide sharpen, local contrast and
  highlights/shadows compile as two DCTLs on adjacent nodes: the first writes the horizontal luma Gaussian to alpha,
  the second blurs alpha vertically and applies the op. Guided detail, denoise and wide RGB blurs/glows stay refused.
  The two-pass result is checked against the engine through the CPU harness; the application's alpha hand-off is a
  separate manual check. Checked in the grading application (21.1): the small-radius texture-form DCTL compiles.
- DCTL for a normal Rec.709 project (`cierra dctl look.cierra out --in KIND [--out program|srgb|rec1886]`): the
  DCTL decodes its input to ACEScct first (Rec.709 / sRGB video, or a camera log: S-Log3 with S-Gamut3 or
  S-Gamut3.Cine, ARRI LogC3 and LogC4, V-Log, Apple Log, D-Log) and encodes the result for the display, so a grade
  goes on a node of a YRGB timeline without colour management. The log curves are written out; the gamut matrices are
  fitted to OpenColorIO's studio config (grey within 2e-5 cv, colours within 4e-3 cv). A `display_encode` output is
  written as code; an ACES 2.0 output goes through a 65-point .cube written next to the DCTL (DEFINE_LUT / APPLY_LUT;
  0.01-0.02 ΔE00 mean from the exact transform on the test frames). With `--split`, the first part decodes and the
  last encodes. Checked in the grading application (21.1): the wrapped DCTL with its LUT compiles. The gcc harness
  applies DEFINE_LUT tetrahedral or trilinear, as the application's project setting does.
- Tracking (`cierra track look.cierra clip.mov --node NAME --from F0 --to F1 [--at F] [--step K]`): a node's
  window, box or graduated mask follows the picture. The patch under the mask at the reference frame is followed
  forwards and backwards (a frame-to-frame correlation for the coarse shift, then Gauss-Newton against the reference
  patch under a similarity warp with gain and offset, so long tracks do not drift) and written as
  `keys(linear, ...)` on the centre, and on the size and angle unless `--motion translate`, thinned to the keys
  needed to stay within 0.5 px. A lost track (match below 0.6: one line naming the frame) or one needing more than
  64 keys per value is an error, and the program is left as it was. No new language: the keys are §8's. Frames are decoded
  in one FFmpeg pass (`video.read_luma`), not one decode per frame.
- Temporal noise reduction: `temporal_denoise(strength, frames, noise, patch_px, align)` with `requires frames`
  (new host data: the neighbouring frames). Temporal non-local means: each neighbour is weighted per pixel by how alike
  a patch round the pixel is, so still areas average over up to nine frames and moving areas keep the current frame;
  `align` undoes a whole-frame shift (phase correlation, whole pixels) first. The neighbours run through the graph up
  to the node, so they are compared after the same grade. `render` decodes the neighbours from the clip in one pass;
  a still is refused in one line. LUT / DCTL / WGSL / CUDA refuse it ("temporal"). On synthetic noise (sigma 0.02 cv,
  five still frames) the noise std falls 2.24x; a box moving 6 px a frame leaves a local error of 0.0045 cv where a
  plain average leaves 0.09.
- Video input: `render`, `fit` and `loop` read one frame of a video file (MOV / MP4 / MXF / MKV ..., any codec the
  system FFmpeg decodes, called as a program) with `--frame N`, frame-accurate (frame N by number, not a time seek).
  Decoded at the stream's own bit depth; the code range defaults to the stream's tag (untagged YUV: video) and the YUV
  matrix follows the tag (untagged: BT.709 from 720 lines, BT.601 below), applied once, in float, sub-blacks and
  super-whites kept. A video's codes are the program's own input, as a `.npy`'s are.
- Display-referred LUTs import (`import-lut --input rec709_display` or `srgb_display`): Rec.709 video LUTs, display
  codes in and out, come back as a program with a photo input and `display_encode` output (no tone scale). New input
  `rec1886_photo`: display-referred Rec.709 video decoded with gamma 2.4.
- TIFF input: 8-bit, 16-bit and float TIFFs load at full depth (tifffile, in the `images` extra).
- 16-bit PNG: grey PNGs load at 16 bits; colour PNGs, which Pillow reads as 8-bit, are refused with one line (save
  them as 16-bit TIFF) instead of being quantised silently.
- `fit` with a reference of another size says so in one line instead of crashing; `fit` prints the input and code
  range it read the image with, and every loop history entry records them (`source`).
- The open-weight driver: an explicit `temperature=0` is no longer overridden by `CIERRA_TEMPERATURE`.
- The grading loop keeps a model's values inside the tuner's bounds before measuring (`CIERRA_BOUND=0` turns it off);
  values it pulled in are recorded as "capped" and told to the model.
- Fixes from a code review (2026-11-04), each with a test in tests/test_review_fixes.py:
  - `import-lut` fits black-and-white LUTs (saturation 0: was 27.7 ΔE00 mean, now exact) and LUTs that remove colour
    at mid grey only (was NaN, 11.2 ΔE00 mean; now 0.37); a NaN domain is refused; NaN gains are no longer accepted;
  - the host check returns "unverified" for a still with NaN pixels (NaN passed every limit), takes timecodes of
    timelines above 24 fps, and refuses rooted LUT paths and minutes or seconds of 60 or more;
  - the open-weight driver follows a redirect only to an address `CIERRA_LLM_URL` itself may use, and an error inside
    a provider call is no longer retried without its arguments (a paid call twice);
  - the damage gate and the PNG writer handle NaN pixels; the tuner counts a NaN measurement as unmet, not met;
  - `track` writes the program through a temporary file, so a failed write never leaves it half written;
  - smaller: `.NPY` in capitals, one-line errors for odd TIFF shapes, unpaired surrogates and `--step 0`, the parser
    refuses an oversized source before copying it, dated model ids are priced as their model, history diffs show
    `version` / `requires` and ignore a written default, the output LUT name in a wrapped DCTL holds only letters,
    digits, `.`, `_` and `-`, `fit_sequence` refuses shot lists of the wrong length.

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
