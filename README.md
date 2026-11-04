# Cierra

**Colour grades as code.** Cierra is a small language for colour grading. A grade is a short text program of nodes,
masks and a graph that you can read, check, diff, review and keep in version control. One reference engine renders
it, and compilers turn it into the formats grading tools already load: DCTL, 3D LUTs (.cube and CLF), ASC CDL, CUDA
and WebGPU (WGSL). Every compiler is tested against the engine, and a target that cannot hold an op refuses the
program and says why instead of approximating it. Cierra is also built so a language model can grade from notes:
the model writes a program, never touches pixels, and Cierra checks, renders and measures what it wrote.

![An ungraded photo, the graded photo, and the Cierra program that grades it](docs/img/readme_pipeline.png)

The program under the pictures is [examples/bw_red_filter.cierra](examples/bw_red_filter.cierra) (its comment and
label lines left out): one monochrome channel mixed mostly from red, so a blue sky goes dark, then a hard print curve,
darkroom clarity, the sky burned in, grain and a vignette.

## Quick start

You need Python 3.12 or later. `make` is optional for this example; the test, build, local-model and pipeline
commands further down use it. A C / C++ compiler (gcc, g++) is optional too: without it the DCTL / CUDA parity tests
skip.

```
git clone https://github.com/EzulFilm/cierra && cd cierra
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[images]"        # Pillow and tifffile, for PNG / JPEG / TIFF files
```

(`make setup` does the same with pinned dependencies; on Debian / Ubuntu `venv` is the `python3-venv` package.) Run
the commands below from the repository root, with the environment active (`. .venv/bin/activate` in each new shell).
`cierra --help` and `cierra COMMAND --help` list every option.

Save this as `hello.cierra` in the repository root:

```
cierra 0.1
name "hello"
input linear_rec709
output aces2(display: srgb)

node warm = printer_lights(red: 2, blue: -2)
node pop = cdl(slope: [1.05, 1.05, 1.05], sat: 1.1)
node crisp = sharpen(amount: 0.4)

graph warm -> pop -> crisp
```

Then:

```
cierra check hello.cierra                                     # check it; prints the canonical form
mkdir -p out                                                  # render does not create the output folder
cierra render hello.cierra tests/data/frames/pens.npy out/pens.png   # render a test frame with the reference engine
cierra dctl hello.cierra out/hello                            # -> out/hello/hello.dctl
cierra lut hello.cierra out/hello                             # refused: sharpen() is spatial, a LUT cannot hold it
```

The checker reports every problem at once, with line and column. A target that cannot hold a node exits non-zero
with one line naming the node and the reason, as `lut` does here. Delete the `node crisp = ...` line and change the
graph to `graph warm -> pop`, and the program compiles to a LUT (`cierra lut hello.cierra out/hello_lut`) and folds
into one ASC CDL (`cierra cdl hello.cierra out/hello_cdl`).

Compiling writes files under the output folder you give; it does not install anything into a grading application.
Load the `.dctl`, `.cube` / `.clf` or CDL file in your tool, or see [Connect to a grading
application](#connect-to-a-grading-application).

To run the tests: `make test` (the whole suite, a few minutes; no network, GPU, API key or model needed) or
`make test-fast` (the language, checker and engine, seconds). `make build` compiles every look in `looks/` to each
target that can hold it, into `out/build/`.

## Examples

Each column is one short program from [examples/](examples/): [red-filter B&W](examples/bw_red_filter.cierra),
[sky drama](examples/sky_drama.cierra), [70s print](examples/warm_70s_print.cierra),
[Orton](examples/orton_landscape.cierra) and [cross-process](examples/cross_process_video.cierra).

![Four photos, ungraded and through five grades written in Cierra](docs/img/readme_looks.png)

[examples/](examples/) has 22 commented programs for common jobs (beauty skin, bleach bypass, day for night, an HDR
deliverable, a tungsten fix, time of day, ...), and [looks/](looks/) three fuller looks. Compiling some of them
(output copied from real runs):

```
cierra dctl looks/cinematic_modern.cierra out/dctl
  wrote out/dctl/cinematic_modern.dctl - whole graph
cierra dctl examples/sky_grad.cierra out/dctl709 --in sony_slog3_sgamut3
  wrote out/dctl709/sky_grad.dctl - whole graph
  wrote out/dctl709/sky_grad_out_aces2_srgb.cube - output LUT; keep it next to sky_grad.dctl
cierra lut looks/cinematic_modern.cierra out/lut
  out/lut/cinematic_modern_ACEScct.cube, cinematic_modern_ACEScct.clf  (ACEScct in and out)
  out/lut/cinematic_modern_ACEScct_to_aces2_srgb.cube                  (graph + output transform, display out)
cierra dctl examples/bw_red_filter.cierra out/split --split
  wrote out/split/bw_red_filter_part1.dctl - nodes red_filter, print
  NOT IN DCTL: not in DCTL: node clarity: local_contrast() is spatial
  wrote out/split/bw_red_filter_part3.dctl - nodes sky_burn, film_grain, edges
```

The README figures come from four CC0 camera RAWs (raw.pixls.us): `python tools/make_readme_figures.py` downloads
them, pinned by SHA-256, and needs rawpy.

## What you can do

### Write and check a grade

- 33 ops: exposure, white balance, printer lights, lift / gamma / gain wheels, CDL, contrast, curve, hue-vs-hue /
  sat / lum, sat-vs-sat, sat-vs-lum, lum-vs-sat, saturation, vibrance, hue rotate, RGB mixer, density (subtractive
  saturation), split tone, gamut compression, photo-to-scene; vignette and grain; and the spatial ops blur, sharpen,
  glow, halation, diffusion, local contrast, guided detail, highlights / shadows, noise reduction and temporal noise
  reduction.
- 10 masks: HSL qualifier, RGB key, luma, skin, window (ellipse), box, polygon, graduated, depth, segment; combined
  with `and` / `or` / `not`, each with blur, edge-aware feather and matte clip black / white. Named keys can be reused
  and mixed (`key sky = graduated(...)`, `mask not key(face)`), as in a node graph.
- Keyframes, parallel branches, layer blending (normal, add, multiply, screen, overlay, lighten, darken, subtract,
  difference).
- Types, units and ranges for every parameter, and limits on size, nesting and references, so a program that passes
  the checker is safe to render, whoever wrote it. `cierra check` prints the canonical form, which keeps diffs small.

The language is defined in [docs/spec.md](docs/spec.md); [docs/authoring.md](docs/authoring.md) lists every op with
its parameters, ranges and defaults.

### Render

`cierra render PROGRAM IMAGE OUT.png` (or `.tif`) runs the reference engine: NumPy in float64 on the CPU by default,
or the same code on an NVIDIA GPU with `--device gpu` (below). Inputs:

- `.npy` arrays, PNG and JPEG (8-bit; also 16-bit grey PNG; a 16-bit colour PNG is refused: save it as 16-bit TIFF)
  and TIFF (8-bit, 16-bit, float);
- one frame of a video file (MOV, MP4, MXF, MKV, ...: anything the system `ffmpeg` decodes) with `--frame N`,
  frame-accurate by number. FFmpeg and ffprobe must be on PATH. The code range follows the stream's tag and YUV is
  turned to RGB with the stream's matrix;
- camera log codes (ARRI LogC3 / LogC4, Sony S-Log3, Panasonic V-Log, Canon Log, RED Log3G10, Blackmagic Film Gen 5,
  Apple Log, DJI D-Log), display-referred photos and scene-linear images, as the program's `input` line declares.

Image files are read as full range by default. `--range video` reads a legal-range (16..235) PNG or TIFF so those
codes become 0 and 1, keeping values outside them instead of clipping; it is refused on float images, which have no
codes. Outputs go through ACES 2.0 (OpenColorIO) for SDR Rec.709 / sRGB, P3 and HDR (PQ 1000 to 4000 nits, HLG).

On an NVIDIA GPU, `pip install -e ".[cuda]"` adds CuPy and `cierra --device gpu render ...` (or `CIERRA_DEVICE=gpu`)
runs the same float64 code there; the tests compare it with the CPU (within 1e-9 in ACEScct). A 1920x1080 S-Log3
frame through an 11-node grade with glow and guided detail takes 7.4 s on the CPU and 0.7 s on an RTX 3070 Laptop
GPU. The camera input and ACES output transforms stay on the CPU (`CIERRA_OCIO_THREADS`, default 4). CuPy brings
NVIDIA's CUDA runtime libraries from PyPI, under NVIDIA's licence; none of it ships with Cierra.

### Work with moving pictures

- **Tracking.** `cierra track look.cierra clip.mov --node face --from 0 --to 120` makes a node's window, box or
  graduated mask follow the picture. It follows the patch under the mask (from frame `--at`, default `--from`,
  forwards and backwards) and writes its centre, size and angle (`--motion translate` for position only) as
  `keys(linear, ...)`, thinned to the keys needed to stay within 0.5 px (`--tolerance`). The program is rewritten
  in canonical form (`-o` writes another file). A lost track (match below 0.6, `--min-match`) or one needing more
  than 64 keys stops with one line naming the frame, and nothing is written.
- **Temporal noise reduction.** A program with `requires frames` and a node such as
  `temporal_denoise(strength: 0.8, frames: 2, noise: 0.02)` averages each pixel with the same pixel in neighbouring
  frames where a small patch round it matches, so still areas lose grain (noise std down 2.2x over five still frames
  in the tests) and moving areas keep the current frame instead of ghosting; `align` undoes a pan first. `render`
  reads the neighbours from the clip; on a still image it is refused in one line.
- **Keyframes.** Any value can be keyed over frames (`keys(linear, ...)`, up to 64 keys a value).
- **Camera RAW to a movie.** `pip install -e ".[raw]"` (LibRaw through rawpy; FFmpeg installed separately), then
  `make pipeline-demo` downloads a CC0 RAW still (pinned by SHA-256), runs
  [examples/tungsten_fix.cierra](examples/tungsten_fix.cierra) frame by frame so a keyframed white balance takes the
  tungsten orange out, and writes a ProRes-compatible movie and before / after stills to `out/pipeline/`. With your
  own files: `python tools/raw_to_video.py my_look.cierra shot_*.dng --out shot.mov` (one RAW file per frame), or
  `... still.dng --frames 96 --codec h264` to hold a still and animate the grade. Stills formats LibRaw reads (DNG,
  CR2 / CR3, NEF, ARW, RAF, ...) work; cinema RAW formats need their vendors' SDKs, so convert them to DNG or EXR
  first.

### Compile to a target

All targets work on ACEScct values (the DCTL can also wrap itself for a Rec.709 project, below).

| command | writes | holds | parity with the reference engine |
|---|---|---|---|
| `cierra dctl` | `.dctl` | pointwise and position ops (grain, vignette); masks except depth, segment and blurred or feathered ones; keyframes evaluated at one frame (`--frame`); one blur, sharpen, glow, halation or diffusion node per `.dctl`, with a kernel radius of at most 12 px | example looks, worst frame: max 0.0013 ΔE00 (through a CPU harness) |
| `cierra lut` | `.cube` and CLF v3 (65³ by default, `--size`) | pointwise ops without keyframes | example looks, worst frame: max 0.48, mean ≤ 0.06 ΔE00 (tetrahedral) |
| `cierra cdl` | ASC CDL (`.cc`, `.ccc`, `.cdl`) | a serial graph of `printer_lights`, `contrast`, `cdl` and `saturation`, without masks or keyframes, folded into one slope / offset / power / saturation | within 2e-5 in ACEScct (tests) |
| `cierra cuda` | one `.cu` kernel | as DCTL, without the spatial ops | max 0.0013 ΔE00 (CPU harness with g++; an nvcc runner is included, not yet run on a GPU) |
| `cierra wgsl` | WebGPU compute shader | as CUDA | the same per-op parity tests as DCTL, where wgpu finds an adapter (skipped in CI) |

A ΔE00 of about 1 is roughly the smallest visible difference. Full numbers: [docs/targets.json](docs/targets.json)
(`make targets` regenerates them); exact coverage per target: [docs/spec.md](docs/spec.md), section 11.

- **Refusals.** Anything a target cannot hold is refused with the node and the reason, never approximated. That
  includes every spatial op in LUT, CDL, CUDA and WGSL; in DCTL, local contrast, guided detail, highlights /
  shadows, noise reduction, temporal ops and Gaussian ops wider than 12 px. The 12 px are at the program's
  `reference_width` (default 1920) and scale with the frame, so a 4K frame takes four times the samples. The glow,
  halation and diffusion nodes in the bundled examples are all wider than that, so they are refused.
- **`--split`** cuts the graph around nodes a target cannot hold, and before a second Gaussian node in a DCTL, and
  writes the parts it can (`bw_red_filter_part1.dctl`, `..._part3.dctl`, with the node in between named).
- **DCTL for a Rec.709 project.** `cierra dctl look.cierra out --in KIND` makes a DCTL for a project with no colour
  management. It decodes the input to ACEScct first, from `rec1886_photo` or `srgb_photo` (display-referred video)
  or a camera log (`sony_slog3_sgamut3`, `sony_slog3_sgamut3cine`, `arri_logc3`, `arri_logc4`, `panasonic_vlog`,
  `apple_log`, `dji_dlog`), and encodes the result for the display. `--out` picks the encoding: `program` (the
  program's own `output` line, the default when `--in` is given), `srgb`, `rec1886`, or `acescct` (no encoding, the
  default otherwise). An ACES 2.0 output is written as a 65-point .cube next to the DCTL, which applies it; keep the
  two files together.

### Bring an existing grade in

- `cierra import-lut grade.cube imported.cierra` fits a short, readable program (primary, saturation, hue and luma
  curves, and a keyed node when needed) to a 3D LUT and reports how far it is from the LUT in ΔE00. `--input` names
  the LUT's domain: `acescct` (default), `acescg`, `linear_rec709`, or display-referred `rec709_display` /
  `srgb_display` for ordinary Rec.709 video LUTs. A LUT holds one colour-to-colour function, so blur, glow, grain,
  noise reduction and windows cannot come back from it, and a qualifier comes back only approximately.
- `cierra fit image reference fitted.cierra` fits a program that turns one image into a graded reference of the same
  size.

### Grade from notes

```
cierra loop "warmer faces, a little more contrast, keep the blacks clean" tests/data/frames/crowd.npy history/
```

Each round a driver proposes the next program. Cierra checks it, renders a preview, measures each note against the
ungraded frame (right way, no change or wrong way, and which node caused it) and commits the version to `history/`
with a readable diff. It also reports qualities that moved when no note asked ("also moved, no note asked:
brightness 0.31 -> 0.38"). A damage gate checks every version for clipped highlights, crushed or milky blacks,
posterisation and added banding: a damaged version is kept but marked for a human, it never ends the loop as done,
and a clean version is preferred. With no model configured, the deterministic rules driver gives:

```
+ node skin_warmth added: saturation(amount: 1.08) where skin(width: 30)
+ node contrast_up added: contrast(amount: 1.08, pivot: 0.38)
graph: none  ->  skin_warmth -> contrast_up
```

The model gets the language reference, the notes and numbers measured on the preview, never pixels, and its reply is
checked like any other program; nothing it writes is executed. The driver comes from the environment:

| variable | values |
|---|---|
| `CIERRA_DRIVER` | `auto` (default: a local open-weight server when `CIERRA_LLM_URL` is set, else the Anthropic API when `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set, else the offline rules driver), `openweight`, `anthropic`, `rules` |
| `CIERRA_LLM_URL` | an OpenAI-compatible server (llama.cpp, vLLM, Ollama), e.g. `http://127.0.0.1:8090/v1`; `CIERRA_MODEL` is the served name |
| `CIERRA_GRAMMAR` | default `1`: replies constrained by a grammar generated from the op table on llama.cpp, so names and value shapes are always valid; `0` for servers that reject the field |
| `CIERRA_MODEL`, `CIERRA_EFFORT` | for the Anthropic driver (`pip install -e ".[llm]"`): `claude-sonnet-5-5` (default), `claude-opus-5-5`, `claude-haiku-5-5`; effort `low` (default), `medium`, `high` |
| `CIERRA_LLM_TIMEOUT` | seconds per request to a local server (default 300; raise it for CPU inference) |
| `CIERRA_PROMPT`, `CIERRA_REVISE` | `short` / `full` language reference (open-weight default `short`); `0` turns off the revision turn |
| `CIERRA_BUDGET_USD` | spending cap per run (default 2.00), after which the loop finishes with the rules driver; `CIERRA_GPU_USD_PER_HOUR` prices a local server's time |
| `CIERRA_BOUND` | default `1`: a model's values are kept inside the tuner's bounds, and the capped values are recorded and told to the model |

Keys come only from the environment through the official SDK, never from code, files or logs. Every round records
model, tokens, time and estimated cost. `loop` also takes `--frame` for a video file and `--range`.

To run a model on your own machine:

```
make llm-setup            # needs git, cmake, a C++ compiler, curl. llama.cpp (pinned, CPU build) + Qwen2.5-7B-Instruct Q4_K_M, SHA-256 checked (~4.7 GB)
make llm-start            # serves it on 127.0.0.1:8090
make eval-llm             # the held-out evaluations with it (CPU: tens of minutes)
make llm-stop
```

`MODEL=1.5b` for the smaller model. Cierra ships no weights; both models are Apache-2.0, and llama.cpp is MIT.
`make eval` scores the rules driver and tuner on two held-out sets of notes × frames, and `tools/eval_sheet.py` puts
runs side by side for judging by eye, because meeting the checks is not the same as a good grade.

### Connect to a grading application

Cierra talks to a grading application through its scripting API with a few fixed scripts that take their data as
JSON (`cierra/hosts/grading_app.py`). Nothing a model writes is run in the application.

- **From the application.** The API can read each node's label, the tools it uses and the LUT or DCTL on it, and
  export the clip's whole grade as a 17, 33 or 65-point .cube. It cannot read wheel, curve, window or qualifier
  values, so Cierra reads the grade back from that LUT with `import-lut` (above). Over the test frames, programs
  fitted to 33-point LUTs of the example looks are 0.06 ΔE00 from the LUT on average for Cinematic Modern and 0.36
  for Beauty Modern, whose skin qualifier the fit can only approximate.
- **To the application.** A program goes on the clip node for node: each Cierra node becomes a CDL when it is one,
  otherwise its own DCTL, in a new colour version that is rolled back if any step fails. The API cannot create
  nodes, so the clip must already have enough serial nodes (the script says so if not), and spatial nodes the DCTL
  cannot hold are refused with the reason. The script then exports the clip's LUT and Cierra checks it against the
  program.

## Limits and status

- Version 0.1 of the language, before a 1.0: the spec may still change, and the CHANGELOG says how.
- Most spatial and all temporal work renders in the reference engine only; see the refusals under
  [Compile to a target](#compile-to-a-target).
- The CUDA kernel has been tested through a CPU harness, not yet on a GPU. WGSL tests skip without a GPU adapter.
- A LUT cannot carry spatial effects, windows or exact qualifiers, so importing a grade from a LUT is a fit, with its
  error reported.
- The damage checks are simple measures set on the test frames, not validated against viewers; they veto damage and
  never rank looks.
- Video input needs FFmpeg on PATH; cinema RAW formats need conversion first.

## Repository

| path | what |
|---|---|
| `cierra/` | parser, checker, formatter, reference engine, compilers (`compile/`), grading loop, note checks, damage checks, tuner, fitting, tracking, video input, host adapter |
| `examples/`, `looks/` | commented example programs and fuller looks |
| `docs/` | [specification](docs/spec.md), [conformance table](docs/conformance.md) (each spec section and the tests that hold it), [authoring profile](docs/authoring.md), target report, SBOM |
| `tests/` | the suite; `tests/data` test frames (see its SOURCES.md), `tests/golden` reference outputs |
| `tools/` | evaluation, golden, documentation and SBOM tools; RAW-to-movie; local model scripts |

What changed and when: [CHANGELOG.md](CHANGELOG.md). Contributions are welcome: see
[CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

## Licence

Copyright 2026 CIERRAEZUL LLC. Apache License 2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). "Cierra" is a name used
by CIERRAEZUL LLC and is not licensed under the Apache License. The test frames and README figures are made from CC0
camera RAW photos (raw.pixls.us) and a synthetic chart (tests/data/SOURCES.md).
