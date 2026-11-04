# Cierra

Cierra is a small language for colour grades. A grade is a plain text file that lists nodes and masks and how they
connect, so it can live in git next to everything else. Cierra checks the file, renders it, and compiles it to DCTL,
3D LUTs (.cube and CLF), ASC CDL, a CUDA kernel or a WebGPU shader.

I started it because I wanted my grading notes to end up as something I could check and keep, not as settings
inside one application's project. The NumPy renderer is the reference: every compiler is tested against it, and
when a target can't hold an op, the compiler refuses and names the node rather than approximating it.

The other half is grading from written notes with a language model. The model returns Cierra source and gets
measurements back, never pixels, and its program is checked like anyone else's.

![An ungraded photo, the graded photo, and the Cierra program that grades it](docs/img/readme_pipeline.png)

The program under the pictures is [examples/bw_red_filter.cierra](examples/bw_red_filter.cierra), with its comments
left out. It makes a black and white image mostly from the red channel, so the blue sky goes dark, then adds a hard
print curve, some clarity, a burned-in sky, grain and a vignette.

## Quick start

You need Python 3.12 or later. `make` and a C/C++ compiler are optional; without gcc and g++ the DCTL and CUDA
parity tests skip.

```
git clone https://github.com/EzulFilm/cierra && cd cierra
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[images]"        # Pillow and tifffile, for PNG / JPEG / TIFF
```

On Debian and Ubuntu, `venv` comes from the `python3-venv` package. `make setup` does the same install with pinned
versions. Run everything from the repository root with the environment active (`. .venv/bin/activate` in each new
shell); `cierra --help` and `cierra COMMAND --help` list the options. Save this as `hello.cierra`:

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
cierra check hello.cierra                                     # prints the canonical form
mkdir -p out                                                  # render won't create the folder
cierra render hello.cierra tests/data/frames/pens.npy out/pens.png
cierra dctl hello.cierra out/hello                            # -> out/hello/hello.dctl
cierra lut hello.cierra out/hello                             # refused
```

The last one fails on purpose. `sharpen()` looks at neighbouring pixels and a LUT can't, so `lut` exits with an error
naming the node. Delete the `crisp` line, change the graph to `warm -> pop`, and the same file compiles to a LUT with
`cierra lut hello.cierra out/hello_lut` and folds into one CDL with `cierra cdl hello.cierra out/hello_cdl`.

When the checker finds problems it reports all of them, with line and column. Nothing is installed into a grading
application; you load the `.dctl`, `.cube`/`.clf` or CDL file yourself.

`make test` runs the whole suite in a few minutes, with no network, GPU, API key or model. `make test-fast` covers
the language, checker and engine in seconds. `make build` compiles every look in `looks/` to each target that can
hold it, into `out/build/`.

## Examples

![Four photos, ungraded and through five grades written in Cierra](docs/img/readme_looks.png)

The columns are [red-filter B&W](examples/bw_red_filter.cierra), [sky drama](examples/sky_drama.cierra),
[70s print](examples/warm_70s_print.cierra), [Orton](examples/orton_landscape.cierra) and
[cross-process](examples/cross_process_video.cierra). [examples/](examples/) has 22 commented programs for ordinary
jobs (skin, bleach bypass, day for night, an HDR deliverable, a tungsten fix, ...) and [looks/](looks/) has three
longer ones. The photos are CC0 camera RAWs from raw.pixls.us; `python tools/make_readme_figures.py` downloads them
(checked by SHA-256) and rebuilds the figures. It needs rawpy.

## The language

[docs/authoring.md](docs/authoring.md) lists the 33 ops and 10 masks with their parameters and defaults, and
[docs/spec.md](docs/spec.md) is the specification. Most of the ops are what you'd find on a grading panel (wheels,
CDL, curves, hue curves, mixers); some are spatial, like blur, glow, halation, local contrast and noise reduction.
Masks combine with `and`, `or` and `not`, values can be keyframed, and branches can be blended back together.

The checker enforces every parameter's unit and range and caps the size of a program. That's what makes it safe to
render a model's output without trusting the model.

## Rendering

`cierra render PROGRAM IMAGE OUT.png` (or `.tif`) runs the reference engine, in float64 NumPy on the CPU. With an
NVIDIA GPU, `pip install -e ".[cuda]"` and `cierra --device gpu render ...` run the same code there, and the tests
compare the two.

It reads `.npy`, PNG, JPEG and TIFF, and single frames of video with `--frame N` (ffmpeg and ffprobe have to be on
PATH). Image files are read as full range unless you pass `--range video`. The program's
`input` line says what the pixels are: a camera log encoding (LogC, S-Log3, V-Log, Apple Log and others), a
display-referred photo, or scene-linear. Output goes through ACES 2.0 for SDR, P3 or HDR (PQ and HLG).

`cierra track look.cierra clip.mov --node face --from 0 --to 120` makes the window, box or graduated mask on node
`face` follow the picture, and writes the motion into the program as keyframes. If the track is lost, it stops and
names the frame, and nothing is written.

`temporal_denoise` needs `requires frames` in the program and a clip to render from; it averages each pixel with
neighbouring frames where they match. Like the other temporal work, it only runs in the reference engine.

To grade camera RAW stills into a movie, `pip install -e ".[raw]"` and run
`python tools/raw_to_video.py my_look.cierra shot_*.dng --out shot.mov` (one RAW file per frame; ffmpeg needed).
`make pipeline-demo` does this with a CC0 example. Cinema RAW formats need converting to DNG or EXR first.

## Compiling

All targets work on ACEScct values.

- `cierra dctl`: pointwise ops, grain, vignette, masks other than depth, segment and blurred or feathered ones,
  and one blur, sharpen, glow, halation or diffusion node per file with a radius up to 12 px (at the program's
  `reference_width`, default 1920). Keyframes are evaluated at one frame (`--frame`). `--in KIND` makes a DCTL for
  a Rec.709 project without colour management; keep any `.cube` it writes next to the `.dctl`.
- `cierra lut`: `.cube` and CLF, for pointwise ops without keyframes.
- `cierra cdl`: a serial chain of `printer_lights`, `contrast`, `cdl` and `saturation`, without masks or
  keyframes, folded into one CDL.
- `cierra cuda` and `cierra wgsl`: what DCTL holds, minus the spatial ops. The CUDA kernel has only been tested
  through a CPU harness so far, and the WGSL tests skip without a GPU adapter.

If a target can't hold part of the graph, `--split` writes the parts it can and names what's left out:

```
$ cierra dctl examples/bw_red_filter.cierra out/split --split
wrote out/split/bw_red_filter_part1.dctl - nodes red_filter, print
NOT IN DCTL: not in DCTL: node clarity: local_contrast() is spatial
wrote out/split/bw_red_filter_part3.dctl - nodes sky_burn, film_grain, edges
```

The measured differences between each target and the engine are in [docs/targets.json](docs/targets.json)
(`make targets` regenerates it), and section 11 of the spec says exactly what each target covers.

`cierra import-lut grade.cube imported.cierra` fits a program to an existing 3D LUT and prints how far it is from
the LUT in ΔE00 (`--input` says what the LUT expects, if it isn't ACEScct). A LUT only maps colours, so blur, grain
and windows can't come back out of it. `cierra fit image reference fitted.cierra` does the same from a pair of
images, an ungraded one and a graded reference of the same size.

## Grading from notes

```
$ cierra loop "warmer faces, a little more contrast, keep the blacks clean" tests/data/frames/crowd.npy history/
iteration 1: v1 4c311f3abb26 notes met 4/4
iteration 1: every note is met; stopping
```

The loop writes each candidate and its diff to `history/`, with how each note moved against the ungraded frame and
anything that moved when no note asked for it. A candidate with clipped highlights, crushed or milky blacks or added
banding can't finish the run. These checks are simple measures tuned on the test frames, not validated with viewers.

With nothing configured, the driver is a fixed set of rules, which is what produced the run above. Set
`CIERRA_LLM_URL` to an OpenAI-compatible local server, or `ANTHROPIC_API_KEY` for the Anthropic API (after
`pip install -e ".[llm]"`), and `CIERRA_DRIVER` to force one. When a run reaches `CIERRA_BUDGET_USD` (default
2.00), it finishes with the rules driver. The model choice and the other settings are described at the top of
[cierra/llm.py](cierra/llm.py). Nothing the model writes is executed, and keys are read only from the environment.
`loop` takes `--frame` and `--range` like `render`.

To run a model locally (a pinned llama.cpp CPU build and Qwen2.5-7B-Instruct Q4_K_M, about 4.7 GB, SHA-256
checked):

```
make llm-setup      # needs git, cmake, a C++ compiler and curl
make llm-start      # serves on 127.0.0.1:8090
make eval-llm       # the held-out evaluations; tens of minutes on a CPU
make llm-stop
```

Cierra doesn't ship any weights. `make eval` scores the rules driver on held-out notes. I still compare
runs by eye with `tools/eval_sheet.py`; the checks don't judge the look.

`cierra/hosts/grading_app.py` connects to a grading application through its scripting API with a few fixed scripts
that take JSON. It can put a program on a clip node for node, in a new colour version that's rolled back if
a step fails, and read a grade back as a LUT for `import-lut`. The API can't create nodes, so the clip needs enough
serial nodes already, and it can't read wheel, curve or qualifier values, which is why the grade comes back as a
LUT.

## Status

This is version 0.1 of the language, and the spec may still change before 1.0. [CHANGELOG.md](CHANGELOG.md) records
what changed. Contributions are welcome, see [CONTRIBUTING.md](CONTRIBUTING.md); security reports go to
[SECURITY.md](SECURITY.md).

## Licence

Copyright 2026 CIERRAEZUL LLC. Apache License 2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE). "Cierra" is a name used
by CIERRAEZUL LLC and is not licensed under the Apache License. The test frames and README figures are made from CC0
camera RAW photos (raw.pixls.us) and a synthetic chart (tests/data/SOURCES.md).
