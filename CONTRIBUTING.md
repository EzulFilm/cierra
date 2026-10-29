# Contributing to Cierra

Thank you for helping. Cierra is small on purpose: a closed language, a checker that refuses what it cannot
guarantee, one reference engine, and compilers that are tested against it.

## Before a change

- Open an issue first for anything larger than a fix, especially a new op, mask or target: each one changes the
  specification (docs/spec.md), the op table (cierra/ops.py), the engine, every compiler and the conformance table.
- Contributions are accepted under the Apache License 2.0. Sign off every commit (`git commit -s`) to certify the
  Developer Certificate of Origin (https://developercertificate.org/): that you wrote the change or have the right to
  submit it under this licence.

## The rules a change must keep

- **The reference engine is the source of truth.** A target (DCTL, WGSL, LUT, CDL) that cannot express an op refuses
  it with a reason; it never approximates silently. New targets come with parity tests against the engine.
- **The op table is the single source.** The checker, formatter, LLM prompt, grammar, authoring profile and spec
  tables are generated from or checked against `cierra/ops.py`. After changing it, run `make docs` and commit the result.
- **Goldens change only on purpose.** If an engine or look change moves the goldens, regenerate them with
  `python tools/make_golden.py` and say why in the commit.
- **No pixels to the model, no code from the model.** The LLM writes Cierra programs, which are parsed and checked
  like any other input; nothing it writes is executed.
- **Tests need no network, GPU, key or model.** Use the rules driver or the scripted provider in tests.

## Checks

```
make test        # the whole suite
make docs        # regenerate the generated docs (CI fails if they are stale)
make sbom        # dependency licences: no GPL / AGPL / non-commercial component in the library
```

Keep to the existing style: plain names, short docstrings that say what and why, comments only where the code
cannot say it.
