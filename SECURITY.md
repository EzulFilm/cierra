# Security

Cierra parses and checks programs that may come from a language model or from other people. The checker is a
security boundary: it bounds program size, nesting, node and reference counts, number formats and every parameter
range (docs/spec.md, "Checker guarantees"). The host adapter only runs fixed script templates with data passed as
JSON, never code written by a model.

Please report a way around these limits privately instead of opening a public issue. Examples: a program that passes
the checker but crashes or hangs the engine, emits code a target compiler did not intend, or escapes the host
adapter's templates. Use GitHub's "Report a vulnerability" (private advisory) on this repository. We will reply and
agree a disclosure date with you.
