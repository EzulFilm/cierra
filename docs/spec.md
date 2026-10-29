# Cierra 0.1 language specification

Status: Cierra 0.1, the first public version. **This document is normative**: it defines what a program means.

Compatibility: every program starts with `cierra 0.1`, and a checker refuses any other version. Cierra is young: 0.1
keeps growing in place (new ops, masks, parameters and outputs are added without a version change), and there is no
compatibility promise before 1.0. A change that alters how an existing program renders is recorded in the
CHANGELOG with the goldens regenerated. A compiled target lists the ops it can express (ops_manifest.json) and
refuses the rest with a reason. The reference engine
(`cierra/engine.py`) implements it and the conformance tests (`tests/`) check that the engine and every compiled target
follow it; where they disagree with this text, the code is wrong, or this text is changed deliberately: before 1.0
such a change happens in place, without a version bump, and is recorded in the CHANGELOG with the goldens
regenerated. The author-facing profile with examples is [authoring.md](authoring.md); the machine-readable op
table is [ops_manifest.json](ops_manifest.json) (both generated from the same table as the checker).

## 1. Model

A Cierra program describes one grade: an input colour space, a set of named **nodes** (one operation each, with an
optional mask and mix), a **graph** that wires the nodes, and an **output** transform. A program has no variables, no
arithmetic, no loops, no functions, no I/O and no randomness without a seed. It is data with a grammar.

Pipeline for one frame `f` of an image `I`:

1. `W0 = input_to_working(I, kind)` — ACEScct code values in AP1, shape (3, H, W). `kind` is the program's `input`
   unless the host renders with another input kind (a look written for scene-linear footage applied to a photo).
2. If `kind` is a photo kind (`srgb_photo` or `linear_photo`) and the program has a `photo_graph`:
   `W0 = photo_graph(W0)`. For any other kind the `photo_graph` does not run (the linter says so when the host tells
   it the input kind).
3. `W1 = graph(W0)`; a program without a `graph` line uses the identity (an output-only program).
4. `out = output(W1)`.

All maths is per frame; keyframed values are evaluated at `f` before the frame is processed.

## 2. Grammar (EBNF)

```
program     = ws* "cierra" NUMBER NL { statement | NL } EOF
statement   = meta | node | modifier | graphdecl | keydecl
meta        = "name" STRING | "about" STRING | "input" IDENT | "output" call
            | "reference_width" INT | "fps" NUMBER | "seed" INT
            | "requires" IDENT { "," IDENT }
node        = "node" IDENT "=" call { modifier_inline } NL
modifier    = ( "mask" mask | "mix" value | "label" STRING | "off" ) { modifier_inline } NL
modifier_inline = "mask" mask | "mix" value | "label" STRING | "off"
graphdecl   = ( "graph" | "photo_graph" ) [ "=" ] gexpr NL
keydecl     = "key" IDENT "=" mask NL
call        = IDENT [ "(" [ arg { "," arg } [ "," ] ] ")" ]
arg         = IDENT ":" value
value       = NUMBER [ ".." NUMBER ] | STRING | "true" | "false" | IDENT
            | "[" [ value { "," value } [ "," ] ] "]"
            | "keys" "(" IDENT { "," INT ":" value } [ "," ] ")"
mask        = mand { "or" mand } | mand { "+" mand } | mand { "-" mand }    (one kind per level: mixing needs ( ))
mand        = munary { "and" munary }
munary      = "not" munary | "(" mask ")" | keyref | call
keyref      = "key" "(" IDENT { "," ( "gain" | "offset" ) ":" value } ")"
gexpr       = gterm { "->" gterm }
gterm       = IDENT | "pass" | "(" gexpr ")"
            | "parallel" "(" gexpr { "," gexpr } ")"
            | "layer" "(" gexpr "," gexpr { "," ("mode" ":" IDENT | "opacity" ":" value) } ")"
```

Lexical rules: `NUMBER = -?\d+(\.\d+)?([eE][+-]?\d+)?` (no leading `.`, no `inf`/`nan`); `INT` is a NUMBER without
`.`/exponent; `IDENT = [A-Za-z_][A-Za-z0-9_]*` (≤ 40 characters); `STRING` is double-quoted, one line, escapes `\"`
and `\\` only, ≤ 400 characters. `#` starts a comment to end of line. Newlines end statements except inside `( )` and
`[ ]`. A modifier line (`mask`, `mix`, `label`, `off`) belongs to the node statement directly above it. Reserved
words that cannot name a node or key: `pass parallel layer keys true false not and or`. In a mask, `key(` starts a
key reference (§6.1); `key` at the start of a line starts a key statement.

Uniqueness: each program-level statement (`name`, `about`, `input`, `output`, `reference_width`, `fps`, `seed`,
`requires`, `graph`, `photo_graph`) appears at most once; node and key names are unique (one name space); a parameter appears at most once in
a call; each node modifier (`mask`, `mix`, `label`, `off`) appears at most once per node, whether written on the node's
line or on the lines after it (combine masks with `and` / `or`); `layer()` takes `mode` and `opacity` at most once
each; keyframe frames increase strictly. Each of these is an error, not a "last one wins".

Modifier defaults: `mix` num 0..1, default 1, keyframable; `label` empty; a node is on unless `off`. `layer()`:
`mode` one of normal / add / multiply / lighten / darken / screen / overlay / subtract / difference (§7), default
`normal`; `opacity` num 0..1, default 1,
keyframable.

`photo_graph`: `photo_to_scene()` may appear only there, at most once, and only as the whole `photo_graph` or the first
item of its top-level serial chain (`photo_graph s -> a -> b`), never inside `parallel` or `layer`. Unused nodes are
legal and draw a warning.

Canonical form (`cierra.format.format_program`): header, metadata, one `node` per line with parameters in table order
and defaults omitted, modifiers indented on their own lines, then `photo_graph` and `graph`. Numbers print in shortest
round-trip form. The **program hash** is the first 12 hex digits of SHA-256 of the canonical text.

## 3. Types, units, limits

| kind | written | meaning |
|---|---|---|
| num | `0.25` | finite real in [lo, hi]; keyframes allowed |
| int | `7` | integer in [lo, hi] |
| bool | `true` | |
| enum | `luma` | one of the listed identifiers |
| rgb | `[r, g, b]` | three nums; each component may be keyframed |
| pair | `[a, b]` | two nums; each component may be keyframed |
| range | `0.2..0.6` | lo ≤ hi; hue ranges (deg) may wrap (`330..30`) |
| points | `[[x, y], [x, y, w]]` | control points; per-column ranges; sorted where stated |
| ints | `[1, 4]` | list of integers |
| keys | `keys(linear, 0: 1.0, 24: 1.2)` | keyframes, see §8 |

Units: **cv** ACEScct code value (18 % grey = 0.4135884; above the toe one stop = 1/17.52 = 0.05708 cv);
**stops**; **deg** hue angle of the working values (§4); **px** pixels at the program's `reference_width`
(default 1920; the engine scales every px value by `image_width / reference_width`); **m** metres; **x** a
dimensionless factor (1 = unchanged).

Limits (checker; all in `LIMITS`, exported in the manifest): source ≤ 64 KiB; ≤ 64 nodes; ≤ 64 `key` statements; ≤ 16 **spatial passes**
for the whole program (`graph` and `photo_graph` together), counted per use of a node: 1 for a spatial op, plus 1 for
every mask call in its mask expression whose `blur_px` or `feather_px` is > 0 at any frame (`skin(blur_px: 5) or window(blur_px: 10)`
is 2; a keyframed `blur_px` counts if any of its keys is > 0); graph nesting ≤ 8; ≤ 128 node uses in `graph` and
`photo_graph` together; `parallel` ≤ 8 branches; one mask expression ≤ 16 mask calls and nested ≤ 16 deep (`not`,
`and`, `or`, `+`, `-`, parentheses), and once the `key` statements it uses are expanded (§6.1) ≤ 64 mask calls
and nested ≤ 64 deep, counted per key statement without expanding; values nested ≤ 4 levels of `[ ]` / `keys(...)`; number literals ≤ 40 digits and finite;
≤ 64 keyframes per value, frames integers 0..10⁶ strictly increasing; sorted control points at least 1e−4 apart in x
(curves stay well conditioned, and distinct in float32 targets); every number is range-checked, and a number of any
size is reported as a checker problem, never an internal error. Worst case: 128 pointwise node evaluations per pixel
and 16 spatial passes. A Gaussian pass (blur_px, glow, blur, ...) has σ ≤ 600 px at reference width (radius ⌊3σ⌋ ≤
1800 px, so ≤ 3601 taps per pass per axis, scaled by image width / reference_width). A guided pass (guided_detail,
feather_px) counts as one spatial pass but is 6 box means (M(I), M(p), M(I·p), M(I²), M(a), M(b), §6), each separable
and linear in the pixel count whatever the radius, so its cost does not grow with radius_px.

Program-level fields: `input` (default `linear_rec709`), `output` (default `aces2(display: srgb)`),
`reference_width` 64..16384, `fps` 1..240 (informational in v0), `seed` 0..2³¹−1 (informational in v0; `grain`
takes its own required seed), `requires` ⊆ {`depth`, `segments`}.

## 4. Colour

**ACEScct** (Academy S-2016-001): `enc(x) = 10.5402377416545·x + 0.0729055341958355` for x ≤ 0.0078125, else
`(log2 x + 9.72) / 17.52`; `dec(y) = (y − 0.0729055341958355) / 10.5402377416545` for y ≤ 0.155251141552511, else
`2^(17.52·y − 9.72)`.

**Gamut matrices** are built from primaries and white points (Rec.709 D65; AP1 and AP0 with the ACES white
(0.32168, 0.33767)); different whites are adapted with Bradford. AP0→AP1 needs no adaptation.

**Inputs**: `linear_rec709` W = enc(max(M709→AP1·I, 0)); `aces2065_1` W = enc(max(MAP0→AP1·I, 0)); `acescg`
W = enc(max(I, 0)); `acescct` W = I; `srgb_photo` W = enc(max(M709→AP1·sRGB⁻¹(clip(I, 0, 1)), 0)) (code values,
so never over display white); `linear_photo` W = enc(max(M709→AP1·max(I, 0), 0)) with I display-linear Rec.709
floats that may exceed 1 where a highlight-recovery stage restored them. For both photo kinds the program's
`photo_graph` runs first. Camera inputs: W = the OpenColorIO ColorSpaceTransform from the studio config's colour space
to "ACEScct" (`arri_logc3` "ARRI LogC3 (EI800)", `arri_logc4` "ARRI LogC4", `sony_slog3_sgamut3` "S-Log3 S-Gamut3",
`sony_slog3_sgamut3cine` "S-Log3 S-Gamut3.Cine", `panasonic_vlog` "V-Log V-Gamut", `canon_log2` / `canon_log3`
"CanonLog2 / CanonLog3 CinemaGamut D55", `red_log3g10` "Log3G10 REDWideGamutRGB", `bmd_film_gen5` "BMDFilm WideGamut
Gen5", `apple_log` "Apple Log", `dji_dlog` "D-Log D-Gamut", `linear_rec2020` "Linear Rec.2020", `linear_p3d65`
"Linear P3-D65").

**Opponent space** of working values: `Y = 0.2126 R + 0.7152 G + 0.0722 B`, `Cb = (B − Y)/1.8556`,
`Cr = (R − Y)/1.5748`, chroma `C = √(Cb² + Cr²)`, hue `h = atan2(Cr, Cb)` in degrees mod 360. Inverse:
`R = Y + 1.5748 Cr`, `B = Y + 1.8556 Cb`, `G = (Y − 0.2126 R − 0.0722 B)/0.7152`. Landmark hues: red ~123, skin
~141, orange ~150, yellow ~173, green ~206, cyan ~300, blue ~350, magenta ~60.

Helpers: `smoothstep(e0, e1, x) = t²(3 − 2t)`, `t = clip((x − e0)/max(e1 − e0, 1e−9), 0, 1)`;
`band(x, lo, hi, s) = smoothstep(lo − s, lo + s, x)·(1 − smoothstep(hi − s, hi + s, x))`;
`wrap(d) = ((d + 180) mod 360) − 180` (floor modulo).

**Monotone curve** `curve(x; P)`: through sorted points (xᵢ, yᵢ), secants dₖ, tangents m₀ = d₀, mₙ = dₙ₋₁,
mₖ = (dₖ₋₁ + dₖ)/2; then for k = 0..n−2 in order: if dₖ = 0, mₖ = mₖ₊₁ = 0; else a = mₖ/dₖ, b = mₖ₊₁/dₖ, and if
a² + b² > 9 scale both by 3/√(a² + b²). Inside [x₀, xₙ₋₁] cubic Hermite on segment k where x ∈ (xₖ, xₖ₊₁]
(segment 0 includes x₀); outside, straight lines with the end tangents.

**Hue curve** `H(h; P, neutral)`: each point (h₀, v, w = 60) has weight `ω = ½ + ½ cos(π·clip(wrap(h − h₀)/w, −1, 1))`;
`H = neutral + Σ ω(v − neutral) / max(Σ ω, 1)`.

**Gaussian blur** σ (px, after scaling): unchanged if σ < 0.3; else separable kernel of radius r = max(1, ⌊3σ⌋),
weights exp(−½(t/σ)²) normalised, horizontal then vertical, edge pixels replicated.

## 5. Operations

Notation: input W (per channel R, G, B), output O. `S = 1/17.52`. A node's result is
`W + (O − W)·mix·mask(W)` (mask = 1 when absent; mask computed from the node's input).

Pointwise (LUT and DCTL):

| op | parameters (range, default) | maths |
|---|---|---|
| exposure | stops −6..6 (0) | O = enc(dec(W)·2^stops), a linear-light gain (= W + stops·S above the ACEScct toe, W > 0.1553) |
| white_balance | temperature −2..2 (0), tint −2..2 (0) | O = enc(dec(W)·2^[temperature, −tint, −temperature]) per AP1 channel, a linear-light gain (= W + S·[…] above the toe) |
| printer_lights | red, green, blue −24..24 (0) points | O = W + p·0.025/(17.52·log10 2) per channel |
| wheels | lift, gamma, gain, offset rgb −0.3..0.3 (0); master_* −0.3..0.3 (0) cv | y = clip(Y, 0, 1); sh = 1 − smoothstep(0.1, G+0.05, y); mi = exp(−((y − G)/0.18)²); hi = smoothstep(G − 0.05, 0.75, y); O = W + (lift+ml)·sh + (gamma+mg)·mi + (gain+mn)·hi + offset + mo, G = 0.4135884 |
| cdl | slope rgb 0..4 (1), offset rgb −0.5..0.5 (0), power rgb 0.2..4 (1), sat 0..4 (1) | v = W·slope + offset; v = v^power where v ≥ 0 (else unchanged); O = Yᵥ + sat·(v − Yᵥ) with Rec.709 luma weights (ASC CDL, CLF "FwdNoClamp") |
| contrast | amount 0.3..3 (1), pivot 0..1 (0.4136) | O = (W − pivot)·amount + pivot |
| saturation | amount 0..3 (1), protect_skin 0..1 (0) | k = amount; skin = 1 − smoothstep(15, 40, abs(wrap(h − 141))); k += (1 − k)·skin·protect_skin; O = opp⁻¹(Y, k·Cb, k·Cr) |
| curve | points 2..16 [x −0.2..1.5, y −0.2..1.5] sorted; channel luma\|rgb\|red\|green\|blue (luma); space log\|display (log) | luma: O = W + (curve(Y) − Y); rgb: each channel; single channel: that one. display: always through the fixed sRGB display encoding of §9 `display_encode(eotf: srgb)` (soft clip included), whatever the program's output: D = clip(display_encode(W), 0, 1); D′ = the curve applied to D as above (luma uses the same Rec.709 weights on D); O = enc(max(M709→AP1 · sRGB⁻¹(clip(D′, 0, 1)), 0)). The soft clip is not inverted, so colours it compressed stay compressed |
| hue_vs_hue | points 1..12 [hue 0..360, shift −90..90 (, width 5..180)] | θ = H(h; P, 0) in radians; (Cb, Cr) rotated by θ |
| hue_vs_sat | points 1..12 [hue, gain 0..3 (, width)] | k = H(h; P, 1); O = opp⁻¹(Y, k·Cb, k·Cr) |
| hue_vs_lum | points 1..12 [hue, offset −0.3..0.3 (, width)] | O = opp⁻¹(Y + H(h; P, 0)·smoothstep(0, 0.06, C), Cb, Cr) |
| lum_vs_sat | points 2..12 [luma −0.1..1.5, gain 0..3] sorted | k = curve(Y); O = opp⁻¹(Y, k·Cb, k·Cr) |
| sat_vs_sat | points 2..12 [c_in 0..1, c_out 0..1] sorted | k = curve(C)/max(C, 1e−6); O = opp⁻¹(Y, k·Cb, k·Cr) |
| sat_vs_lum | points 2..12 [chroma 0..1, offset −0.3..0.3] sorted | O = opp⁻¹(Y + curve(C), Cb, Cr) |
| rgb_mixer | red, green, blue rgb −2..2 (rows of the identity); preserve_luma bool (false) | O = M·W, rows red / green / blue; preserve_luma: O += Y(W) − Y(O) |
| density | amount −1..2 (0) | O = opp⁻¹(Y − amount·C, Cb, Cr) |
| hue_rotate | degrees −180..180 (0) | (Cb, Cr) rotated by degrees; O = opp⁻¹(Y, Cb′, Cr′) |
| vibrance | amount −1..1 (0) | k = 1 + amount·(1 − smoothstep(0, 0.12, C)); O = opp⁻¹(Y, k·Cb, k·Cr) |
| split_tone | shadow_hue 0..360 (315), shadow_amount 0..0.1 (0), highlight_hue 0..360 (141), highlight_amount 0..0.1 (0), balance −0.2..0.3 (0) cv | p = G + balance; s = (1 − smoothstep(0.15, p, Y))·shadow_amount; t = smoothstep(p, 0.75, Y)·highlight_amount; O = opp⁻¹(Y, Cb + s·cos h_s + t·cos h_h, Cr + s·sin h_s + t·sin h_h) |
| gamut_compress | threshold rgb 0.5..0.99 (0.815, 0.803, 0.880), limit rgb 1.01..2 (1.147, 1.264, 1.312), power 1..3 (1.2) | L = ACEScct⁻¹(W) (linear AP1); a = max(L); d = (a − L)/abs(a) (0 when a = 0); per channel s = (lim − thr)/(((1 − thr)/(lim − thr))^(−power) − 1)^(1/power); d′ = d below thr, else thr + s·x/(1 + x^power)^(1/power), x = (d − thr)/s; O = ACEScct(a − d′·abs(a)). The open ACES reference gamut compression |

Position (DCTL yes, LUT no). Pixel coordinates: x̂ = i/(w − 1), ŷ = j/(h − 1), origin top-left, i, j integer pixel
indices.

| op | parameters | maths |
|---|---|---|
| vignette | amount −0.5..0.5 (0.1), roundness 0..1 (1), softness 0.05..1.2 (0.6) | u = 2x̂ − 1, v = 2ŷ − 1; r = √((u·(w/h)^(1 − roundness))² + v²)/√2; O = W − amount·smoothstep(1 − softness, 1.2, r)·4S |
| grain | amount 0..1 (0.2), size_px 1..8 (1.5), chroma 0..1 (0.3), seed int (required) | see below |

**grain**: hash(k₁..kₙ): h = 0x9E3779B9; for each key k: h ^= k; h ^= h≫16; h *= 0x7FEB352D; h ^= h≫15;
h *= 0x846CA68B; h ^= h≫16 (uint32 wrap-around). Cell noise n(ix, iy, ch) = (Σⱼ₌₀..₃ u₍ⱼ₎ − 2)·√3 with
h₀ = hash(ix, iy, frame, seed, ch), hⱼ₊₁ = hash(hⱼ, j) for j = 0..3 (hash restarts from 0x9E3779B9 each time) and
uⱼ = (hⱼ₊₁ ≫ 8)/2²⁴. Lattice: gx = i/s, gy = j/s with s = max(size_px·scale, 0.001); value noise with smoothstep-weighted
bilinear blending of the four neighbouring cells. mono = noise(ch = 3), n_c = mono + chroma·(noise(c) − mono).
O = W + amount·0.02·n·4y(1 − y), y = clip(Y, 0, 1). Same seed + frame + size selects the same random cells on every target; floating-point grain values follow the
parity rules in §12.

Spatial (reference engine only in v0; σ = radius_px·scale; B(·) = Gaussian blur):

| op | parameters | maths |
|---|---|---|
| glow | threshold 0..1.2 (0.6) cv, radius_px 1..400 (40), strength 0..2 (0.3), warm −0.5..0.5 (0) | L = Rec.709 linear of W; b = max(L − dec(threshold), 0); halo = B(b)·strength·[1 + warm, 1, 1 − warm]; O = enc(max(dec(W) + M709→AP1·halo, 0)) |
| halation | threshold (0.55), radius_px 1..200 (12), strength 0..2 (0.3), tint rgb 0..1 ([1, 0.25, 0.05]) | as glow with b reduced to its Rec.709 luminance, halo = B(b)·strength·tint |
| diffusion | radius_px 1..200 (20), strength 0..1 (0.25) | l = max(dec(W), 0); O = enc(l + strength·max(B(l) − l, 0)) |
| blur | radius_px 0..100 (2) | O = B(W) |
| sharpen | amount 0..2 (0.3), radius_px 0.3..10 (1.5), coring 0..0.05 (0.008) | d = Y − B(Y); O = W + amount·sign(d)·max(abs(d) − coring, 0) |
| local_contrast | amount −1..1 (0.15), radius_px 5..600 (200), protect_blacks (true) | δ = amount·(Y − B(Y)); with protect_blacks δ·smoothstep(0.05, 0.25, Y) where δ < 0; O = W + δ |
| guided_detail | amount −1..2 (0.5), radius_px 1..200 (8), edge 0.001..0.2 cv (0.02), protect_blacks (true) | δ = amount·(Y − G(Y, Y; r, edge²)) with G the guided filter below; protect_blacks as local_contrast; O = W + δ |
| highlights_shadows | highlights −3..3 (0), shadows −3..3 (0), radius_px 1..400 (60) | b = B(Y); O = W + 0.1·(highlights·smoothstep(G + 0.05, 0.75, b) + shadows·(1 − smoothstep(0.15, G, b))) |

Input stage (only in `photo_graph`):

| op | parameters | maths |
|---|---|---|
| photo_to_scene | recovered_highlights (true), knee_stops 1..10 (4.5), max_stops 1.5..12 (6), knee_stops < max_stops at every integer frame (checker, exact: between consecutive keys of either value both are polynomials of degree ≤ 3 in the frame, so their difference is a cubic and is checked at the interval ends and at the integer frames next to its stationary points) | recovered_highlights only has an effect for `linear_photo` input (srgb_photo never exceeds 1); see note N1. L = max(Rec.709 lin, 0); o = max(max_c L, 1); D = sRGB(L/o); F = ACES2⁻¹(D); near white (w = smoothstep(0.70, 0.92, max D)·(1 − smoothstep(0.10, 0.30, sat D))) F blends to the photo's chromaticity at the luminance the inverse gives; with recovered_highlights F += log2(max(min_c L, 1))/17.52; the brightest channel rolls off from G + knee/17.52 to G + max/17.52 with tanh, all channels shifted equally |

## 6. Masks

A mask is a function of the node's input W (and position or a host plane) to [0, 1]. `not m = 1 − m`,
`a and b = a·b`, `a or b = max(a, b)`. Each mask takes `blur_px` 0..200 (0): when > 0 the matte is Gaussian-blurred
(spatial). Each mask also takes `feather_px` 0..200 (0): when > 0 the matte (after any blur) is replaced by
clip(G(clip(Y, 0, 1), matte; feather_px, 10⁻³), 0, 1), so its soft edge follows the picture's luma edges (spatial). Last, each mask takes `clip_black` 0..1 (0) and `clip_white` 0..1 (1): matte = clip((matte − clip_black) /
max(clip_white − clip_black, 10⁻⁶), 0, 1) (pointwise; skipped at the defaults).

Guided filter G(I, p; r, ε) (He, Sun, Tang, ECCV 2010, gray guide): with M the mean over a (2r + 1)² window (r = px
scaled to the image, rounded; edges replicated), a = (M(I·p) − M(I)M(p)) / (M(I²) − M(I)² + ε), b = M(p) − a·M(I),
G = M(a)·I + M(b).

| mask | parameters | value |
|---|---|---|
| qualifier | hue range deg, sat range 0..1, lum range −0.1..1.5, soft 0..0.5 (0.05) | product of: the hue term [1 − smoothstep(half, half + 360·soft, abs(wrap(h − centre)))] with span = hi − lo when hi > lo, else hi − lo + 360 (a wrapping range, 330..30 = 60), centre = lo + span/2, half = span/2 (a pixel with chroma C < 1e-5 has hue 0°, as atan2(0, 0), on every target, so a hue range containing 0° also selects neutral greys: add a sat range to keep them out); a span of 360 (0..360) selects every hue: the whole hue term is left out, so
greys are included as well; lo = hi is rejected by the checker; band(C, sat, 0.2·soft); band(Y, lum, soft). Omitted terms are 1 |
| luma | range (required), soft 0..0.5 (0.05) | band(Y, lo, hi, soft) |
| rgb_key | red, green, blue ranges −0.1..1.5 (each optional), soft 0..0.5 (0.05) | product of band(W_c, lo, hi, soft) over the channels given (1 when none is) |
| skin | width 10..90 (28), soft (0.05) | qualifier(hue 141 ± width/2, sat 0.006..0.09, lum 0.15..0.7) |
| window | centre pair 0..1 (0.5, 0.5), radius pair 0.01..2 (0.4, 0.4), angle −180..180 deg (0), soft 0..1 (0.3) | angle 0: d = √(((x̂ − cx)/rx)² + ((ŷ − cy)/ry)²); else in the rotated frame R(θ) below, d = √((u/rx)² + (v/(k·ry))²); 1 − smoothstep(1 − soft, 1 + soft, d) |
| box | centre pair 0..1 (0.5, 0.5), size pair 0.01..2 (0.3, 0.2) half extents, angle −180..180 (0), roundness 0..1 (0.2), soft 0..1 (0.1) | in R(θ): n = 2 + 18·(1 − roundness); d = (abs(u/size₀)ⁿ + abs(v/(k·size₁))ⁿ)^(1/n); 1 − smoothstep(1 − soft, 1 + soft, d) |
| polygon | points 3..16 [x −1..2, y −1..2] (required), soft 0..0.5 (0.01) frame widths | in frame widths (y·k): s = distance to the outline, + inside (even-odd), − outside; smoothstep(−soft, soft, s) |
| graduated | centre pair 0..1 (0.5, 0.5), angle −180..180 deg (90), falloff pair 0..2 ([0.25, 0.25]) | smoothstep(−falloff₀, falloff₁, u): 0 behind the line through the centre, 1 ahead of it in the direction of the angle (90 = up) |
| depth | range metres 0.01..10⁴ (required), soft 0..2 (0.1) | band(ln depth, ln lo, ln hi, soft); needs `requires depth` |
| segment | ids list of ints 0..65535 (required) | 1 where the host's segment plane is one of ids; needs `requires segments` |

### 6.1 Keys (key routing)

`key NAME = mask` names a mask that grades nothing; `key(NAME, gain: 1, offset: 0)` in any mask is
clip(gain·m + offset, 0, 1), gain 0..4, offset −1..1, both keyframable. Gain and offset apply before an enclosing
`not`: `not key(a, gain: 0.5)` = 1 − 0.5·m. The key mixer: `a + b` = min(a + b, 1), `a − b` = max(a − b, 0) (the
grading app's add and subtract / mask modes), `and` multiplies, `or` takes the maximum. `+`, `-` and `or` share a
level and read left to right, but one level takes one kind: `a or b - c` is an error, `(a or b) - c` is not. `not`
binds tighter than all of them, `and` tighter than `+`, `-`, `or`.

What m is depends on the name:
- **a `key` statement**: its mask, made on this node's input W (a template; every user makes its own).
- **a node**: that node's key carried from where the node ran: its mask expression on that node's own input, before
  its grade, without its `mix` and outside any `layer` opacity, as the grading app routes a node's key output. So
  `mask not key(face)` after a node that changes skin colour still selects exactly what `face` did not (the
  "outside node"). The node's key is made even when the node is `off`. The node must run exactly once, in the same
  graph (`graph` or `photo_graph`), before every node that reads its key in depth-first order (serial items in
  order, parallel branches in order, a layer's bottom before its top); the checker reports any other order.

Names resolve in the checker: an unknown name, a node without a mask and a cycle (`key a = key(b)`, `key b =
key(a)`, or a node reading its own key) are errors; an unused key statement and a key read from an `off` node are
warnings. A key statement's mask counts in full (mask calls, blurred calls as spatial passes, planes it needs) at
every use; a carried node key counts once, at its node. Targets: DCTL / WGSL / CUDA make a carried key once, in a
function k_NAME called before the node, and pass it to the nodes that read it; a split (`--split`) part boundary
between a node and a reader of its key is refused. A LUT takes keys whose masks are pointwise.

Rotated frame R(θ) (window with angle ≠ 0, graduated): k = (H − 1)/(W − 1) (1 when W = 1 or H = 1); offsets in frame widths with
y up: dx = x̂ − cx, up = (cy − ŷ)·k; u = dx·cos θ + up·sin θ, v = up·cos θ − dx·sin θ (θ counter-clockwise on screen).

## 7. Graph

`a -> b` runs a then b. `pass` returns its input unchanged: `layer(pass, glow, mode: screen)` screens a
glow over the picture as it arrives, and `parallel(pass, a)` is a at half strength. `parallel(g₁, …, gₙ)` runs every branch on the same input and returns the unweighted mean
of the branch outputs, in ACEScct code values: (g₁(W) + … + gₙ(W)) / n.
`layer(bottom, top, mode: normal, opacity: 1)` (defaults shown, §2) runs both on the same input and returns
`B + opacity·(blend(B, T) − B)` with
blend: normal T; lighten max(B, T); darken min(B, T); add enc(max(dec B + dec T, 0)); multiply
enc(max(dec B·dec T / 0.18, 0)) (a mid-grey top is neutral); subtract enc(max(dec B − dec T, 0)); difference
enc(abs(dec B − dec T)); and on the code values themselves: screen 1 − (1 − B)(1 − T); overlay 2BT where B < 0.5,
else 1 − 2(1 − B)(1 − T). A node may be used more than once. `off` nodes pass
their input through.

## 8. Time

`keys(interp, f₀: v₀, …)` gives a value per integer frame f: v₀ for f ≤ f₀, vₙ for f ≥ fₙ; between keys `step`
holds the last key ≤ f, `linear` interpolates, `smooth` is the monotone curve of §4 through the keys (no
overshoot). Keys are allowed on num parameters, rgb / pair components, `mix` and layer `opacity`, and on mask
numbers; not on points, ints, enums or ranges in v0.

## 9. Outputs

| output | parameters | maths |
|---|---|---|
| aces2 | display srgb\|rec1886\|display_p3\|p3d65\|pq_p3_1000\|pq_rec2020_1000\|pq_p3_2000\|pq_p3_4000\|hlg_p3_1000 (srgb) | OpenColorIO built-in config `studio-config-v4.0.0_aces-v2.0_ocio-v2.5`, ACEScct in, (display, view): srgb ("sRGB - Display", "ACES 2.0 - SDR 100 nits (Rec.709)"); rec1886 ("Rec.1886 Rec.709 - Display", same view); display_p3 / p3d65 ("Display P3 - Display" / "P3-D65 - Display", "ACES 2.0 - SDR 100 nits (P3 D65)"); pq_* ("Rec.2100-PQ - Display", "ACES 2.0 - HDR 1000 / 2000 / 4000 nits (P3 D65)" or "(Rec.2020)" at 1000); hlg_p3_1000 ("Rec.2100-HLG - Display", "ACES 2.0 - HDR 1000 nits (P3 D65)"). Exact per-pixel evaluation; clipped to [0, 1]. A pixel with an input channel > 0.5 cv whose output is ≤ 10⁻⁵ in every channel (the library's gamut-compression NaN, clamped) is re-evaluated at the nearest of ±1, ±2, 4·10⁻⁴ cv. Note checks and the grading loop measure SDR Rec.709 outputs only |
| display_encode | eotf srgb\|gamma24 (srgb) | L = M_AP1→709·dec(W); soft clip: y = clip(Y_L, 0, 1), p = max_c L; where p > 0.9 target 0.9 + 0.1·tanh((p − 0.9)/0.1) and L ← y + (L − y)·clip((target − y)/max(p − y, 1e−9), 0, 1); where min_c L < 0, L ← y + (L − y)·clip(y/max(y − min, 1e−9), 0, 1); clip [0, 1]; then sRGB or L^(1/2.4). No tone scale |
| acescct | | none: the graded working values (an LMT) |

## 10. Checker guarantees

A program that passes `cierra.check` has: a known version; known input/output; only known ops and masks; only known
parameters with the right types, each in range; all required parameters (including grain's seed); sorted points
where required; valid, increasing keyframes; graph references to defined nodes, photo-only ops only in
`photo_graph`; aux planes declared for depth / segment masks; and every size limit of §3. There is no construct for
file, network or process access, code, expressions or loops, so the checker does not need to look for them: the
parser cannot produce them. Rejections list every problem with line and node.

Warnings (the program is legal but probably not what was meant): a node not in any graph; a node that is `off`; a
node whose op is the identity at its defaults and whose parameters are all at their defaults (it does nothing). Only
these ops are the identity at their defaults (`identity_with_defaults` in the manifest): exposure, white_balance,
printer_lights, wheels, cdl, contrast, rgb_mixer, density, hue_rotate, vibrance, split_tone, saturation,
highlights_shadows. Every other op does something with its defaults
(vignette darkens by 0.1, glow adds 0.3, blur blurs 2 px, photo_to_scene decodes the photo), so no warning. Host-dependent lints (`cierra.checker.lint`, given
the host's context — clip length, provided planes, available segment ids, input kind, wanted targets): keyframes past
the clip's last frame, planes the host does not provide, segment ids that do not exist in the shot, a `photo_graph`
that will not run for this input, and nodes the wanted targets cannot express.

Generation loop (LLM or any author): generate → parse / check → canonicalise → lint → render a preview → measure →
revise with the diagnostics and the numbers, for a bounded number of rounds; every accepted version is committed with
its readable diff (`cierra.llm.grade_loop`, `cierra.history`).

## 11. Targets

| target | covers | notes |
|---|---|---|
| reference engine (NumPy float64) | everything | source of truth |
| DCTL | pointwise + position ops, all masks except depth / segment / blurred (blur_px) / feathered (feather_px) | ACEScct in/out, float32, keyframes at one frame; spatial nodes refused, or with split=True the top-level serial graph is cut into several DCTLs around them; a curve whose points or tangents do not survive float32 is refused |
| CLF v3 / .cube | pointwise graphs without keyframes (on nodes, masks or layer opacity) | 65³, domain 0..1 ACEScct, lattice fitted by least squares for tetrahedral interpolation (8 points per cell); LMT (ACEScct→ACEScct) and graph + output (ACEScct→display) |
| WGSL | same as DCTL | one compute shader: ACEScct float32 r, g, b interleaved in (binding 0), the same layout out (binding 1), uniform width / height (binding 2), workgroup 8 × 8; ACEScct in/out, float32, keyframes at one frame; the same refusals and split=True as DCTL (one shared code generator) |
| ASC CDL (.cc / .ccc / .cdl) | a serial graph of printer_lights, contrast, cdl, saturation (protect_skin 0); exposure and white_balance only at their defaults (a linear gain is not an ACEScct offset below the toe) without masks, mix < 1 or keyframes, in CDL order (affine ops, at most one power, then saturation) | folded into one slope / offset / power / saturation on ACEScct (an LMT); everything else refused with the reasons. ASC readers clamp to 0..1 after the SOP and after saturation, Cierra's cdl op does not |
| CUDA | same as DCTL | one CUDA C++ source needing no headers: `__device__` helper and node functions and an `extern "C" __global__ void cierra_grade(const float* src, float* dst, int width, int height, float px_scale)` kernel, one thread per pixel (16 × 16 blocks); ACEScct float32 r, g, b interleaved, row-major, in and out (not in place); px_scale = width / reference_width (the source #defines CIERRA_REFERENCE_WIDTH); float32, keyframes at one frame; the same refusals and split=True as DCTL (one shared code generator) |
| Triton, OFX | — | later milestones |

Parity requirement, scoped to a conformance set rather than promised for every legal program: on the conformance
corpus (tests/data/frames: a synthetic chart and CC0 camera RAW photos) and the conformance programs (the shipped looks and the
every-op program in tests/test_targets.py), the target's **final output** is compared with the reference engine's
final output for the same export:
- a target whose output is display values (a graph + output LUT, a whole-program render): compared directly, with
  no transform added, as ΔE00 (CIEDE2000, D65) after decoding both with the program's own display EOTF (sRGB for
  `srgb`, gamma 2.4 for `rec1886` / `gamma24`);
- a target whose output is ACEScct (the DCTL, the WGSL shader, the CUDA kernel, the LMT .cube / .clf): numerically, |ΔACEScct| ≤ 0.002 cv at the 99.9th
  percentile and ≤ 0.005 cv max for LUTs; and perceptually, after applying the program's output transform to both
  (once, on each side) and decoding as above.

ΔE00 must be ≤ 0.5 at the 99.9th percentile and ≤ 0.1 mean for LUT targets, ≤ 0.3 mean for the graph + output LUT,
and ≤ 0.5 max for DCTL, WGSL and CUDA. A 3D LUT samples a function on a 65³ lattice over ACEScct 0..1;
it cannot promise this for every program (sharp qualifier edges, hue changes near neutral), so a LUT export reports its
measured error and anything outside that domain is clamped to it by the reading tool (ACEScct 1.0 ≈ 222 linear).
A reading tool that interpolates trilinearly instead of tetrahedrally is less accurate (on the conformance set,
Beauty Modern's worst LMT pixel is ΔE00 0.48 tetrahedral, 0.54 trilinear). Programs that need more accuracy should use
the DCTL, the WGSL shader, the CUDA kernel or the reference engine.

## 12. Determinism

Same program text, same input, same frame, same machine → bit-identical output from the reference engine (no
threads with reductions in varying order, no unseeded randomness, no time or environment input). Across machines
float64 transcendental functions may differ in the last bits; golden tests use a 2e−6 tolerance on display values.

Grain across targets: the integer hash (§5) is exact on every target (uint32 arithmetic), so every target picks the
same random cells. What follows is floating point (float64 in the reference, float32 in DCTL, WGSL and CUDA), so grain values agree
within the target parity tolerance above, not bit for bit.

## Notes (non-normative)

N1. `photo_to_scene` `recovered_highlights`: with the default roll-off (4.5 → 6 stops over grey) its effect is small —
under 0.001 cv measured on a pixel 1.6 stops over display white (tests/test_regression.py), because the inverse
output transform already sends display white past the roll-off ceiling. It is kept for parity with the engine the looks were first built on.
Plan for 0.2 (2026-10-27): remove the parameter (the behaviour stays as with true) unless that engine gives it a
measurable effect by then; a program that sets it then draws a warning for one version before it becomes an error.
