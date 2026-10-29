"""The operation table: every op and mask, its parameters, types, units, ranges and defaults, and which targets can
express it. The checker, the formatter, the LLM prompt and docs/spec.md all read this table; the engines implement it.

Parameter kinds
    num      a finite number in [lo, hi]                         (keyframes allowed)
    int      an integer in [lo, hi]
    bool     true / false
    enum     one of the listed identifiers
    rgb      [r, g, b], each in [lo, hi]                         (keyframes allowed per component)
    pair     [a, b], each in [lo, hi]                            (keyframes allowed per component)
    range    lo..hi with lo <= hi (hue ranges may wrap: 330..30)
    points   a list of [x, y] or [x, y, w] control points (see `cols`)
    ints     a list of integers in [lo, hi]

Units: cv = ACEScct code value (mid-grey 18 % = 0.4135884; 1 stop = 1/17.52 cv above the toe), deg = hue degrees in
the opponent space of the working values (docs/spec.md "Hue"), px = pixels at the program's reference_width, stops,
x = a dimensionless factor (1 = unchanged), m = metres.

Op flags
    pointwise  output pixel depends only on the same input pixel (a 3D LUT can hold it)
    position   needs the pixel's position (x, y) but no neighbours (DCTL yes, LUT no)
    spatial    reads neighbouring pixels (blur kernels): neither LUT nor v0 DCTL
    aux        needs a host-supplied plane ("depth" metres, "segments" integer ids)
"""

from dataclasses import dataclass, field

GREY = 0.4135884  # ACEScct code value of 18 % grey


@dataclass(frozen=True)
class P:
    kind: str
    lo: float = None
    hi: float = None
    default: object = None
    unit: str = ""
    doc: str = ""
    choices: tuple = ()
    cols: tuple = ()          # points: ((lo, hi, label), ...) required columns
    opt_cols: tuple = ()      # points: optional trailing columns
    min_len: int = 0
    max_len: int = 0
    sorted: bool = False      # points: x strictly increasing
    required: bool = False


@dataclass(frozen=True)
class OpSpec:
    name: str
    category: str
    doc: str
    params: dict
    pointwise: bool = True
    position: bool = False
    spatial: bool = False
    aux: tuple = ()
    dctl: bool = True         # the v0 DCTL compiler can emit it
    photo_only: bool = False  # only allowed in photo_graph
    extra: dict = field(default_factory=dict)


HUE_NOTE = ("Hue is the angle of (Cb, Cr) of the ACEScct working values, degrees: red ~123, skin ~141, orange ~150, "
            "yellow ~173, green ~206, cyan ~300, blue ~350, magenta ~60.")
PT_HUE = (0.0, 360.0, "hue deg")
PT_WIDTH = ((5.0, 180.0, "width deg"),)

OPS = {}


def _op(name, category, doc, params, **kw):
    OPS[name] = OpSpec(name, category, doc, params, **kw)


# ---------------------------------------------------------------- input stage (photo_graph only)
_op("photo_to_scene", "transform",
    "Display-referred photo (sRGB JPEG / PNG read as linear light) back to scene light through the inverse ACES 2.0 "
    "SDR sRGB output transform, so the program's ACES 2.0 output gives the photo's own placement back. Highlights "
    "roll off from knee_stops to max_stops over grey. Allowed only in photo_graph.",
    {"recovered_highlights": P("bool", default=True, doc="keep stops over display white where every channel is > 1 "
                               "(only linear_photo input can exceed 1; srgb_photo is clipped)"),
     "knee_stops": P("num", 1.0, 10.0, 4.5, "stops", "roll-off start over 18 % grey"),
     "max_stops": P("num", 1.5, 12.0, 6.0, "stops", "roll-off ceiling over 18 % grey")},
    dctl=False, photo_only=True)

# ---------------------------------------------------------------- primaries
_op("exposure", "primary", "Exposure: a linear-light gain (decode ACEScct, multiply by 2^stops, encode); above the "
    "toe it is the same as adding stops / 17.52, and it never lifts black below the toe.",
    {"stops": P("num", -6.0, 6.0, 0.0, "stops", "+1 = one stop brighter")})
_op("white_balance", "primary", "Temperature / tint as linear-light gains per AP1 channel, in stops: decode, "
    "multiply by 2^[temperature, -tint, -temperature], encode (above the toe the same as adding [...] / 17.52).",
    {"temperature": P("num", -2.0, 2.0, 0.0, "stops", "> 0 warmer (red up, blue down)"),
     "tint": P("num", -2.0, 2.0, 0.0, "stops", "> 0 magenta, < 0 green")})
_op("printer_lights", "primary", "Film-lab printer points: 1 point = 0.025 log10 exposure (~1/12 stop) per channel.",
    {c: P("num", -24.0, 24.0, 0.0, "points", f"{c} points") for c in ("red", "green", "blue")})
_op("wheels", "primary",
    "Lift / gamma / gain / offset colour wheels as ACEScct offsets weighted by soft tonal ranges of luma "
    "(shadows, mids, highlights). 0.01 cv is a visible tint, 0.03 strong.",
    {"lift": P("rgb", -0.3, 0.3, (0.0, 0.0, 0.0), "cv", "shadow colour / level"),
     "gamma": P("rgb", -0.3, 0.3, (0.0, 0.0, 0.0), "cv", "mid-tone colour / level"),
     "gain": P("rgb", -0.3, 0.3, (0.0, 0.0, 0.0), "cv", "highlight colour / level"),
     "offset": P("rgb", -0.3, 0.3, (0.0, 0.0, 0.0), "cv", "whole-image colour / level"),
     "master_lift": P("num", -0.3, 0.3, 0.0, "cv", "shadow level"),
     "master_gamma": P("num", -0.3, 0.3, 0.0, "cv", "mid level"),
     "master_gain": P("num", -0.3, 0.3, 0.0, "cv", "highlight level"),
     "master_offset": P("num", -0.3, 0.3, 0.0, "cv", "overall level")})
_op("cdl", "primary",
    "ASC CDL on ACEScct code values, unclamped (CLF 'FwdNoClamp'): v = in * slope + offset; v = v ** power where "
    "v >= 0; then saturation around Rec.709 luma.",
    {"slope": P("rgb", 0.0, 4.0, (1.0, 1.0, 1.0), "x", "per-channel slope"),
     "offset": P("rgb", -0.5, 0.5, (0.0, 0.0, 0.0), "cv", "per-channel offset"),
     "power": P("rgb", 0.2, 4.0, (1.0, 1.0, 1.0), "x", "per-channel power"),
     "sat": P("num", 0.0, 4.0, 1.0, "x", "saturation, 1 = unchanged")})
_op("contrast", "primary", "Log contrast around a pivot: out = (in - pivot) * amount + pivot.",
    {"amount": P("num", 0.3, 3.0, 1.0, "x", "1 = unchanged"),
     "pivot": P("num", 0.0, 1.0, GREY, "cv", "level that stays put (0.4136 = 18 % grey)")})
_op("saturation", "primary", "Saturation: scales opponent chroma (Cb, Cr) of the working values. protect_skin 0..1 "
    "keeps skin hues (141 +- 15..40 deg) near their own saturation.",
    {"amount": P("num", 0.0, 3.0, 1.0, "x", "1 = unchanged, 0 = monochrome"),
     "protect_skin": P("num", 0.0, 1.0, 0.0, "x", "1 = skin untouched")})
_op("curve", "primary",
    "Monotone cubic curve (Fritsch-Carlson, no overshoot) through points; straight lines with the end slopes beyond. "
    "channel luma keeps colour (adds the luma change to all channels); rgb applies to each channel. "
    "space log: points are ACEScct code values; display: on-screen sRGB code values.",
    {"points": P("points", cols=((-0.2, 1.5, "x in"), (-0.2, 1.5, "y out")), min_len=2, max_len=16, sorted=True,
                 required=True, doc="[[x, y], ...], x strictly increasing"),
     "channel": P("enum", choices=("luma", "rgb", "red", "green", "blue"), default="luma"),
     "space": P("enum", choices=("log", "display"), default="log")})
_op("hue_vs_hue", "primary", "Rotate selected hues. Each point [hue, shift, width] acts with a raised-cosine "
    "influence `width` degrees each side (default 60); overlapping points blend. " + HUE_NOTE,
    {"points": P("points", cols=(PT_HUE, (-90.0, 90.0, "shift deg")), opt_cols=PT_WIDTH, min_len=1, max_len=12,
                 required=True, doc="[[hue, shift(, width)], ...]")})
_op("hue_vs_sat", "primary", "Saturation by hue: points [hue, gain, width]; gain 1 = unchanged. " + HUE_NOTE,
    {"points": P("points", cols=(PT_HUE, (0.0, 3.0, "gain")), opt_cols=PT_WIDTH, min_len=1, max_len=12,
                 required=True, doc="[[hue, gain(, width)], ...]")})
_op("hue_vs_lum", "primary", "Brightness by hue (cv offset, weighted by chroma so greys stay): points [hue, offset, "
    "width]. 0.03 is clearly visible. " + HUE_NOTE,
    {"points": P("points", cols=(PT_HUE, (-0.3, 0.3, "offset cv")), opt_cols=PT_WIDTH, min_len=1, max_len=12,
                 required=True, doc="[[hue, offset(, width)], ...]")})
_op("lum_vs_sat", "primary", "Saturation by luma: monotone curve [[luma, gain], ...] over ACEScct luma "
    "(0.15 deep shadow, 0.41 grey, 0.6 bright, 0.75+ highlights).",
    {"points": P("points", cols=((-0.1, 1.5, "luma cv"), (0.0, 3.0, "gain")), min_len=2, max_len=12, sorted=True,
                 required=True, doc="[[luma, gain], ...]")})
_op("sat_vs_sat", "primary", "Remap opponent chroma: monotone curve [[chroma_in, chroma_out], ...] "
    "(0.05 moderate, 0.1+ strong).",
    {"points": P("points", cols=((0.0, 1.0, "chroma in"), (0.0, 1.0, "chroma out")), min_len=2, max_len=12,
                 sorted=True, required=True, doc="[[c_in, c_out], ...]")})
_op("sat_vs_lum", "primary", "Luma by saturation: monotone curve [[chroma, offset], ...] adds offset (cv) to the luma "
    "of colours of that chroma (negative darkens saturated colours).",
    {"points": P("points", cols=((0.0, 1.0, "chroma"), (-0.3, 0.3, "offset cv")), min_len=2, max_len=12,
                 sorted=True, required=True, doc="[[chroma, offset], ...]")})
_op("rgb_mixer", "primary", "Channel mixer on the working values: out = M·in with rows red, green, blue "
    "([1, 0, 0] etc. = unchanged); preserve_luma keeps each pixel's luma.",
    {"red": P("rgb", -2.0, 2.0, (1.0, 0.0, 0.0), "x", "output red = red·[r, g, b]"),
     "green": P("rgb", -2.0, 2.0, (0.0, 1.0, 0.0), "x", "output green = green·[r, g, b]"),
     "blue": P("rgb", -2.0, 2.0, (0.0, 0.0, 1.0), "x", "output blue = blue·[r, g, b]"),
     "preserve_luma": P("bool", default=False, doc="add back the luma the mix changed")})
_op("density", "primary", "Subtractive (film-like) colour density: darkens each pixel by amount times its opponent "
    "chroma, so saturated colours get deeper rather than brighter. 0.5 subtle, 1.5 strong.",
    {"amount": P("num", -1.0, 2.0, 0.0, "x", "luma change = -amount · chroma (cv)")})
_op("hue_rotate", "primary", "Rotates every hue by the same angle (opponent Cb / Cr plane).",
    {"degrees": P("num", -180.0, 180.0, 0.0, "deg", "counter-clockwise on the hue circle")})
_op("vibrance", "primary", "Saturation weighted toward muted colours: k = 1 + amount · (1 − smoothstep(0, 0.12, C)); "
    "already saturated colours move little.",
    {"amount": P("num", -1.0, 1.0, 0.0, "x", "+ richer muted colours, − calmer")})
_op("split_tone", "primary", "Tints shadows and highlights toward two hues (opponent-plane offsets weighted by luma), "
    "balance moves the split point.",
    {"shadow_hue": P("num", 0.0, 360.0, 315.0, "deg", "315 teal, 141 skin / orange"),
     "shadow_amount": P("num", 0.0, 0.1, 0.0, "chroma", "0.01 subtle"),
     "highlight_hue": P("num", 0.0, 360.0, 141.0, "deg", "hue of the highlight tint"),
     "highlight_amount": P("num", 0.0, 0.1, 0.0, "chroma", "0.01 subtle"),
     "balance": P("num", -0.2, 0.3, 0.0, "cv", "moves the split from 18 % grey")})
_op("gamut_compress", "transform", "Reference gamut compression (the open ACES method) on linear AP1: pulls colours "
    "outside the gamut back in along their distance from the achromatic axis, leaving colours inside threshold "
    "untouched.",
    {"threshold": P("rgb", 0.5, 0.99, (0.815, 0.803, 0.880), "x", "[cyan, magenta, yellow] distance where "
                    "compression starts"),
     "limit": P("rgb", 1.01, 2.0, (1.147, 1.264, 1.312), "x", "[cyan, magenta, yellow] distance mapped to the "
                "gamut boundary"),
     "power": P("num", 1.0, 3.0, 1.2, "x", "curve exponent: higher = sharper knee")})

# ---------------------------------------------------------------- position (DCTL yes, LUT no)
_op("vignette", "position", "Darken (amount > 0) or brighten (< 0) the frame edges. 0.1 subtle, 0.3 strong.",
    {"amount": P("num", -0.5, 0.5, 0.1, "x", "strength"),
     "roundness": P("num", 0.0, 1.0, 1.0, "x", "1 = circular, 0 = follows the frame shape"),
     "softness": P("num", 0.05, 1.2, 0.6, "x", "edge softness")},
    pointwise=False, position=True)
_op("grain", "position", "Deterministic film-like grain: value noise on a `size_px` lattice from an integer hash of "
    "(cell, frame, seed, channel), strongest in the mid-tones. Same seed + frame = the same random cells on every "
    "target (grain values within float tolerance).",
    {"amount": P("num", 0.0, 1.0, 0.2, "x", "0.1 fine, 0.3 heavy"),
     "size_px": P("num", 1.0, 8.0, 1.5, "px", "grain cell size"),
     "chroma": P("num", 0.0, 1.0, 0.3, "x", "0 = monochrome grain, 1 = independent per channel"),
     "seed": P("int", 0, 2 ** 31 - 1, None, "", "required: the random seed", required=True)},
    pointwise=False, position=True)

# ---------------------------------------------------------------- spatial (reference engine; DCTL v0 refuses)
_SP = dict(pointwise=False, spatial=True, dctl=False)
_op("glow", "spatial", "Bloom: light above `threshold` (cv) spreads by a Gaussian of `radius_px` and is added back "
    "in linear light, `warm` tints the halo.",
    {"threshold": P("num", 0.0, 1.2, 0.6, "cv", "level where glow starts (0.6 ~ 3.4 stops over grey)"),
     "radius_px": P("num", 1.0, 400.0, 40.0, "px", "Gaussian sigma"),
     "strength": P("num", 0.0, 2.0, 0.3, "x", "amount"),
     "warm": P("num", -0.5, 0.5, 0.0, "x", "> 0 warmer halo")}, **_SP)
_op("halation", "spatial", "Film-style halation: like glow, but the halo is tinted (default red-orange) and starts "
    "lower; light scattered back from the film base round bright edges.",
    {"threshold": P("num", 0.0, 1.2, 0.55, "cv", "level where halation starts"),
     "radius_px": P("num", 1.0, 200.0, 12.0, "px", "Gaussian sigma"),
     "strength": P("num", 0.0, 2.0, 0.3, "x", "amount"),
     "tint": P("rgb", 0.0, 1.0, (1.0, 0.25, 0.05), "x", "halo colour weights")}, **_SP)
_op("diffusion", "spatial", "Diffusion / mist filter: light from bright areas spreads into darker neighbours "
    "(adds max(blur - in, 0) in linear light); lowers local contrast round highlights.",
    {"radius_px": P("num", 1.0, 200.0, 20.0, "px", "Gaussian sigma"),
     "strength": P("num", 0.0, 1.0, 0.25, "x", "amount")}, **_SP)
_op("blur", "spatial", "Gaussian blur of the working values.",
    {"radius_px": P("num", 0.0, 100.0, 2.0, "px", "Gaussian sigma (< 0.3 = no blur)")}, **_SP)
_op("sharpen", "spatial", "Unsharp mask on luma with noise coring (detail below `coring` is left alone).",
    {"amount": P("num", 0.0, 2.0, 0.3, "x", "strength"),
     "radius_px": P("num", 0.3, 10.0, 1.5, "px", "Gaussian sigma"),
     "coring": P("num", 0.0, 0.05, 0.008, "cv", "detail threshold")}, **_SP)
_op("local_contrast", "spatial", "Wide-radius local contrast on luma, blacks protected.",
    {"amount": P("num", -1.0, 1.0, 0.15, "x", "> 0 more, < 0 softer"),
     "radius_px": P("num", 5.0, 600.0, 200.0, "px", "Gaussian sigma"),
     "protect_blacks": P("bool", default=True)}, **_SP)
_op("guided_detail", "spatial", "Edge-aware detail on luma (guided filter): edges stronger than `edge` keep their "
    "shape, so no halos; amount > 0 crisper texture, < 0 smoother.",
    {"amount": P("num", -1.0, 2.0, 0.5, "x", "> 0 more detail, < 0 smoother"),
     "radius_px": P("num", 1.0, 200.0, 8.0, "px", "box radius of the guided filter"),
     "edge": P("num", 0.001, 0.2, 0.02, "cv", "luma step treated as an edge (sqrt of the filter's eps)"),
     "protect_blacks": P("bool", default=True)}, **_SP)
_op("highlights_shadows", "spatial", "Local tone from a blurred luma: highlights < 0 pulls bright areas down, "
    "shadows > 0 opens dark areas. 1.0 = 0.1 cv at full strength.",
    {"highlights": P("num", -3.0, 3.0, 0.0, "x", "< 0 recover highlights"),
     "shadows": P("num", -3.0, 3.0, 0.0, "x", "> 0 lift shadows"),
     "radius_px": P("num", 1.0, 400.0, 60.0, "px", "Gaussian sigma")}, **_SP)

# Ops whose defaults are the identity (all defaults = the node does nothing). Every other op does something with its
# defaults (vignette darkens 0.1, glow adds 0.3, blur blurs 2 px, photo_to_scene decodes the photo).
IDENTITY_DEFAULTS = frozenset({"exposure", "white_balance", "printer_lights", "wheels", "cdl", "contrast", "rgb_mixer",
                               "density", "hue_rotate", "vibrance", "split_tone",
                               "saturation", "highlights_shadows"})

# Node and layer modifiers (not op parameters).
MODIFIERS = {
    "mix": P("num", 0.0, 1.0, 1.0, "x", "blend of the node's result with its input; keyframable"),
    "layer.opacity": P("num", 0.0, 1.0, 1.0, "x", "blend of the blended result over the bottom; keyframable"),
    "layer.mode": P("enum", choices=("normal", "add", "multiply", "lighten", "darken", "screen", "overlay", "subtract", "difference"), default="normal"),
}

# ---------------------------------------------------------------- masks (qualifiers)
MASKS = {}


def _mask(name, doc, params, **kw):
    MASKS[name] = OpSpec(name, "mask", doc, params, **kw)


_MASK_COMMON = {"blur_px": P("num", 0.0, 200.0, 0.0, "px", "blur the matte (spatial: reference engine only)"),
                "feather_px": P("num", 0.0, 200.0, 0.0, "px", "edge-aware feather: the matte follows the picture's "
                                "edges (guided filter on luma; spatial: reference engine only)"),
                "clip_black": P("num", 0.0, 1.0, 0.0, "x", "matte values at or below this become 0"),
                "clip_white": P("num", 0.0, 1.0, 1.0, "x", "matte values at or above this become 1")}
_mask("qualifier", "HSL key on hue (deg), opponent chroma and ACEScct luma; each with smooth edges `soft` wide.",
      {"hue": P("range", 0.0, 360.0, None, "deg", "hue range, may wrap (330..30)"),
       "sat": P("range", 0.0, 1.0, None, "chroma", "chroma range"),
       "lum": P("range", -0.1, 1.5, None, "cv", "luma range"),
       "soft": P("num", 0.0, 0.5, 0.05, "x", "edge softness"), **_MASK_COMMON})
_mask("rgb_key", "RGB key: smooth bands on the red, green and blue ACEScct code values (each optional; all given "
      "must hold).",
      {"red": P("range", -0.1, 1.5, None, "cv", "red range"),
       "green": P("range", -0.1, 1.5, None, "cv", "green range"),
       "blue": P("range", -0.1, 1.5, None, "cv", "blue range"),
       "soft": P("num", 0.0, 0.5, 0.05, "cv", "edge softness"), **_MASK_COMMON})
_mask("luma", "Luma range key (ACEScct).",
      {"range": P("range", -0.1, 1.5, None, "cv", "[lo, hi]", required=True),
       "soft": P("num", 0.0, 0.5, 0.05, "cv", "edge softness"), **_MASK_COMMON})
_mask("skin", "Skin key: hue 141 deg +- width/2, chroma 0.006..0.09, luma 0.15..0.7 (a qualifier preset).",
      {"width": P("num", 10.0, 90.0, 28.0, "deg", "hue width"),
       "soft": P("num", 0.0, 0.5, 0.05, "x", "edge softness"), **_MASK_COMMON})
_mask("window", "Soft ellipse in normalised frame coordinates (0..1, origin top-left), optionally rotated.",
      {"centre": P("pair", 0.0, 1.0, (0.5, 0.5), "frame", "[x, y]"),
       "radius": P("pair", 0.01, 2.0, (0.4, 0.4), "frame", "[rx, ry]"),
       "angle": P("num", -180.0, 180.0, 0.0, "deg", "rotation, counter-clockwise on screen"),
       "soft": P("num", 0.0, 1.0, 0.3, "x", "edge softness"), **_MASK_COMMON},
      pointwise=False, position=True)
_mask("box", "Soft rounded rectangle in normalised frame coordinates, optionally rotated: a superellipse with "
      "exponent 2 + 18·(1 − roundness) (roundness 1 = the ellipse of window(), 0 = nearly square corners).",
      {"centre": P("pair", 0.0, 1.0, (0.5, 0.5), "frame", "[x, y]"),
       "size": P("pair", 0.01, 2.0, (0.3, 0.2), "frame", "[half width, half height]"),
       "angle": P("num", -180.0, 180.0, 0.0, "deg", "rotation, counter-clockwise on screen"),
       "roundness": P("num", 0.0, 1.0, 0.2, "x", "corner rounding"),
       "soft": P("num", 0.0, 1.0, 0.1, "x", "edge softness"), **_MASK_COMMON},
      pointwise=False, position=True)
_mask("polygon", "Soft polygon through 3..16 points in normalised frame coordinates (origin top-left); the soft edge "
      "is measured in frame widths, centred on the outline.",
      {"points": P("points", cols=((-1.0, 2.0, "x"), (-1.0, 2.0, "y")), min_len=3, max_len=16, required=True,
                   doc="[[x, y], ...] in order around the shape"),
       "soft": P("num", 0.0, 0.5, 0.01, "frame widths", "edge softness"), **_MASK_COMMON},
      pointwise=False, position=True)
_mask("graduated", "Graduated filter: 0 behind a straight line through `centre`, 1 ahead of it, in the direction "
      "`angle` (90 = towards the top: a sky grad); `falloff` = [behind, ahead] widths of the soft edge.",
      {"centre": P("pair", 0.0, 1.0, (0.5, 0.5), "frame", "[x, y] on the line"),
       "angle": P("num", -180.0, 180.0, 90.0, "deg", "direction the matte rises towards, counter-clockwise from right"),
       "falloff": P("pair", 0.0, 2.0, (0.25, 0.25), "frame", "[behind, ahead] soft widths (frame widths)"),
       **_MASK_COMMON},
      pointwise=False, position=True)
_mask("depth", "Scene depth key in metres (log-distance band); needs the host's depth plane.",
      {"range": P("range", 0.01, 10000.0, None, "m", "[near, far] metres", required=True),
       "soft": P("num", 0.0, 2.0, 0.1, "log m", "softness"), **_MASK_COMMON},
      pointwise=False, aux=("depth",), dctl=False)
_mask("segment", "Segment-ID key: 1 where the host's integer segment plane is one of `ids`.",
      {"ids": P("ints", 0, 65535, None, "", "segment ids", min_len=1, max_len=64, required=True), **_MASK_COMMON},
      pointwise=False, aux=("segments",), dctl=False)

# ---------------------------------------------------------------- program-level
INPUTS = {
    "linear_rec709": "scene-linear Rec.709 / sRGB primaries, D65",
    "aces2065_1": "ACES2065-1: scene-linear AP0",
    "acescg": "ACEScg: scene-linear AP1",
    "acescct": "ACEScct code values (AP1)",
    "srgb_photo": "display-referred sRGB photo, code values clipped to 0..1 and decoded (photo_graph runs first)",
    "linear_photo": "display-referred photo as display-linear Rec.709 floats, may exceed 1 where a highlight-recovery "
                    "stage restored them (photo_graph runs first)",
    "arri_logc3": "ARRI LogC3, EI 800, ARRI Wide Gamut 3 (through the OpenColorIO studio config)",
    "arri_logc4": "ARRI LogC4, ARRI Wide Gamut 4 (through the OpenColorIO studio config)",
    "sony_slog3_sgamut3": "Sony S-Log3, S-Gamut3 (through the OpenColorIO studio config)",
    "sony_slog3_sgamut3cine": "Sony S-Log3, S-Gamut3.Cine (through the OpenColorIO studio config)",
    "panasonic_vlog": "Panasonic V-Log, V-Gamut (through the OpenColorIO studio config)",
    "canon_log2": "Canon Log 2, Cinema Gamut (D55) (through the OpenColorIO studio config)",
    "canon_log3": "Canon Log 3, Cinema Gamut (D55) (through the OpenColorIO studio config)",
    "red_log3g10": "RED Log3G10, REDWideGamutRGB (through the OpenColorIO studio config)",
    "bmd_film_gen5": "Blackmagic Film Gen 5, Wide Gamut Gen 5 (through the OpenColorIO studio config)",
    "apple_log": "Apple Log (Rec.2020 primaries) (through the OpenColorIO studio config)",
    "dji_dlog": "DJI D-Log, D-Gamut (through the OpenColorIO studio config)",
    "linear_rec2020": "scene-linear Rec.2020 (through the OpenColorIO studio config)",
    "linear_p3d65": "scene-linear P3-D65 (through the OpenColorIO studio config)",
}
OUTPUTS = {
    "aces2": OpSpec("aces2", "output", "ACES 2.0 output transform (OpenColorIO built-in ACES 2.0 studio config): "
                    "SDR 100-nit Rec.709 on an sRGB or Rec.1886 (gamma 2.4) display, SDR P3-D65 (Display P3 or DCI-style "
                    "P3-D65), or HDR: PQ (Rec.2100) at 1000 / 2000 / 4000 nits in P3-D65 or Rec.2020, HLG at 1000 nits. "
                    "The grading loop measures on SDR Rec.709 only.",
                    {"display": P("enum", choices=("srgb", "rec1886", "display_p3", "p3d65", "pq_p3_1000",
                                                   "pq_rec2020_1000", "pq_p3_2000", "pq_p3_4000", "hlg_p3_1000"),
                                  default="srgb")}),
    "display_encode": OpSpec("display_encode", "output", "Working values to Rec.709 linear, saturated colours above "
                             "0.9 soft-compressed toward their luminance, then the display encoding. No tone scale.",
                             {"eotf": P("enum", choices=("srgb", "gamma24"), default="srgb")}),
    "acescct": OpSpec("acescct", "output", "No output transform: ACEScct code values out (an LMT).", {}),
}
BLEND_MODES = ("normal", "add", "multiply", "lighten", "darken", "screen", "overlay", "subtract", "difference")
INTERPS = ("step", "linear", "smooth")

LIMITS = {
    "source_bytes": 65536, "nodes": 64, "graph_depth": 8, "graph_refs": 128, "keys": 64, "max_frame": 10 ** 6,
    "string": 400, "ident": 40,
    "spatial_passes": 16,      # per program: spatial node uses + blurred mask calls, both graphs together
    "parallel_branches": 8,
    "mask_depth": 16,          # nesting of not / and / or / parentheses in one mask expression
    "mask_calls": 16,          # mask calls in one mask expression
    "mask_calls_expanded": 64,  # mask calls a node's mask makes once the key statements it uses are expanded
    "key_statements": 64,      # `key` statements in one program
    "mask_depth_expanded": 64,  # nesting of a node's mask once the key statements it uses are expanded
    "value_depth": 4,          # nesting of [ ] in a value (points are depth 2)
    "number_digits": 40,       # digits in a number literal
    "point_gap": 1e-4,         # smallest x step between sorted control points (curves stay well conditioned)
}


def all_specs():
    return {**OPS, **{f"mask:{k}": v for k, v in MASKS.items()}}
