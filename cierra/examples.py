"""Authoring examples: valid programs for each language feature and invalid ones with the checker's diagnostics.
One source for the LLM prompt (llm.language_reference), docs/authoring.md (tools/make_authoring.py) and the tests
(tests/test_authoring.py checks every valid example passes clean and every invalid one fails with its listed
message)."""

VALID = [
    ("minimal: an output transform only (no graph = identity)", """cierra 0.1
output aces2(display: srgb)
"""),
    ("serial primaries", """cierra 0.1
name "warm, a little richer"
input linear_rec709
output aces2(display: srgb)

node balance = white_balance(temperature: 0.1)
node tone = contrast(amount: 1.05, pivot: 0.35)
node colour = saturation(amount: 1.08, protect_skin: 0.5)

graph balance -> tone -> colour
"""),
    ("masks: a skin key, and its inverse combined with a luma key", """cierra 0.1
output aces2(display: srgb)

node skin_glow = saturation(amount: 1.12)
  label "richer skin"
  mask skin(width: 30)
node cool_shadows = wheels(lift: [-0.006, 0, 0.01])
  mask not skin() and luma(range: -0.1..0.35, soft: 0.05)
  mix 0.8

graph skin_glow -> cool_shadows
"""),
    ("hue curves and a hue qualifier that wraps through 0", """cierra 0.1
output aces2(display: srgb)

node greens_to_gold = hue_vs_hue(points: [[206, -12, 40]])
node calm_reds = saturation(amount: 0.9)
  mask qualifier(hue: 330..30, sat: 0.02..1, soft: 0.05)

graph greens_to_gold -> calm_reds
"""),
    ("parallel (average of branches) and layer (blend)", """cierra 0.1
output aces2(display: srgb)

node lift = exposure(stops: 0.3)
node mute = saturation(amount: 0.7)
node bright = exposure(stops: 1)
node punch = contrast(amount: 1.3)

graph parallel(lift, mute) -> layer(bright, punch, mode: multiply, opacity: 0.3)
"""),
    ("key routing: a named key, a node's key carried downstream, the outside node, the key mixer", """cierra 0.1
output aces2(display: srgb)

# a named key: made again on the input of each node that uses it
key sky = graduated(centre: [0.5, 0.4], angle: 90, falloff: [0.1, 0.2])
# a node's own key: key(face) below is this key as it was made here, before face graded
node face = saturation(amount: 1.2)
  mask skin()
# the outside node: everything face did not select, minus half the sky (mixing or / + / - needs brackets)
node rest = saturation(amount: 0.8)
  mask not key(face) - key(sky, gain: 0.5)
node sky_cool = white_balance(temperature: -0.2)
  mask key(sky)

graph face -> rest -> sky_cool
"""),
    ("keyframes: a fade up over 24 frames, smooth", """cierra 0.1
output aces2(display: srgb)

node fade = exposure(stops: keys(smooth, 0: -3, 24: 0))

graph fade
"""),
    ("texture with a fixed grain seed", """cierra 0.1
output aces2(display: srgb)

node bloom = glow(threshold: 0.62, radius_px: 30, strength: 0.2)
node film_grain = grain(amount: 0.15, size_px: 1.5, seed: 1234)

graph bloom -> film_grain
"""),
    ("display-referred photo input: photo_to_scene first, in photo_graph", """cierra 0.1
input srgb_photo
output aces2(display: srgb)

node to_scene = photo_to_scene()
node tone = contrast(amount: 1.04, pivot: 0.3)

photo_graph to_scene
graph tone
"""),
    ("depth and segment keys (the host provides the planes)", """cierra 0.1
output aces2(display: srgb)
requires depth, segments

node far_cool = white_balance(temperature: -0.15)
  mask depth(range: 20..10000, soft: 0.2)
node person_warm = white_balance(temperature: 0.08)
  mask segment(ids: [3])

graph far_cool -> person_warm
"""),
]

INVALID = [
    ("value out of range", "cierra 0.1\nnode a = exposure(stops: 12)\ngraph a\n", "12 is outside [-6, 6]"),
    ("unknown parameter", "cierra 0.1\nnode a = saturation(amout: 1.1)\ngraph a\n", "has no parameter 'amout'"),
    ("expression instead of a number", "cierra 0.1\nnode a = exposure(stops: 1 + 1)\ngraph a\n",
     "no arithmetic (+, -)"),
    ("grain without a seed (non-deterministic)", "cierra 0.1\nnode a = grain(amount: 0.2)\ngraph a\n",
     "grain() needs 'seed'"),
    ("depth key without declaring the plane", "cierra 0.1\nnode a = exposure()\n  mask depth(range: 1..5)\ngraph a\n",
     "add `requires depth`"),
    ("graph names a node that does not exist", "cierra 0.1\nnode a = exposure(stops: 1)\ngraph a -> b\n",
     "no node named 'b'"),
    ("a node's key read before that node runs", "cierra 0.1\nnode a = exposure(stops: 1)\n  mask skin()\n"
     "node b = saturation(amount: 0.5)\n  mask not key(a)\ngraph b -> a\n", "is read before node a runs"),
    ("photo_to_scene outside photo_graph", "cierra 0.1\nnode s = photo_to_scene()\ngraph s\n",
     "may only run in photo_graph"),
    ("empty hue range", "cierra 0.1\nnode a = saturation(amount: 1.1)\n  mask qualifier(hue: 40..40)\ngraph a\n",
     "is empty"),
    ("unsorted curve points", "cierra 0.1\nnode a = curve(points: [[0.6, 0.6], [0.2, 0.2]])\ngraph a\n",
     "increase strictly"),
]

GUIDANCE = [
    "Write the whole program every time, starting with `cierra 0.1`.",
    "Name nodes after what they do (warm_skin, cool_shadows); one operation per node.",
    "Leave parameters at their defaults out; give only what you change.",
    "One `graph` line wires every node you want to run; nodes not in a graph do nothing (and draw a warning).",
    "Declare `requires depth` / `requires segments` before using depth() / segment() masks, and use only segment ids"
    " the host lists.",
    "grain() always needs an integer seed.",
    "Keep keyframes inside the clip (frames 0 .. length - 1).",
    "photo_to_scene() goes first in photo_graph, once, outside parallel / layer; photo_graph runs only for photo"
    " input (srgb_photo, or linear_photo when highlights were recovered above display white).",
    "Defaults are not always 'no change': vignette(), glow(), blur(), grain() do something with their defaults;"
    " give the values you mean.",
    "mix (0..1, default 1) and layer opacity (0..1, default 1) can be keyframed; layer mode defaults to normal.",
    "Small steps: wheel offsets of 0.005-0.02 cv, saturation 0.9-1.15, exposure within +-1 stop are typical.",
]
