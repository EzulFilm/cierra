"""The two README figures, deterministic: docs/img/readme_pipeline.png (a photo -> the program in
examples/bw_red_filter.cierra -> the graded photo) and docs/img/readme_looks.png (four photos, ungraded and through five
of the example programs).

    python tools/make_readme_figures.py

The photos are CC0 camera RAWs from raw.pixls.us. They are downloaded once into cache/raw/readme/ and pinned by
SHA-256 (a file with any other hash is refused), decoded with LibRaw through rawpy (`pip install -e ".[raw]"`) to
scene-linear ACES2065-1 with the camera's white balance and no auto brightness, and box-averaged to 960 pixels wide.
"""

import hashlib
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from cierra.checker import load, load_file  # noqa: E402
from cierra.engine import render  # noqa: E402
from raw_to_video import decode_raw, downscale  # noqa: E402

# (title, camera, URL, SHA-256): raw.pixls.us, CC0
PHOTOS = {
    "tea": ("tea plantation", "Sony DSC-RX10M4",
            "https://raw.pixls.us/getfile.php/2247/nice/Sony - DSC-RX10M4 - 12bit 12bit compressed (3:2).ARW",
            "1825b615a472962bbef13a6717e754e2aeb53d353de10d3503767932c420070c"),
    "city": ("city skyline", "Google Pixel 2 XL",
             "https://raw.pixls.us/getfile.php/2206/nice/Google - Pixel 2 XL - 16bit (4:3).dng",
             "5541093a369f0487fef92bcd1f5fef1be6e4e3cec2aa4d52ad38d2211f3309d7"),
    "tree": ("lone tree", "Sony DSLR-A380",
             "https://raw.pixls.us/getfile.php/5346/nice/Sony - DSLR-A380 - 12bit 12bit compressed (3:2).ARW",
             "1bdf11f9bcc65568b855c4ab9aa456d566ba6104a888a5567b237cf175e75a6b"),
    "pens": ("coloured pens", "Panasonic DMC-G6",
             "https://raw.pixls.us/getfile.php/1096/nice/Panasonic - DMC-G6 - 4:3.RW2",
             "5b20c3ec272a4c8eca453205519ea92d061126a5b3153b7f60b7143d24f399d9"),
}
CACHE = ROOT / "cache" / "raw" / "readme"
WIDTH = 960

EXAMPLE = ROOT / "examples" / "bw_red_filter.cierra"
PIPELINE_PHOTO = "city"
# the program as the figure shows it: no comment, `about` or `label` lines
PROGRAM = "".join(ln for ln in EXAMPLE.read_text(encoding="utf-8").splitlines(True)
                  if not ln.lstrip().startswith(("#", "about ", "label ")))

UNGRADED = "cierra 0.1\ninput aces2065_1\noutput aces2(display: rec1886)\n"  # the same output as the looks
OUT = ROOT / "docs" / "img"
LOOKS = [("red-filter B&W", "examples/bw_red_filter.cierra"), ("sky drama", "examples/sky_drama.cierra"),
         ("70s print", "examples/warm_70s_print.cierra"), ("Orton", "examples/orton_landscape.cierra"),
         ("cross-process", "examples/cross_process_video.cierra")]
ROWS = ["tea", "city", "tree", "pens"]

BG, PANEL, TEXT, DIM = (40, 42, 46), (28, 30, 33), (225, 226, 228), (150, 153, 158)
SYNTAX = {"kw": (198, 146, 234), "op": (130, 190, 255), "num": (247, 178, 103), "str": (195, 232, 141),
          "param": (160, 168, 180)}
KEYWORDS = {"cierra", "name", "input", "output", "node", "mask", "graph", "not", "photo_graph", "label"}
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(size, mono=False, bold=False):
    name = "DejaVuSansMono" if mono else "DejaVuSans"
    try:
        return ImageFont.truetype(str(FONT_DIR / (name + ("-Bold" if bold else "") + ".ttf")), size)
    except OSError:
        return ImageFont.load_default()


def fetch(key):
    """The pinned RAW file for a photo: from the cache when its hash matches, else downloaded and checked."""
    _, _, url, sha = PHOTOS[key]
    path = CACHE / (sha[:16] + Path(url).suffix.lower())
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    with urllib.request.urlopen(urllib.request.Request(url.replace(" ", "%20")), timeout=120) as r:
        data = r.read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"{url}: SHA-256 {got}, expected {sha}; refusing it")
    path.write_bytes(data)
    return path


_PHOTO = {}


def photo(key):
    if key not in _PHOTO:
        _PHOTO[key] = downscale(decode_raw(fetch(key), "camera"), WIDTH)
    return _PHOTO[key]


def tile(prog, key, width):
    img = photo(key)
    a = (np.clip(render(prog, img), 0, 1).transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)
    t = Image.fromarray(a)
    return t.resize((width, round(t.height * width / t.width)), Image.LANCZOS) if t.width != width else t


def text_w(draw, s, f):
    return draw.textlength(s, font=f)


def credit(draw, w, h, keys):
    cams = ", ".join(PHOTOS[k][1] for k in keys)
    s = f"Photo{'s' if len(keys) > 1 else ''}: raw.pixls.us, CC0 ({cams})"
    f = font(13)
    draw.text(((w - text_w(draw, s, f)) / 2, h - 24), s, fill=DIM, font=f)


TOKEN = re.compile(r'(?P<str>"[^"]*")|(?P<num>-?\d+(?:\.\d+)?)|(?P<word>[A-Za-z_]\w*)|(?P<other>\s+|.)')


def code_line(draw, x, y, line, f):
    for m in TOKEN.finditer(line):
        s, kind = m.group(), m.lastgroup
        rest = line[m.end():]
        if kind == "word":
            if s in KEYWORDS and not rest.startswith(":"):
                kind = "kw"
            elif rest.startswith("("):
                kind = "op"
            elif rest.startswith(":"):
                kind = "param"
        draw.text((x, y), s, fill=SYNTAX.get(kind, TEXT), font=f)
        x += text_w(draw, s, f)


def arrow(draw, x0, x1, y):
    draw.line([(x0, y), (x1 - 10, y)], fill=DIM, width=3)
    draw.polygon([(x1, y), (x1 - 14, y - 8), (x1 - 14, y + 8)], fill=DIM)


def pipeline():
    """Input and graded photo side by side, the program that made it underneath (its lines are too long to fit
    between the pictures at a readable size)."""
    m, gap, top, lh = 24, 72, 50, 22
    mono, lab = font(15, mono=True), font(17, bold=True)
    lines = PROGRAM.rstrip("\n").split("\n")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    cw = int(max(text_w(probe, s, mono) for s in lines)) + 36
    iw = (cw - gap) // 2
    left, right = tile(load(UNGRADED), PIPELINE_PHOTO, iw), tile(load_file(EXAMPLE), PIPELINE_PHOTO, iw)
    ch = len(lines) * lh + 32
    w = m + cw + m
    py = top + left.height + 52
    h = py + ch + 56
    sheet = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(sheet)
    xs = [m, m + iw + gap]
    sheet.paste(left, (xs[0], top))
    sheet.paste(right, (xs[1], top))
    for x, s in [(xs[0], "input"), (xs[1], "output")]:
        d.text((x + (iw - text_w(d, s, lab)) / 2, top - 32), s, fill=TEXT, font=lab)
    arrow(d, xs[0] + iw + 12, xs[1] - 12, top + left.height // 2)
    d.text((m, py - 30), "the Cierra program between them", fill=TEXT, font=lab)
    d.rounded_rectangle([m, py, m + cw, py + ch], radius=10, fill=PANEL)
    for i, line in enumerate(lines):
        code_line(d, m + 18, py + 16 + i * lh, line, mono)
    credit(d, w, h, [PIPELINE_PHOTO])
    sheet.save(OUT / "readme_pipeline.png", optimize=True)


def looks():
    tw, gap, lw, m, top = 300, 8, 118, 20, 40
    progs = [("ungraded", load(UNGRADED))] + [(t, load_file(ROOT / f)) for t, f in LOOKS]
    rows = [[tile(p, r, tw) for _, p in progs] for r in ROWS]
    w = m + lw + len(progs) * tw + (len(progs) - 1) * gap + m
    h = top + sum(r[0].height for r in rows) + (len(rows) - 1) * gap + 48
    sheet = Image.new("RGB", (w, h), BG)
    d = ImageDraw.Draw(sheet)
    hf, rf = font(16, bold=True), font(15)
    for j, (title, _) in enumerate(progs):
        x = m + lw + j * (tw + gap)
        d.text((x + (tw - text_w(d, title, hf)) / 2, top - 28), title, fill=TEXT, font=hf)
    y = top
    for key, row in zip(ROWS, rows):
        th, name = row[0].height, PHOTOS[key][0]
        d.text((m + lw - 12 - text_w(d, name, rf), y + th / 2 - 9), name, fill=TEXT, font=rf)
        for j, t in enumerate(row):
            sheet.paste(t, (m + lw + j * (tw + gap), y))
        y += th + gap
    credit(d, w, h, ROWS)
    sheet.save(OUT / "readme_looks.png", optimize=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    looks()
    for n in ("readme_pipeline.png", "readme_looks.png"):
        p = OUT / n
        print(p, f"{p.stat().st_size / 1e6:.2f} MB", Image.open(p).size)


if __name__ == "__main__":
    main()
