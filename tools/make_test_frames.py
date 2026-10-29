"""Build Cierra's small test frames (tests/data/frames/*.npy, scene-linear Rec.709 float32, (3, H, W), 480 px wide).

Sources (licence-clean, recorded in tests/data/SOURCES.md):
  chart       synthetic: neutral ramp, 24-patch reference-chart from published sRGB values, skin tones, saturated
              primaries and an HDR highlight row. Made here; no third-party content.
  crowd       a crowd outdoors, faces and skin (Sony ILCE-7SM2), raw.pixls.us, CC0
  pens        coloured pens on green felt, saturated colour (Panasonic DMC-G6), raw.pixls.us, CC0
  window_0/1/2  a red pot on a sill against a bright window (Sony DSC-RX100M5), raw.pixls.us, CC0: three crops
              stepping right by 2 % of the width, standing in for consecutive frames of one shot; +1.5 stops so the
              window clips well above 1.0 (highlight tests)
The RAW files are downloaded once into cache/raw/frames/, pinned by SHA-256, decoded with LibRaw through rawpy
(camera white balance, no auto brightness) to ACES2065-1, converted to linear Rec.709, scaled so the median luma is
0.15 (window: then +1.5 stops), box-downsampled to 480 px wide and saved as float32.

    python tools/make_test_frames.py
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from cierra import colour as C  # noqa: E402

OUT = ROOT / "tests" / "data" / "frames"
CACHE = ROOT / "cache" / "raw" / "frames"
WIDTH = 480
# name -> (camera, URL, SHA-256): raw.pixls.us, CC0
RAWS = {
    "crowd": ("Sony ILCE-7SM2", "https://raw.pixls.us/getfile.php/1080/nice/Sony - ILCE-7SM2 - 12bit 12bit compressed (3:2).ARW",
              "46044fb6a9c805f2b4915cd4970316b4d04b0566bcf5287afaad3ef648d7a3db"),
    "pens": ("Panasonic DMC-G6", "https://raw.pixls.us/getfile.php/1096/nice/Panasonic - DMC-G6 - 4:3.RW2",
             "5b20c3ec272a4c8eca453205519ea92d061126a5b3153b7f60b7143d24f399d9"),
    "window": ("Sony DSC-RX100M5", "https://raw.pixls.us/getfile.php/1324/nice/Sony - DSC-RX100M5 - 12bit 12bit compressed (3:2).ARW",
               "82a4c49ca797eb67732021be9fb2ed19b4dc1f6d90286167b1fc80f688f21bce"),
}


def fetch(name):
    """The pinned RAW file: from the cache when its hash matches, else downloaded and checked."""
    _, url, sha = RAWS[name]
    path = CACHE / (sha[:16] + Path(url).suffix.lower())
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url.replace(" ", "%20"), timeout=120) as r:
        data = r.read()
    if hashlib.sha256(data).hexdigest() != sha:
        raise SystemExit(f"{url}: SHA-256 mismatch, refusing it")
    path.write_bytes(data)
    return path


def linear709(name):
    """Full-resolution scene-linear Rec.709 of a pinned RAW, median luma 0.15."""
    from raw_to_video import decode_raw
    lin = C.apply_matrix(C.matrix("ap0", "rec709"), decode_raw(fetch(name), "camera"))
    return lin * (0.15 / float(np.median(C.luma_cv(lin))))


def box_down(img, width=WIDTH):
    h, w = img.shape[1:]
    s = max(1, w // width)
    hh, ww = h // s * s, w // s * s
    out = img[:, :hh, :ww].reshape(3, hh // s, s, ww // s, s).mean(axis=(2, 4))
    x0 = (out.shape[2] - width) // 2 if out.shape[2] > width else 0  # centre crop to exactly `width`
    return out[:, :, x0:x0 + width]


def chart():
    """Synthetic chart, 480 x 270."""
    h, w = 270, 480
    img = np.zeros((3, h, w))
    # row 1: neutral ramp in stops, -8 .. +4 around 18 % grey (24 steps)
    for i in range(24):
        v = 0.18 * 2.0 ** (-8 + 12 * i / 23)
        img[:, 0:45, i * 20:(i + 1) * 20] = v
    # rows 2-5: 24 patches of a reference chart (published sRGB values, approximate), linearised
    cc = ["735244", "c29682", "627a9d", "576c43", "8580b1", "67bdaa", "d67e2c", "505ba6", "c15a63", "5e3c6c",
          "9dbc40", "e0a32e", "383d96", "469449", "af363c", "e7c71f", "bb5695", "0885a1", "f3f3f2", "c8c8c8",
          "a0a0a0", "7a7a79", "555555", "343434"]
    for k, hexv in enumerate(cc):
        r, c = divmod(k, 6)
        rgb = C.srgb_decode(np.array([int(hexv[j:j + 2], 16) / 255 for j in (0, 2, 4)]))
        img[:, 45 + r * 40:45 + (r + 1) * 40, c * 50:(c + 1) * 50] = rgb.reshape(3, 1, 1)
    # right block: skin tones (light .. deep, display sRGB references) and saturated primaries
    skins = ["f1c27d", "e0ac69", "c68642", "8d5524", "ffdbac", "a1665e"]
    for k, hexv in enumerate(skins):
        rgb = C.srgb_decode(np.array([int(hexv[j:j + 2], 16) / 255 for j in (0, 2, 4)]))
        img[:, 45 + k * 26:45 + (k + 1) * 26, 300:390] = rgb.reshape(3, 1, 1)
    prim = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    for k, rgb in enumerate(prim):
        img[:, 45 + k * 26:45 + (k + 1) * 26, 390:480] = 0.7 * np.array(rgb, float).reshape(3, 1, 1) + 0.02
    # bottom: horizontal hue sweep at two levels, and HDR highlights
    x = np.linspace(0, 1, w)
    hue = np.stack([np.clip(np.abs(x * 6 - 3) - 1, 0, 1), np.clip(2 - np.abs(x * 6 - 2), 0, 1),
                    np.clip(2 - np.abs(x * 6 - 4), 0, 1)])
    img[:, 205:225, :] = (0.05 + 0.5 * hue)[:, None, :]
    img[:, 225:245, :] = (0.01 + 0.08 * hue)[:, None, :]
    img[:, 245:270, :] = (2.0 ** np.linspace(0, 7, w) * 0.18)[None, None, :] * np.array([1.0, 0.9, 0.75]).reshape(3, 1, 1)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "chart.npy", chart().astype(np.float32))
    for name in ("crowd", "pens"):
        np.save(OUT / f"{name}.npy", box_down(linear709(name)).astype(np.float32))
    win = linear709("window") * 2 ** 1.5
    w = win.shape[2]
    crop = w - 2 * (w // 50)  # three crops of the same width, 2 % of the width apart
    for i in range(3):
        x0 = i * (w // 50)
        np.save(OUT / f"window_{i}.npy", box_down(win[:, :, x0:x0 + crop]).astype(np.float32))
    for p in sorted(OUT.glob("*.npy")):
        a = np.load(p)
        print(f"{p.name}: {a.shape} min {a.min():.4f} p50 {np.median(a):.4f} max {a.max():.2f}")


if __name__ == "__main__":
    main()
