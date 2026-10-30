"""The command line's image reader keeps bit depth: 16-bit and float TIFFs (a 16-bit TIFF used to fail to load)."""

import numpy as np
import pytest

from cierra.__main__ import read_image

tifffile = pytest.importorskip("tifffile")


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_tiff_bit_depths_load_as_code_values(dtype, tmp_path):
    rng = np.random.default_rng(7)
    v = rng.random((5, 6, 3))
    top = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    data = (v * top).round().astype(dtype) if top != 1.0 else v.astype(dtype)
    tifffile.imwrite(tmp_path / "a.tif", data, photometric="rgb")
    got = read_image(tmp_path / "a.tif")
    assert got.shape == (3, 5, 6) and got.dtype == np.float64
    assert np.allclose(got, data.transpose(2, 0, 1) / top, atol=1e-7)


def test_16bit_keeps_steps_8bit_would_merge(tmp_path):
    ramp = np.repeat(np.arange(0, 512, dtype=np.uint16)[None, :, None], 3, axis=2)  # 512 steps below 8-bit code 2
    tifffile.imwrite(tmp_path / "r.tif", ramp, photometric="rgb")
    assert len(np.unique(read_image(tmp_path / "r.tif")[0])) == 512


def test_grey_and_alpha_tiffs_give_three_channels(tmp_path):
    tifffile.imwrite(tmp_path / "g.tif", np.full((4, 4), 65535, np.uint16))
    tifffile.imwrite(tmp_path / "a.tif", np.full((4, 4, 4), 255, np.uint8), photometric="rgb", extrasamples=["unassalpha"])
    assert np.allclose(read_image(tmp_path / "g.tif"), 1.0) and read_image(tmp_path / "g.tif").shape == (3, 4, 4)
    assert read_image(tmp_path / "a.tif").shape == (3, 4, 4)


@pytest.mark.parametrize("h", [3, 4, 6])  # 3 and 4 rows must not be taken for planar channels
def test_grey_alpha_and_one_channel_tiffs_give_three_channels(tmp_path, h):
    la = np.zeros((h, 5, 2), np.uint16)
    la[..., 0], la[..., 1] = 32768, 65535
    tifffile.imwrite(tmp_path / "la.tif", la, photometric="minisblack", extrasamples=["unassalpha"])
    tifffile.imwrite(tmp_path / "one.tif", la[..., :1], photometric="minisblack")
    for name in ("la.tif", "one.tif"):
        img = read_image(tmp_path / name)
        assert img.shape == (3, h, 5) and np.allclose(img, 32768 / 65535), (name, img.shape)


def test_render_keeps_a_photo_programs_own_input_for_png(tmp_path):
    """A program fitted from a Rec.709 video LUT declares rec1886_photo: a PNG is decoded with gamma 2.4, so the
    identity program gives the picture back (decoded as sRGB it would come back changed)."""
    from PIL import Image

    from cierra.__main__ import main
    codes = np.linspace(0.05, 0.95, 64 * 3).reshape(8, 8, 3)
    Image.fromarray((codes * 255 + 0.5).astype(np.uint8)).save(tmp_path / "in.png")
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput rec1886_photo\noutput display_encode(eotf: gamma24)\n")
    assert main(["render", str(tmp_path / "p.cierra"), str(tmp_path / "in.png"), str(tmp_path / "out.png")]) in (0, None)
    a = np.asarray(Image.open(tmp_path / "in.png"), dtype=float)
    b = np.asarray(Image.open(tmp_path / "out.png"), dtype=float)
    assert np.abs(a - b).max() <= 1


def test_render_decodes_png_as_srgb_for_a_linear_photo_program(tmp_path):
    """linear_photo takes display-linear floats; a PNG holds sRGB codes, so it is decoded as srgb_photo and an
    identity program gives it back (not brightened, as if the codes were linear)."""
    from PIL import Image

    from cierra.__main__ import main
    Image.fromarray(np.full((4, 4, 3), 128, np.uint8)).save(tmp_path / "in.png")
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput linear_photo\noutput display_encode(eotf: srgb)\n")
    main(["render", str(tmp_path / "p.cierra"), str(tmp_path / "in.png"), str(tmp_path / "out.png")])
    assert np.abs(np.asarray(Image.open(tmp_path / "out.png"), dtype=float) - 128).max() <= 1


def test_planar_grey_alpha_and_multi_page_tiffs(tmp_path):
    planar = np.zeros((2, 6, 7), np.uint16)
    planar[0], planar[1] = 32768, 65535
    tifffile.imwrite(tmp_path / "p.tif", planar, photometric="minisblack", planarconfig="separate",
                     extrasamples=["unassalpha"])
    pages = np.zeros((2, 6, 7, 3), np.uint16)  # one series of two pages: imread alone returns a 4-D array
    pages[0] = 65535
    tifffile.imwrite(tmp_path / "m.tif", pages, photometric="rgb")
    p = read_image(tmp_path / "p.tif")
    assert p.shape == (3, 6, 7) and np.allclose(p, 32768 / 65535)
    m = read_image(tmp_path / "m.tif")
    assert m.shape == (3, 6, 7) and np.allclose(m, 1.0)  # the first page
