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


@pytest.mark.parametrize("dtype,step", [(np.uint8, 1), (np.uint16, 256)])
def test_video_range_tiff_maps_legal_black_and_white_to_0_and_1(dtype, step, tmp_path):
    """Video (legal) range: code 16*2^(n-8) is 0 and 235*2^(n-8) is 1; sub-blacks and super-whites are kept outside
    0..1, not clipped. Read as full range (the default) the same file gives the old values, black lifted."""
    codes = np.array([0, 16, 126, 235, 255], dtype=np.int64) * step
    if dtype == np.uint16:
        codes[-1] = 65535
    data = np.repeat(codes.astype(dtype)[None, :, None], 3, axis=2)
    tifffile.imwrite(tmp_path / "v.tif", data, photometric="rgb")
    video = read_image(tmp_path / "v.tif", "video")[0, 0]
    assert video[1] == 0.0 and video[3] == 1.0
    assert np.isclose(video[2], 110 / 219)
    assert video[0] < 0 and video[4] > 1  # sub-black and super-white kept
    full = read_image(tmp_path / "v.tif", "full")
    assert np.array_equal(full, read_image(tmp_path / "v.tif"))
    assert np.array_equal(full[0, 0], codes / np.iinfo(dtype).max)


def test_video_range_png(tmp_path):
    from PIL import Image
    Image.fromarray(np.array([[[16] * 3, [235] * 3]], np.uint8)).save(tmp_path / "v.png")
    assert np.array_equal(read_image(tmp_path / "v.png", "video")[:, 0], [[0, 1]] * 3)
    assert np.allclose(read_image(tmp_path / "v.png")[:, 0], [[16 / 255, 235 / 255]] * 3)


def test_video_range_refuses_float_images(tmp_path):
    """A float file has no code scale to read 16..235 against: video range is refused, full range reads it as
    before."""
    from cierra.__main__ import ImageRangeError
    v = np.full((4, 4, 3), 0.5, np.float32)
    tifffile.imwrite(tmp_path / "f.tif", v, photometric="rgb")
    np.save(tmp_path / "f.npy", v.transpose(2, 0, 1).astype(np.float64))
    for name in ("f.tif", "f.npy"):
        assert np.allclose(read_image(tmp_path / name, "full"), 0.5)
        with pytest.raises(ImageRangeError, match="video"):
            read_image(tmp_path / name, "video")


def test_render_range_flag(tmp_path, capsys):
    """--range video on the command line: legal black and white come back as 0 and 255 through an identity
    program; on a float image it is one plain line on stderr and exit 1, no traceback."""
    from PIL import Image

    from cierra.__main__ import main
    data = np.repeat(np.array([16, 235], np.uint8)[None, :, None], 3, axis=2)
    tifffile.imwrite(tmp_path / "v.tif", data, photometric="rgb")
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\n")
    args = ["render", str(tmp_path / "p.cierra"), str(tmp_path / "v.tif"), str(tmp_path / "out.png")]
    assert main(args + ["--range", "video"]) in (0, None)
    out = np.asarray(Image.open(tmp_path / "out.png"), dtype=float)[0, :, 0]
    assert abs(out[0]) <= 1 and abs(out[1] - 255) <= 1
    tifffile.imwrite(tmp_path / "f.tif", np.full((2, 2, 3), 0.5, np.float32), photometric="rgb")
    capsys.readouterr()
    assert main(["render", str(tmp_path / "p.cierra"), str(tmp_path / "f.tif"), str(tmp_path / "o.png"),
                 "--range", "video"]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err and "--range video" in err


def _png16(path, data, colour_type):
    """A 16-bit PNG written by hand (Pillow cannot write 16-bit colour); data (H, W[, C]) uint16."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    h, w = data.shape[:2]
    raw = b"".join(b"\0" + row.astype(">u2").tobytes() for row in data)  # filter type 0 on every row
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, colour_type, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def test_16bit_grey_png_keeps_16_bits(tmp_path):
    _png16(tmp_path / "g.png", np.array([[0, 32768, 65535]], np.uint16), 0)
    a = read_image(tmp_path / "g.png")
    assert a.shape == (3, 1, 3) and np.allclose(a[:, 0], [[0, 32768 / 65535, 1]] * 3)
    assert np.isclose(read_image(tmp_path / "g.png", "video")[0, 0, 1], (32768 - 4096) / 56064)


def test_16bit_colour_png_is_refused_not_cut_to_8_bits(tmp_path, capsys):
    """Pillow reads 16-bit RGB PNGs as 8-bit: refused with one plain line (pointing at TIFF), never quantised."""
    from cierra.__main__ import ImageInputError, main
    _png16(tmp_path / "c.png", np.full((2, 2, 3), 32768, np.uint16), 2)
    with pytest.raises(ImageInputError, match="16-bit TIFF"):
        read_image(tmp_path / "c.png")
    (tmp_path / "p.cierra").write_text("cierra 0.1\ninput srgb_photo\noutput display_encode(eotf: srgb)\n")
    capsys.readouterr()
    assert main(["render", str(tmp_path / "p.cierra"), str(tmp_path / "c.png"), str(tmp_path / "o.png")]) == 1
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "Traceback" not in err

