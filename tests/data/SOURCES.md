# Test frames (tests/data/frames/*.npy)

Scene-linear Rec.709, float32, (3, H, W), 480 px wide; made by `tools/make_test_frames.py`, which downloads the RAW
files from raw.pixls.us, checks each against its SHA-256 and decodes it with LibRaw (rawpy).

| file | source | licence | changes |
|---|---|---|---|
| chart.npy | synthetic (neutral ramp, chart patches from published approximate sRGB values, skin tones, primaries, hue sweeps, HDR highlights) | made for Cierra | — |
| crowd.npy | a crowd outdoors (Sony ILCE-7SM2 ARW), raw.pixls.us file 1080, SHA-256 46044fb6…a3db | CC0 | camera white balance, ACES2065-1 → linear Rec.709, median luma 0.15, box-downsampled, centre crop to 480 |
| pens.npy | coloured pens on green felt (Panasonic DMC-G6 RW2), raw.pixls.us file 1096, SHA-256 5b20c3ec…99d9 | CC0 | as crowd |
| window_0/1/2.npy | a red pot on a sill against a bright window (Sony DSC-RX100M5 ARW), raw.pixls.us file 1324, SHA-256 82a4c49c…1bce | CC0 | as crowd, then +1.5 stops; three crops stepping right by 2 % of the width (consecutive frames of one shot) |

The full URLs and hashes are in tools/make_test_frames.py. CC0 asks for no attribution; the sources are listed so the
frames can be rebuilt and checked. The golden renders derived from these frames (tests/golden/cierra, tests/golden/dctl,
tests/golden/origin and the analysis snapshot) are made from them and from Cierra or the looks' original engine.

chart_photo.npy (in tests/data) is the synthetic chart encoded as an sRGB photo (no third-party content).
