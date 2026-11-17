"""A cover image for each color of noise: a picture of that noise.

Each cover is a square of two-dimensional noise in shades of its color. The
darker the color sounds, the faster its power falls with frequency, and the
smoother its picture: white is static, deep brown is clouds.
"""

import struct
import zlib
from functools import cache
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from noise import GENERATORS, Samples

SIZE = 512  # pixels a side
SHADES = 256  # the most a PNG palette holds

Rgb = tuple[int, int, int]


class Look(NamedTuple):
    """How a color is drawn."""

    falloff: float  # the amplitude at frequency f is 1 / f**falloff
    dark: Rgb  # the shade of the lowest values
    light: Rgb  # the shade of the highest


LOOKS: dict[str, Look] = {
    "white": Look(0.0, (24, 24, 28), (255, 255, 255)),
    "pink": Look(1.0, (110, 16, 64), (255, 196, 220)),
    "brown": Look(2.0, (52, 30, 20), (190, 140, 100)),  # walnut
    "deep-brown": Look(3.0, (10, 5, 2), (112, 62, 28)),
}
assert LOOKS.keys() == GENERATORS.keys()


def texture(falloff: float, rng: np.random.Generator) -> Samples:
    """A SIZE by SIZE square of noise, scaled to run from 0 to 1.

    White noise, with each frequency's amplitude divided by the frequency to
    the power falloff. Built in the frequency domain, so it tiles seamlessly.
    """
    spectrum = np.fft.rfft2(rng.standard_normal((SIZE, SIZE)))
    frequency = np.hypot(np.fft.fftfreq(SIZE)[:, None], np.fft.rfftfreq(SIZE)[None, :])
    frequency[0, 0] = 1.0  # the average level, removed below, has no frequency
    spectrum /= frequency**falloff
    spectrum[0, 0] = 0.0
    image = np.fft.irfft2(spectrum, s=(SIZE, SIZE))
    # The extreme one percent at each end is clipped, to use the whole range.
    low, high = np.percentile(image, (1, 99))
    scaled: Samples = np.clip((image - low) / (high - low), 0.0, 1.0)
    return scaled


def palette(dark: Rgb, light: Rgb) -> bytes:
    """SHADES colors from dark to light, as PNG palette data."""
    steps = np.linspace(0.0, 1.0, SHADES)[:, None]
    shades = np.array(dark) + steps * (np.array(light) - np.array(dark))
    return bytes(shades.round().astype(np.uint8))


def png(indices: npt.NDArray[np.uint8], colors: bytes) -> bytes:
    """A PNG of palette indices, one byte a pixel, and the palette they index."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        checksum = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)

    height, width = indices.shape
    # Each row starts with a byte naming its filter; 0 is "none".
    rows = np.insert(indices, 0, 0, axis=1).tobytes()
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 3, 0, 0, 0))
        + chunk(b"PLTE", colors)
        + chunk(b"IDAT", zlib.compress(rows, 9))
        + chunk(b"IEND", b"")
    )


@cache
def cover(color: str) -> bytes:
    """The cover of a color, as a PNG. The same picture every time."""
    look = LOOKS[color]
    # Seeded by the color's place in the list, so each has its own pattern.
    rng = np.random.default_rng(list(LOOKS).index(color))
    indices = (texture(look.falloff, rng) * (SHADES - 1)).round().astype(np.uint8)
    return png(indices, palette(look.dark, look.light))
