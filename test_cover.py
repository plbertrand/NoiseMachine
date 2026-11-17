"""Tests for cover.py. Run with: python3 -m unittest"""

import struct
import unittest
import zlib

import numpy as np

from cover import LOOKS, SHADES, SIZE, cover, palette, png, texture
from noise import GENERATORS, Samples

SIGNATURE = b"\x89PNG\r\n\x1a\n"


def chunks(image: bytes) -> list[tuple[bytes, bytes]]:
    """The (kind, data) of each chunk of a PNG, with its checksum verified."""
    found = []
    position = len(SIGNATURE)
    while position < len(image):
        (length,) = struct.unpack_from(">I", image, position)
        kind = image[position + 4 : position + 8]
        data = image[position + 8 : position + 8 + length]
        (checksum,) = struct.unpack_from(">I", image, position + 8 + length)
        assert checksum == zlib.crc32(kind + data), kind
        found.append((kind, data))
        position += 12 + length
    return found


class TextureTest(unittest.TestCase):
    def make(self, falloff: float) -> Samples:
        return texture(falloff, np.random.default_rng(0))

    def test_is_a_square_from_0_to_1(self) -> None:
        for falloff in (0.0, 1.0, 3.0):
            with self.subTest(falloff=falloff):
                image = self.make(falloff)
                self.assertEqual(image.shape, (SIZE, SIZE))
                self.assertEqual(image.min(), 0.0)
                self.assertEqual(image.max(), 1.0)

    def test_a_steeper_falloff_is_smoother(self) -> None:
        # Roughness: how much a pixel differs from the one beside it.
        roughness = [
            float(np.abs(np.diff(self.make(falloff), axis=1)).mean())
            for falloff in (0.0, 1.0, 2.0, 3.0)
        ]
        self.assertEqual(roughness, sorted(roughness, reverse=True))
        self.assertGreater(roughness[0], 20 * roughness[-1])

    def test_tiles_seamlessly(self) -> None:
        # The step across the edge is no bigger than the steps inside.
        image = self.make(2.0)
        across = np.abs(image[:, 0] - image[:, -1]).mean()
        inside = np.abs(np.diff(image, axis=1)).mean()
        self.assertLess(across, 2 * inside)


class PaletteTest(unittest.TestCase):
    def test_runs_from_dark_to_light(self) -> None:
        colors = palette((10, 20, 30), (200, 100, 50))
        self.assertEqual(len(colors), 3 * SHADES)
        self.assertEqual(tuple(colors[:3]), (10, 20, 30))
        self.assertEqual(tuple(colors[-3:]), (200, 100, 50))


class PngTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        indices = np.arange(6, dtype=np.uint8).reshape(2, 3)
        image = png(indices, palette((0, 0, 0), (255, 255, 255)))
        self.assertEqual(image[:8], SIGNATURE)
        (header, ihdr), (_, plte), (_, idat), (last, _) = chunks(image)
        self.assertEqual((header, last), (b"IHDR", b"IEND"))
        # Width 3, height 2, 8 bits, color type 3: a palette.
        self.assertEqual(struct.unpack(">IIBBBBB", ihdr), (3, 2, 8, 3, 0, 0, 0))
        self.assertEqual(len(plte), 3 * SHADES)
        # Each row is a filter byte, 0, then its pixels.
        self.assertEqual(zlib.decompress(idat), bytes([0, 0, 1, 2, 0, 3, 4, 5]))


class CoverTest(unittest.TestCase):
    def test_every_color_has_a_look(self) -> None:
        self.assertEqual(list(LOOKS), list(GENERATORS))

    def test_is_a_square_png(self) -> None:
        for color in LOOKS:
            with self.subTest(color=color):
                image = cover(color)
                self.assertEqual(image[:8], SIGNATURE)
                kinds = [kind for kind, _ in chunks(image)]
                self.assertEqual(kinds, [b"IHDR", b"PLTE", b"IDAT", b"IEND"])
                width, height = struct.unpack_from(">II", chunks(image)[0][1])
                self.assertEqual((width, height), (SIZE, SIZE))

    def test_is_the_same_every_time(self) -> None:
        cover.cache_clear()
        first = cover("brown")
        cover.cache_clear()
        self.assertEqual(cover("brown"), first)

    def test_colors_differ(self) -> None:
        self.assertEqual(len({cover(color) for color in LOOKS}), len(LOOKS))


if __name__ == "__main__":
    unittest.main()
