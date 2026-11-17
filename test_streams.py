"""Tests for streams.py. Run with: python3 -m unittest"""

import unittest

from noise import GENERATORS, DeepBrownNoise
from streams import COVERS, STREAMS, cover_path, stream_path


class StreamsTest(unittest.TestCase):
    def test_path_of_a_color(self) -> None:
        self.assertEqual(stream_path("deep-brown"), "/deep-brown.flac")

    def test_every_color_has_a_stream_in_order(self) -> None:
        self.assertEqual(list(STREAMS), [stream_path(color) for color in GENERATORS])

    def test_a_path_leads_to_the_generator_of_its_color(self) -> None:
        self.assertIs(STREAMS["/deep-brown.flac"], DeepBrownNoise)

    def test_path_of_the_cover_of_a_color(self) -> None:
        self.assertEqual(cover_path("deep-brown"), "/deep-brown.png")

    def test_every_color_has_a_cover(self) -> None:
        self.assertEqual(COVERS, {cover_path(color): color for color in GENERATORS})


if __name__ == "__main__":
    unittest.main()
