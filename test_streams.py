"""Tests for streams.py. Run with: python3 -m unittest"""

import unittest

from noise import GENERATORS, DeepBrownNoise
from streams import STREAMS, stream_path


class StreamsTest(unittest.TestCase):
    def test_path_of_a_color(self) -> None:
        self.assertEqual(stream_path("deep-brown"), "/deep-brown.flac")

    def test_every_color_has_a_stream_in_order(self) -> None:
        self.assertEqual(list(STREAMS), [stream_path(color) for color in GENERATORS])

    def test_a_path_leads_to_the_generator_of_its_color(self) -> None:
        self.assertIs(STREAMS["/deep-brown.flac"], DeepBrownNoise)


if __name__ == "__main__":
    unittest.main()
