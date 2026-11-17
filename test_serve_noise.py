"""Tests for serve_noise.py. Run with: python3 -m unittest"""

import http.client
import itertools
import subprocess
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pyflac  # type: ignore[import-untyped]

from noise import GENERATORS, SAMPLE_RATE, BrownNoise, Samples
from serve_noise import NoiseHandler, flac_stream, lan_address

SCRIPT = Path(__file__).with_name("serve_noise.py")


def decode(flac: bytes) -> tuple[Samples, int, int]:
    """(samples as 16-bit integers, sample rate, channels) of FLAC bytes."""
    blocks: list[Samples] = []
    formats: set[tuple[int, int]] = set()

    def on_block(audio: Samples, sample_rate: int, channels: int, _samples: int) -> None:
        blocks.append(audio.reshape(-1).copy())
        formats.add((sample_rate, channels))

    decoder = pyflac.StreamDecoder(on_block)
    decoder.process(flac)
    decoder.finish()
    (sample_rate, channels), = formats
    return np.concatenate(blocks), sample_rate, channels


class FlacStreamTest(unittest.TestCase):
    def stream(self, seconds: int, seed: int = 0) -> bytes:
        """That many seconds from the start of a brown stream."""
        chunks = flac_stream(BrownNoise(np.random.default_rng(seed)))
        return b"".join(itertools.islice(chunks, seconds))

    def test_starts_with_the_flac_marker(self) -> None:
        self.assertEqual(self.stream(1)[:4], b"fLaC")

    def test_format(self) -> None:
        _, sample_rate, channels = decode(self.stream(1))
        self.assertEqual(sample_rate, SAMPLE_RATE)
        self.assertEqual(channels, 1)

    def test_is_lossless(self) -> None:
        decoded, _, _ = decode(self.stream(3, seed=1))
        self.assertGreater(len(decoded), 2 * SAMPLE_RATE)
        original = BrownNoise(np.random.default_rng(1)).read(len(decoded))
        np.testing.assert_array_equal(decoded, np.frombuffer(original, dtype="<i2"))

    def test_never_yields_nothing(self) -> None:
        chunks = flac_stream(BrownNoise(np.random.default_rng(0)))
        self.assertTrue(all(itertools.islice(chunks, 50)))

    def test_goes_on_past_any_length_asked_of_it(self) -> None:
        # Ten minutes of audio, far more than one read of the generator.
        chunks = flac_stream(BrownNoise(np.random.default_rng(0)))
        size = sum(len(chunk) for chunk in itertools.islice(chunks, 600))
        self.assertGreater(size, 10_000_000)


class QuietHandler(NoiseHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass  # keep the request log out of the test output


class ServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method: str, path: str) -> http.client.HTTPResponse:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        connection.request(method, path)
        return connection.getresponse()

    def test_every_color_streams_flac(self) -> None:
        for color in GENERATORS:
            with self.subTest(color=color):
                response = self.request("GET", f"/{color}.flac")
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Content-Type"), "audio/flac")
                self.assertIsNone(response.getheader("Content-Length"))
                decoded, sample_rate, _ = decode(response.read(200_000))
                self.assertEqual(sample_rate, SAMPLE_RATE)
                self.assertGreater(decoded.std(), 0, "silence")

    def test_two_listeners_get_different_noise(self) -> None:
        first = self.request("GET", "/white.flac").read(100_000)
        second = self.request("GET", "/white.flac").read(100_000)
        self.assertNotEqual(first, second)

    def test_head_has_the_headers_and_no_body(self) -> None:
        response = self.request("HEAD", "/brown.flac")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "audio/flac")
        self.assertEqual(response.read(), b"")

    def test_a_query_string_is_ignored(self) -> None:
        response = self.request("GET", "/brown.flac?from=sonos")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.read(4), b"fLaC")

    def test_unknown_paths_are_not_found(self) -> None:
        paths = ("/", "/purple.flac", "/brown", "/brown.mp3", "/x/brown.flac", "/.flac")
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path).status, 404)

    def test_serves_the_next_listener_after_one_hangs_up(self) -> None:
        response = self.request("GET", "/brown.flac")
        response.read(10_000)
        response.close()
        self.assertEqual(self.request("GET", "/brown.flac").read(4), b"fLaC")


class CliTest(unittest.TestCase):
    def start(self, *args: str) -> list[str]:
        """Start serve_noise.py on a free port; what it prints before serving."""
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--port", "0", *args],
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(process.wait)
        self.addCleanup(process.terminate)
        assert process.stderr is not None
        self.addCleanup(process.stderr.close)
        return [process.stderr.readline().strip() for _ in range(1 + len(GENERATORS))]

    def test_says_where_it_listens_then_lists_the_streams(self) -> None:
        listening, *urls = self.start("--bind", "127.0.0.1")
        self.assertRegex(listening, r"^Listening on 127\.0\.0\.1:\d+$")
        port = listening.rsplit(":", 1)[1]
        self.assertEqual(urls, [f"http://127.0.0.1:{port}/{color}.flac" for color in GENERATORS])

    def test_serves_on_the_address_it_printed(self) -> None:
        _, url, *_ = self.start("--bind", "127.0.0.1")
        host, port = url.split("/")[2].split(":")
        connection = http.client.HTTPConnection(host, int(port), timeout=10)
        self.addCleanup(connection.close)
        connection.request("GET", "/" + url.split("/", 3)[3])
        self.assertEqual(connection.getresponse().read(4), b"fLaC")

    def test_listens_on_every_address_by_default(self) -> None:
        listening, url, *_ = self.start()
        self.assertRegex(listening, r"^Listening on 0\.0\.0\.0:\d+$")
        self.assertTrue(url.startswith(f"http://{lan_address()}:"), url)

    def test_an_address_it_cannot_listen_on_is_a_usage_error(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--bind", "192.0.2.1"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot listen on 192.0.2.1:8000", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
