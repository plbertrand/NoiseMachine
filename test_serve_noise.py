"""Tests for serve_noise.py. Run with: python3 -m unittest

Every server here listens on 127.0.0.1 only, so nothing on the network can
reach one while the tests run.
"""

import http.client
import itertools
import json
import socket
import subprocess
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import numpy as np
import pyflac  # type: ignore[import-untyped]

from cover import cover
from noise import GENERATORS, SAMPLE_RATE, BrownNoise, Samples
from serve_noise import (
    NoiseHandler,
    argument_parser,
    flac_stream,
    lan_address,
    server_url,
    url_host,
)

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


class ServerCase(unittest.TestCase):
    """A server on a free local port, answering with handler."""

    handler: type[NoiseHandler] = QuietHandler

    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def request(self, method: str, path: str) -> http.client.HTTPResponse:
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        self.addCleanup(connection.close)
        connection.request(method, path)
        return connection.getresponse()


class ServerTest(ServerCase):
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

    def test_every_color_has_a_cover_image(self) -> None:
        for color in GENERATORS:
            with self.subTest(color=color):
                response = self.request("GET", f"/{color}.png")
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("Content-Type"), "image/png")
                body = response.read()
                self.assertEqual(response.getheader("Content-Length"), str(len(body)))
                self.assertEqual(body, cover(color))

    def test_head_of_a_cover_has_its_length_and_no_body(self) -> None:
        response = self.request("HEAD", "/brown.png")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Length"), str(len(cover("brown"))))
        self.assertEqual(response.read(), b"")

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
        paths = ["/", "/purple.flac", "/brown", "/brown.mp3", "/x/brown.flac", "/.flac"]
        paths += ["/purple.png", "/.png", "/brown.jpg"]
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(self.request("GET", path).status, 404)

    def test_serves_the_next_listener_after_one_hangs_up(self) -> None:
        response = self.request("GET", "/brown.flac")
        response.read(10_000)
        response.close()
        self.assertEqual(self.request("GET", "/brown.flac").read(4), b"fLaC")

    def test_a_control_route_without_a_speaker_is_a_bad_request(self) -> None:
        for method, path in (("GET", "/api/status"), ("POST", "/api/stop")):
            with self.subTest(path=path):
                self.assertEqual(self.request(method, path).status, 400)

    def test_a_stream_is_not_posted_to(self) -> None:
        self.assertEqual(self.request("POST", "/brown.flac").status, 404)


class SpeakerTest(ServerCase):
    """The control routes, with a mock for the speaker."""

    def setUp(self) -> None:
        patcher = mock.patch("control.Machine", autospec=True)
        self.machines = patcher.start()
        self.addCleanup(patcher.stop)
        self.machine = self.machines.return_value
        self.machine.status.return_value = {"playing": "brown", "position": "0:01:02"}
        self.machine.stop.return_value = {"playing": None}
        super().setUp()

    def test_the_speaker_is_sent_to_the_address_the_server_listens_on(self) -> None:
        self.request("GET", "/api/status?speaker=Living%20Room").read()
        (speaker, url_for), _ = self.machines.call_args
        self.assertEqual(speaker, "Living Room")
        self.assertEqual(url_for("192.168.1.50"), f"http://127.0.0.1:{self.server.server_port}")

    def test_status_is_json(self) -> None:
        response = self.request("GET", "/api/status?speaker=Bedroom")
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "application/json")
        self.assertEqual(json.loads(response.read()), {"playing": "brown", "position": "0:01:02"})

    def test_a_post_reaches_the_speaker(self) -> None:
        response = self.request("POST", "/api/stop?speaker=Bedroom")
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(response.read()), {"playing": None})
        self.machine.stop.assert_called_once_with()

    def test_a_get_does_not_and_says_to_post(self) -> None:
        response = self.request("GET", "/api/play/brown?speaker=Bedroom")
        self.assertEqual(response.status, 405)
        self.assertEqual(response.getheader("Allow"), "POST")
        self.machine.play.assert_not_called()

    def test_a_speaker_that_fails_is_a_bad_gateway(self) -> None:
        self.machine.stop.side_effect = OSError(65, "No route to host")
        response = self.request("POST", "/api/stop?speaker=Bedroom")
        self.assertEqual(response.status, 502)
        self.assertIn("No route to host", json.loads(response.read())["error"])

    def test_the_streams_are_still_served(self) -> None:
        self.assertEqual(self.request("GET", "/brown.flac").read(4), b"fLaC")


class CliTest(unittest.TestCase):
    def start(self, lines: int = 1 + len(GENERATORS)) -> list[str]:
        """Start serve_noise.py on a free local port; what it prints before serving."""
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--bind", "127.0.0.1", "--port", "0"],
            stderr=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(process.wait)
        self.addCleanup(process.terminate)
        assert process.stderr is not None
        self.addCleanup(process.stderr.close)
        return [process.stderr.readline().strip() for _ in range(lines)]

    def test_lists_the_control_routes_after_the_streams(self) -> None:
        lines = self.start(lines=3 + len(GENERATORS))
        port = lines[0].rsplit(":", 1)[1]
        self.assertIn("Sonos speaker", lines[-2])
        self.assertEqual(lines[-1], f"GET http://127.0.0.1:{port}/api/status?speaker=<speaker>")

    def test_says_where_it_listens_then_lists_the_streams(self) -> None:
        listening, *urls = self.start()
        self.assertRegex(listening, r"^Listening on 127\.0\.0\.1:\d+$")
        port = listening.rsplit(":", 1)[1]
        self.assertEqual(urls, [f"http://127.0.0.1:{port}/{color}.flac" for color in GENERATORS])

    def test_serves_on_the_address_it_printed(self) -> None:
        _, url, *_ = self.start()
        host, port = url.split("/")[2].split(":")
        connection = http.client.HTTPConnection(host, int(port), timeout=10)
        self.addCleanup(connection.close)
        connection.request("GET", "/" + url.split("/", 3)[3])
        self.assertEqual(connection.getresponse().read(4), b"fLaC")

    def test_a_port_it_cannot_listen_on_is_a_usage_error(self) -> None:
        with socket.socket() as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen()
            port = taken.getsockname()[1]
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--bind", "127.0.0.1", "--port", str(port)],
                capture_output=True,
                text=True,
                timeout=30,
            )
        self.assertEqual(result.returncode, 2)
        self.assertIn(f"cannot listen on 127.0.0.1:{port}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)


class ArgumentsTest(unittest.TestCase):
    """The defaults, checked without starting a server on them."""

    def test_listens_on_every_address_by_default(self) -> None:
        self.assertEqual(argument_parser().parse_args([]).bind, "0.0.0.0")

    def test_default_port(self) -> None:
        self.assertEqual(argument_parser().parse_args([]).port, 8000)

    def test_urls_use_the_lan_address_when_listening_on_every_address(self) -> None:
        with mock.patch("serve_noise.lan_address", return_value="192.168.1.10"):
            self.assertEqual(url_host("0.0.0.0"), "192.168.1.10")

    def test_urls_use_the_address_listened_on_otherwise(self) -> None:
        with mock.patch("serve_noise.lan_address") as lan_address:
            self.assertEqual(url_host("127.0.0.1"), "127.0.0.1")
            self.assertEqual(url_host("127.0.0.1", towards="192.168.2.50"), "127.0.0.1")
        lan_address.assert_not_called()

    def test_a_speaker_gets_the_address_on_the_network_that_leads_to_it(self) -> None:
        addresses = {"192.168.1.50": "192.168.1.10", "192.168.2.50": "192.168.2.10"}
        with mock.patch("serve_noise.lan_address", side_effect=addresses.get):
            for speaker, url in (
                ("192.168.1.50", "http://192.168.1.10:8000"),
                ("192.168.2.50", "http://192.168.2.10:8000"),
            ):
                self.assertEqual(server_url("0.0.0.0", 8000, towards=speaker), url)

    def test_the_address_that_leads_to_this_host_is_this_host(self) -> None:
        # The one route every host has; nothing is sent to find it.
        self.assertEqual(lan_address(towards="127.0.0.1"), "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
