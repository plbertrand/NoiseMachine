"""Tests for sonos_play.py. Run with: python3 -m unittest"""

import contextlib
import io
import unittest
from unittest import mock

import sonos_play
from serve_noise import STREAMS

URL = "http://192.168.1.10:8000/deep-brown.flac"


def run(*args: str) -> tuple[int, str]:
    """(exit status, stderr) of sonos_play.py run with those arguments."""
    stderr = io.StringIO()
    with mock.patch("sys.argv", ["sonos_play.py", *args]), contextlib.redirect_stderr(stderr):
        return sonos_play.main(), stderr.getvalue()


class IsIpAddressTest(unittest.TestCase):
    def test_addresses(self) -> None:
        for text in ("192.168.1.50", "10.0.0.1", "fe80::1"):
            with self.subTest(text=text):
                self.assertTrue(sonos_play.is_ip_address(text))

    def test_names(self) -> None:
        for text in ("Bedroom", "Move", "192.168.1", "192.168.1.256", ""):
            with self.subTest(text=text):
                self.assertFalse(sonos_play.is_ip_address(text))


SERVER = "http://192.168.1.10:8000"


@mock.patch("sonos_play.soco")
class MainTest(unittest.TestCase):
    def test_plays_on_a_speaker_given_by_address(self, soco: mock.Mock) -> None:
        status, _ = run("192.168.1.50", URL)
        self.assertEqual(status, 0)
        soco.SoCo.assert_called_once_with("192.168.1.50")
        soco.discovery.by_name.assert_not_called()
        soco.SoCo.return_value.group.coordinator.play_from_queue.assert_called_once()

    def test_plays_on_a_speaker_given_by_name(self, soco: mock.Mock) -> None:
        status, _ = run("Bedroom", URL)
        self.assertEqual(status, 0)
        soco.discovery.by_name.assert_called_once_with("Bedroom")
        coordinator = soco.discovery.by_name.return_value.group.coordinator
        coordinator.play_from_queue.assert_called_once()

    def test_plays_on_the_coordinator_of_the_speakers_group(self, soco: mock.Mock) -> None:
        run("192.168.1.50", URL)
        soco.SoCo.return_value.play_from_queue.assert_not_called()
        soco.SoCo.return_value.clear_queue.assert_not_called()
        soco.SoCo.return_value.group.coordinator.play_from_queue.assert_called_once()

    def test_queues_every_stream_in_order_after_clearing(self, soco: mock.Mock) -> None:
        run("192.168.1.50", URL)
        coordinator = soco.SoCo.return_value.group.coordinator
        queued = [mock.call(f"{SERVER}{path}", mime_type="audio/flac") for path in STREAMS]
        self.assertEqual(
            coordinator.mock_calls[:1] + coordinator.add_uri_to_queue.mock_calls,
            [mock.call.clear_queue()] + queued,
        )

    def test_starts_on_the_stream_given(self, soco: mock.Mock) -> None:
        for index, path in enumerate(STREAMS):
            with self.subTest(path=path):
                soco.reset_mock()
                run("192.168.1.50", SERVER + path)
                coordinator = soco.SoCo.return_value.group.coordinator
                coordinator.play_from_queue.assert_called_once_with(index)

    def test_repeats_so_that_skipping_wraps_around(self, soco: mock.Mock) -> None:
        run("192.168.1.50", URL)
        self.assertEqual(soco.SoCo.return_value.group.coordinator.play_mode, "REPEAT_ALL")

    def test_a_url_that_is_not_a_stream_is_a_usage_error(self, soco: mock.Mock) -> None:
        for url in (SERVER, SERVER + "/purple.flac", "brown"):
            with self.subTest(url=url):
                with self.assertRaises(SystemExit) as raised:
                    run("192.168.1.50", url)
                self.assertEqual(raised.exception.code, 2)
        soco.SoCo.assert_not_called()

    def test_unreachable_speaker_fails_without_a_traceback(self, soco: mock.Mock) -> None:
        coordinator = soco.SoCo.return_value.group.coordinator
        coordinator.clear_queue.side_effect = OSError(65, "No route to host")
        status, stderr = run("192.168.1.50", URL)
        self.assertEqual(status, 1)
        self.assertIn("Could not reach the speaker '192.168.1.50'", stderr)
        self.assertIn("No route to host", stderr)

    def test_unknown_name_fails_with_a_hint(self, soco: mock.Mock) -> None:
        soco.discovery.by_name.return_value = None
        status, stderr = run("Nowhere", URL)
        self.assertEqual(status, 1)
        self.assertIn("'Nowhere'", stderr)
        self.assertIn("IP address", stderr)


if __name__ == "__main__":
    unittest.main()
