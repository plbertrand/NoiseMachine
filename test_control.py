"""Tests for control.py. Run with: python3 -m unittest

No speaker is contacted: the speaker is a mock wherever one is looked up.
"""

import unittest
from http import HTTPStatus
from unittest import mock

from soco.exceptions import SoCoException  # type: ignore[import-untyped]

import control
from control import ROUTES, Machine, Reply, SpeakerNotFound
from noise import GENERATORS

SERVER = "http://192.168.1.10:8000"
SPEAKER_ADDRESS = "192.168.1.50"


def server_url(speaker: str) -> str:
    return SERVER


@mock.patch("sonos_play.find_speaker")
class MachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server_url = mock.Mock(side_effect=server_url)
        self.machine = Machine("Bedroom", self.server_url)

    def speaker_playing(
        self, find_speaker: mock.Mock, state: str, uri: str, position: str = "0:01:02"
    ) -> None:
        coordinator = find_speaker.return_value.group.coordinator
        coordinator.get_current_transport_info.return_value = {"current_transport_state": state}
        coordinator.get_current_track_info.return_value = {"uri": uri, "position": position}

    def test_looks_the_speaker_up_by_what_it_was_given(self, find_speaker: mock.Mock) -> None:
        self.machine.play("brown")
        find_speaker.assert_called_once_with("Bedroom")

    def test_play_queues_the_streams_and_starts_on_the_color(
        self, find_speaker: mock.Mock
    ) -> None:
        self.assertEqual(self.machine.play("brown"), {"playing": "brown"})
        coordinator = find_speaker.return_value.group.coordinator
        queued = [call.args[0] for call in coordinator.add_to_queue.call_args_list]
        self.assertEqual(
            [item.resources[0].uri for item in queued],
            [f"{SERVER}/{color}.flac" for color in GENERATORS],
        )
        coordinator.play_from_queue.assert_called_once_with(list(GENERATORS).index("brown"))

    def test_commands_go_to_the_coordinator_of_the_group(self, find_speaker: mock.Mock) -> None:
        self.machine.play("brown")
        self.speaker_playing(find_speaker, "PLAYING", f"{SERVER}/brown.flac")
        self.machine.stop()
        find_speaker.return_value.play_from_queue.assert_not_called()
        find_speaker.return_value.stop.assert_not_called()

    def test_the_streams_are_at_the_address_the_coordinator_can_reach(
        self, find_speaker: mock.Mock
    ) -> None:
        find_speaker.return_value.group.coordinator.ip_address = SPEAKER_ADDRESS
        self.speaker_playing(find_speaker, "PLAYING", f"{SERVER}/brown.flac")
        self.machine.play("brown")
        self.server_url.assert_called_once_with(SPEAKER_ADDRESS)
        self.server_url.reset_mock()
        self.machine.status()
        self.server_url.assert_called_once_with(SPEAKER_ADDRESS)

    def test_stop_stops_a_speaker_playing_a_color(self, find_speaker: mock.Mock) -> None:
        self.speaker_playing(find_speaker, "PLAYING", f"{SERVER}/brown.flac")
        self.assertEqual(self.machine.stop(), {"playing": None})
        find_speaker.return_value.group.coordinator.stop.assert_called_once_with()

    def test_stop_leaves_other_music_playing(self, find_speaker: mock.Mock) -> None:
        for uri in ("x-sonos-spotify:track", "http://192.168.1.99:8000/brown.flac"):
            with self.subTest(uri=uri):
                self.speaker_playing(find_speaker, "PLAYING", uri)
                self.assertEqual(self.machine.stop(), {"playing": None})
        find_speaker.return_value.group.coordinator.stop.assert_not_called()

    def test_stop_does_nothing_to_a_speaker_already_stopped(
        self, find_speaker: mock.Mock
    ) -> None:
        self.speaker_playing(find_speaker, "STOPPED", f"{SERVER}/brown.flac")
        self.assertEqual(self.machine.stop(), {"playing": None})
        find_speaker.return_value.group.coordinator.stop.assert_not_called()

    def test_status_of_a_speaker_playing_a_color(self, find_speaker: mock.Mock) -> None:
        for state in ("PLAYING", "TRANSITIONING"):
            with self.subTest(state=state):
                self.speaker_playing(find_speaker, state, f"{SERVER}/deep-brown.flac", "8:04:21")
                self.assertEqual(
                    self.machine.status(), {"playing": "deep-brown", "position": "8:04:21"}
                )

    def test_status_of_a_speaker_stopped_or_paused_on_a_color(
        self, find_speaker: mock.Mock
    ) -> None:
        for state in ("STOPPED", "PAUSED_PLAYBACK"):
            with self.subTest(state=state):
                self.speaker_playing(find_speaker, state, f"{SERVER}/brown.flac")
                self.assertEqual(self.machine.status(), {"playing": None, "position": None})

    def test_status_of_a_speaker_playing_something_else(self, find_speaker: mock.Mock) -> None:
        for uri in ("x-sonos-spotify:track", "http://192.168.1.99:8000/brown.flac", ""):
            with self.subTest(uri=uri):
                self.speaker_playing(find_speaker, "PLAYING", uri)
                self.assertEqual(self.machine.status(), {"playing": None, "position": None})

    def test_status_changes_nothing_on_the_speaker(self, find_speaker: mock.Mock) -> None:
        self.speaker_playing(find_speaker, "PLAYING", f"{SERVER}/brown.flac")
        self.machine.status()
        coordinator = find_speaker.return_value.group.coordinator
        self.assertEqual(
            {call[0] for call in coordinator.method_calls},
            {"get_current_transport_info", "get_current_track_info"},
        )

    def test_a_name_no_speaker_answers_to(self, find_speaker: mock.Mock) -> None:
        find_speaker.return_value = None
        with self.assertRaisesRegex(SpeakerNotFound, "'Bedroom'"):
            self.machine.status()


class ControlTest(unittest.TestCase):
    def setUp(self) -> None:
        patcher = mock.patch("control.Machine", autospec=True)
        self.machines = patcher.start()
        self.addCleanup(patcher.stop)
        self.machine = self.machines.return_value
        self.machine.status.return_value = {"playing": "brown", "position": "0:01:02"}
        self.machine.play.side_effect = lambda color: {"playing": color}
        self.machine.stop.return_value = {"playing": None}

    def control(self, method: str, path: str, query: str = "?speaker=Bedroom") -> Reply | None:
        return control.control(server_url, method, path + query)

    def test_the_speaker_is_the_one_in_the_query(self) -> None:
        for query, speaker in (
            ("?speaker=Bedroom", "Bedroom"),
            ("?speaker=192.168.1.50", "192.168.1.50"),
            ("?speaker=Living%20Room", "Living Room"),
            ("?speaker=Living+Room&x=1", "Living Room"),
        ):
            with self.subTest(query=query):
                self.control("GET", "/api/status", query)
                self.machines.assert_called_with(speaker, server_url)

    def test_without_exactly_one_speaker_is_a_bad_request(self) -> None:
        for query in ("", "?speaker=", "?room=Bedroom", "?speaker=Bedroom&speaker=Kitchen"):
            with self.subTest(query=query):
                reply = self.control("POST", "/api/play/brown", query)
                assert reply is not None
                self.assertEqual(reply.code, HTTPStatus.BAD_REQUEST)
                self.assertIn("?speaker=", str(reply.content["error"]))
        self.machines.assert_not_called()

    def test_status(self) -> None:
        self.assertEqual(
            self.control("GET", "/api/status"),
            Reply(HTTPStatus.OK, {"playing": "brown", "position": "0:01:02"}),
        )

    def test_play_every_color(self) -> None:
        for color in GENERATORS:
            with self.subTest(color=color):
                reply = self.control("POST", f"/api/play/{color}")
                self.assertEqual(reply, Reply(HTTPStatus.OK, {"playing": color}))
                self.machine.play.assert_called_with(color)

    def test_stop(self) -> None:
        reply = self.control("POST", "/api/stop")
        self.assertEqual(reply, Reply(HTTPStatus.OK, {"playing": None}))
        self.machine.stop.assert_called_once_with()

    def test_a_get_never_changes_what_the_speaker_does(self) -> None:
        for path in ("/api/stop", *(f"/api/play/{color}" for color in GENERATORS)):
            with self.subTest(path=path):
                reply = self.control("GET", path)
                assert reply is not None
                self.assertEqual(reply.code, HTTPStatus.METHOD_NOT_ALLOWED)
                self.assertEqual(reply.allow, "POST")
        self.machine.play.assert_not_called()
        self.machine.stop.assert_not_called()

    def test_every_route_that_changes_the_speaker_is_a_post(self) -> None:
        self.assertEqual(
            {path for path, (method, _) in ROUTES.items() if method == "GET"}, {"/api/status"}
        )

    def test_status_is_not_posted(self) -> None:
        reply = self.control("POST", "/api/status")
        assert reply is not None
        self.assertEqual(reply.code, HTTPStatus.METHOD_NOT_ALLOWED)
        self.assertEqual(reply.allow, "GET")
        self.machine.status.assert_not_called()

    def test_other_paths_are_not_for_the_speaker(self) -> None:
        for method in ("GET", "POST"):
            for path in ("/api", "/api/play", "/api/play/purple", "/api/play/brown/x", "/"):
                with self.subTest(method=method, path=path):
                    self.assertIsNone(self.control(method, path))
        self.assertEqual(self.machine.method_calls, [])

    def test_a_speaker_that_fails_is_a_bad_gateway(self) -> None:
        for error in (OSError(65, "No route to host"), SoCoException("not the coordinator")):
            with self.subTest(error=error):
                self.machine.stop.side_effect = error
                reply = self.control("POST", "/api/stop")
                assert reply is not None
                self.assertEqual(reply.code, HTTPStatus.BAD_GATEWAY)
                self.assertIn(str(error), str(reply.content["error"]))


if __name__ == "__main__":
    unittest.main()
