"""Control of Sonos speakers over HTTP, for serve_noise.py.

    POST /api/play/brown?speaker=Bedroom   play that color on the speaker
    POST /api/stop?speaker=Bedroom         stop the noise; other music is left playing
    GET  /api/status?speaker=Bedroom       {"playing": "brown", "position": "8:04:21"}

The speaker is its IP address or its room name.

Only a POST changes what the speaker does, so that nothing which merely
fetches a URL, like a browser guessing the next page, can start the noise.
"""

from collections.abc import Callable
from http import HTTPStatus
from typing import Any, NamedTuple
from urllib.parse import parse_qs, urlsplit

from soco.exceptions import SoCoException  # type: ignore[import-untyped]

from noise import GENERATORS
import sonos_play
import streams

Status = dict[str, str | None]
ServerUrl = Callable[[str], str]  # where the speaker at that IP address finds the streams

PLAYING_STATES = ("PLAYING", "TRANSITIONING")
SPEAKER = "speaker"  # the query parameter that says which speaker


class SpeakerNotFound(OSError):
    pass


class Machine:
    """A Sonos speaker, and the server whose streams it is told to play."""

    def __init__(self, speaker: str, server_url: ServerUrl) -> None:
        self.speaker = speaker
        self.server_url = server_url

    def urls(self, coordinator: Any) -> dict[str, str]:
        """The stream of each color, at the address coordinator can reach.

        The coordinator is the speaker that fetches the stream, and a server
        on several networks is at a different address on each.
        """
        server = self.server_url(coordinator.ip_address)
        return {color: server + streams.stream_path(color) for color in GENERATORS}

    def coordinator(self) -> Any:
        """The speaker that takes the commands of the speaker's group.

        Looked up every time: groups change, and so can the coordinator.
        """
        speaker = sonos_play.find_speaker(self.speaker)
        if speaker is None:
            raise SpeakerNotFound(f"no Sonos speaker named {self.speaker!r} answered")
        return speaker.group.coordinator

    def play(self, color: str) -> Status:
        coordinator = self.coordinator()
        sonos_play.queue_streams(coordinator, self.urls(coordinator)[color])
        return {"playing": color}

    def stop(self) -> Status:
        """Stop the speaker if it is playing a color; other music is left alone."""
        coordinator = self.coordinator()
        if self._playing(coordinator)["playing"]:
            coordinator.stop()
        return {"playing": None}

    def status(self) -> Status:
        """The color the speaker is playing, and for how long it has been.

        The speaker is asked, since it can be skipped, stopped or given other
        music without this server hearing of it.
        """
        return self._playing(self.coordinator())

    def _playing(self, coordinator: Any) -> Status:
        state = coordinator.get_current_transport_info()["current_transport_state"]
        track = coordinator.get_current_track_info()
        colors = {url: color for color, url in self.urls(coordinator).items()}
        color = colors.get(track["uri"]) if state in PLAYING_STATES else None
        return {"playing": color, "position": track["position"] if color else None}


Action = Callable[[Machine], Status]


def play(color: str) -> Action:
    return lambda machine: machine.play(color)


ROUTES: dict[str, tuple[str, Action]] = {
    "/api/status": ("GET", lambda machine: machine.status()),
    "/api/stop": ("POST", lambda machine: machine.stop()),
    **{f"/api/play/{color}": ("POST", play(color)) for color in GENERATORS},
}


class Reply(NamedTuple):
    code: HTTPStatus
    content: Status
    allow: str | None = None  # the method to use instead, when the code is 405


def control(server_url: ServerUrl, method: str, url: str) -> Reply | None:
    """Do what a request asks of a speaker; None if url is not one of ROUTES."""
    request = urlsplit(url)
    route = ROUTES.get(request.path)
    if route is None:
        return None
    allowed, action = route
    if method != allowed:
        return Reply(HTTPStatus.METHOD_NOT_ALLOWED, {"error": f"use {allowed}"}, allowed)
    speakers = parse_qs(request.query).get(SPEAKER, [])
    if len(speakers) != 1:
        error = f"say which speaker: ?{SPEAKER}=<IP address or room name>"
        return Reply(HTTPStatus.BAD_REQUEST, {"error": error})
    try:
        return Reply(HTTPStatus.OK, action(Machine(speakers[0], server_url)))
    except (OSError, SoCoException) as error:  # the speaker did not do it
        return Reply(HTTPStatus.BAD_GATEWAY, {"error": str(error)})
