#!/usr/bin/env python3
"""Tell a Sonos speaker to play the streams of serve_noise.py.

    sonos_play.py 192.168.1.50 http://192.168.1.10:8000/brown.flac

The speaker's queue is replaced with every color the server has, and it
starts on the one given. Next and Previous, on the speaker or in the Sonos
app, then change color.

The speaker is its IP address or its room name, e.g. "Bedroom". If it is in a
group or a stereo pair, the whole group plays.
"""

import argparse
import ipaddress
import sys
from typing import Any
from urllib.parse import urlsplit

import soco  # type: ignore[import-untyped]

from serve_noise import STREAMS


def is_ip_address(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
    except ValueError:
        return False
    return True


def find_speaker(speaker: str) -> Any | None:
    """The soco.SoCo for an IP address or a room name, None if no room has that name."""
    if is_ip_address(speaker):
        return soco.SoCo(speaker)
    return soco.discovery.by_name(speaker)


def queue_streams(coordinator: Any, url: str) -> None:
    """Replace the queue with every stream of the server at url, and play url.

    Each stream is endless, so the queue only moves when someone skips. It
    repeats, so skipping past the last color comes round to the first.
    """
    server = urlsplit(url)
    paths = list(STREAMS)
    coordinator.clear_queue()
    for path in paths:
        coordinator.add_uri_to_queue(server._replace(path=path).geturl(), mime_type="audio/flac")
    coordinator.play_from_queue(paths.index(server.path))
    coordinator.play_mode = "REPEAT_ALL"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tell a Sonos speaker to play the streams of serve_noise.py."
    )
    parser.add_argument("speaker", help="the speaker's IP address or room name")
    parser.add_argument(
        "url", help="the stream to start on, e.g. http://192.168.1.10:8000/brown.flac"
    )
    args = parser.parse_args()
    if urlsplit(args.url).path not in STREAMS:
        paths = ", ".join(STREAMS)
        parser.error(f"{args.url} is not a stream; its path must be one of {paths}")

    try:
        speaker = find_speaker(args.speaker)
        if speaker is None:
            print(
                f"No Sonos speaker named {args.speaker!r} answered; try its IP address.",
                file=sys.stderr,
            )
            return 1
        # Only the coordinator of a group or stereo pair takes play commands,
        # and what it plays, every speaker with it plays.
        queue_streams(speaker.group.coordinator, args.url)
    except OSError as error:  # includes every network error soco lets through
        print(f"Could not reach the speaker {args.speaker!r}: {error}", file=sys.stderr)
        if sys.platform == "darwin":
            print(
                "If the speaker answers ping, macOS may be blocking this terminal: see"
                " System Settings > Privacy & Security > Local Network.",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
