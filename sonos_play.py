#!/usr/bin/env python3
"""Tell a Sonos speaker to play a stream from serve_noise.py.

    sonos_play.py 192.168.1.50 http://192.168.1.10:8000/brown.flac

The speaker is its IP address or its room name, e.g. "Bedroom". If it is in a
group or a stereo pair, the whole group plays.
"""

import argparse
import ipaddress
import sys
from typing import Any

import soco  # type: ignore[import-untyped]


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Tell a Sonos speaker to play a stream.")
    parser.add_argument("speaker", help="the speaker's IP address or room name")
    parser.add_argument("url", help="the stream, e.g. http://192.168.1.10:8000/brown.flac")
    args = parser.parse_args()

    title = args.url.rsplit("/", 1)[-1].removesuffix(".flac").replace("-", " ") + " noise"
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
        coordinator = speaker.group.coordinator
        coordinator.play_uri(args.url, title=title.capitalize(), mime_type="audio/flac")
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
