#!/usr/bin/env python3
"""Generate noise as a WAV file on stdout.

    generate_noise.py --duration 10s > output.wav
"""

import argparse
import os
import re
import struct
import sys
from typing import BinaryIO

SAMPLE_RATE = 44100  # frames per second
SAMPLE_WIDTH = 2  # bytes per sample, 16-bit signed
CHANNELS = 1
CHUNK_FRAMES = SAMPLE_RATE  # one second at a time, so memory stays flat

COLORS = ("white", "brown")

_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600}


def parse_duration(text: str) -> float:
    """Seconds from '10', '10s', '5m', '12h' or '1.5h'."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", text.strip().lower())
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid duration {text!r}: use a number with an optional s, m or h suffix"
        )
    seconds = float(match.group(1)) * _UNIT_SECONDS[match.group(2) or "s"]
    if seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be greater than zero")
    return seconds


def generate(color: str, n_frames: int) -> bytes:
    """Return n_frames of 16-bit mono PCM for the given noise color.

    Stub: returns silence whatever the color.
    """
    return bytes(n_frames * SAMPLE_WIDTH * CHANNELS)


def wav_header(total_frames: int) -> bytes:
    """A 44-byte PCM WAV header for total_frames frames.

    Written by hand rather than with the wave module, which seeks back to
    patch the header on close and so fails when stdout is a pipe.
    """
    block_align = CHANNELS * SAMPLE_WIDTH
    data_size = total_frames * block_align
    if data_size > 0xFFFFFFFF - 36:
        raise ValueError("too long for a WAV file (4 GiB limit)")
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, CHANNELS, SAMPLE_RATE, SAMPLE_RATE * block_align,
        block_align, SAMPLE_WIDTH * 8,
        b"data", data_size,
    )


def write_wav(out: BinaryIO, color: str, total_frames: int) -> None:
    out.write(wav_header(total_frames))
    remaining = total_frames
    while remaining > 0:
        n_frames = min(CHUNK_FRAMES, remaining)
        out.write(generate(color, n_frames))
        remaining -= n_frames
    out.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate noise as a WAV file on stdout.")
    parser.add_argument(
        "--duration",
        type=parse_duration,
        required=True,
        help="length of the output, e.g. 10s, 5m, 12h (a bare number is seconds)",
    )
    parser.add_argument(
        "--color",
        choices=COLORS,
        default="brown",
        help="kind of noise (default: %(default)s)",
    )
    args = parser.parse_args()

    if sys.stdout.isatty():
        parser.error("refusing to write WAV data to a terminal; redirect stdout to a file or pipe")

    try:
        write_wav(sys.stdout.buffer, args.color, round(args.duration * SAMPLE_RATE))
    except ValueError as error:
        parser.error(str(error))
    except BrokenPipeError:
        # The reader went away (e.g. `| head`). Point stdout at /dev/null so
        # the interpreter's exit-time flush does not raise a second time.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
