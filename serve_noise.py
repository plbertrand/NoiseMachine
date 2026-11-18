#!/usr/bin/env python3
"""Serve endless noise as FLAC over HTTP, for a network speaker to play.

    serve_noise.py --port 8000

Each color is a stream that never ends, e.g. http://<this host>:8000/brown.flac
"""

import argparse
import socket
import sys
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
import pyflac  # type: ignore[import-untyped]

from noise import SAMPLE_RATE, Noise
from streams import STREAMS

CHUNK_FRAMES = SAMPLE_RATE  # noise is generated a second at a time
EVERY_ADDRESS = "0.0.0.0"


def flac_stream(noise: Noise) -> Iterator[bytes]:
    """The noise as an endless FLAC stream: the header, then frame after frame.

    FLAC needs no length up front, so unlike a WAV this can go on forever.
    """
    encoded: list[bytes] = []
    encoder = pyflac.StreamEncoder(
        sample_rate=SAMPLE_RATE,
        write_callback=lambda buffer, _bytes, _samples, _frame: encoded.append(buffer),
    )
    try:
        while True:
            encoder.process(np.frombuffer(noise.read(CHUNK_FRAMES), dtype="<i2"))
            yield b"".join(encoded)
            encoded.clear()
    finally:
        encoder.finish()


class NoiseHandler(BaseHTTPRequestHandler):
    """GET /<color>.flac streams that color until the listener hangs up."""

    def do_GET(self) -> None:
        self._respond(with_body=True)

    def do_HEAD(self) -> None:
        self._respond(with_body=False)

    def _respond(self, with_body: bool) -> None:
        generator = STREAMS.get(urlsplit(self.path).path)
        if generator is None:
            self.send_error(
                HTTPStatus.NOT_FOUND, explain=f"The streams are {', '.join(STREAMS)}"
            )
            return
        # No Content-Length: the body ends when the connection does.
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/flac")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        if not with_body:
            return
        # Nothing here paces the stream. A player stops reading once its
        # buffer is full, the write below then blocks, and so the noise is
        # generated only as fast as it is played.
        try:
            for chunk in flac_stream(generator(np.random.default_rng())):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the listener hung up, which is how every stream ends


def lan_address() -> str:
    """This host's address on the local network, to print URLs a speaker can use.

    Only for display, when listening on every address. A host on several
    networks has several addresses; this is the one on its default route.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            # Connecting a UDP socket sends nothing; it only picks the route.
            probe.connect(("192.0.2.1", 9))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def url_host(bind: str) -> str:
    """The host to print in the stream URLs of a server listening on bind."""
    return lan_address() if bind == EVERY_ADDRESS else bind


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve endless noise as FLAC over HTTP.")
    parser.add_argument(
        "--bind",
        default=EVERY_ADDRESS,
        metavar="ADDRESS",
        help="address to listen on (default: %(default)s, every address of this host)",
    )
    parser.add_argument("--port", type=int, default=8000, help="default: %(default)s")
    return parser


def main() -> int:
    parser = argument_parser()
    args = parser.parse_args()

    try:
        server = ThreadingHTTPServer((args.bind, args.port), NoiseHandler)
    except OSError as error:
        parser.error(f"cannot listen on {args.bind}:{args.port}: {error}")
    with server:
        port = server.server_port
        print(f"Listening on {args.bind}:{port}", file=sys.stderr)
        for path in STREAMS:
            print(f"http://{url_host(args.bind)}:{port}{path}", file=sys.stderr)
        sys.stderr.flush()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
