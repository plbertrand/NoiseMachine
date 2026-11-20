#!/usr/bin/env python3
"""Serve endless noise as FLAC over HTTP, for a network speaker to play.

    serve_noise.py --port 8000

Each color is a stream that never ends, e.g. http://<this host>:8000/brown.flac,
with a cover image beside it at /brown.png.

A Sonos speaker can also be told over HTTP to play them: see control.py.
"""

import argparse
import json
import socket
import sys
from collections.abc import Iterator
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import numpy as np
import pyflac  # type: ignore[import-untyped]

import control
import cover
import mdns
from noise import SAMPLE_RATE, Noise
from streams import COVERS, STREAMS

CHUNK_FRAMES = SAMPLE_RATE  # noise is generated a second at a time
EVERY_ADDRESS = "0.0.0.0"
ANYWHERE = "192.0.2.1"  # reserved for documentation: nothing is there


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
    """GET /<color>.flac streams that color until the listener hangs up.

    GET /<color>.png is its cover image. The routes of control.py tell a Sonos
    speaker to play these streams.
    """

    def do_GET(self) -> None:
        if not self._control("GET"):
            self._respond(with_body=True)

    def do_HEAD(self) -> None:
        self._respond(with_body=False)

    def do_POST(self) -> None:
        if not self._control("POST"):
            self.send_error(HTTPStatus.NOT_FOUND)

    def _control(self, method: str) -> bool:
        """Answer a request to control a speaker; False if it is not one."""
        address = self.server.server_address
        assert isinstance(address, tuple)  # of an internet socket: (host, port)
        bind, port = str(address[0]), int(address[1])
        reply = control.control(
            lambda speaker: server_url(bind, port, towards=speaker), method, self.path
        )
        if reply is None:
            return False
        body = json.dumps(reply.content).encode() + b"\n"
        self.send_response(reply.code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if reply.allow:
            self.send_header("Allow", reply.allow)
        self.end_headers()
        self.wfile.write(body)
        return True

    def _respond(self, with_body: bool) -> None:
        path = urlsplit(self.path).path
        if path in STREAMS:
            self._stream(STREAMS[path], with_body)
        elif path in COVERS:
            self._image(cover.cover(COVERS[path]), with_body)
        else:
            self.send_error(
                HTTPStatus.NOT_FOUND, explain=f"The streams are {', '.join(STREAMS)}"
            )

    def _image(self, image: bytes, with_body: bool) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(image)))
        self.end_headers()
        if with_body:
            self.wfile.write(image)

    def _stream(self, generator: type[Noise], with_body: bool) -> None:
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


def lan_address(towards: str = ANYWHERE) -> str:
    """This host's address on the network that leads to the address towards.

    A host on several networks has several addresses, and a speaker can only
    use the one on its own network. Towards anywhere, it is the address on
    the default route.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            # Connecting a UDP socket sends nothing; it only picks the route.
            probe.connect((towards, 9))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def url_host(bind: str, towards: str = ANYWHERE) -> str:
    """The host in the URLs of a server listening on bind, for a listener at towards.

    A server told to listen on one address is at that address for everyone.
    """
    return lan_address(towards) if bind == EVERY_ADDRESS else bind


def server_url(bind: str, port: int, towards: str = ANYWHERE) -> str:
    """Where a listener at towards finds a server listening on bind and port."""
    return f"http://{url_host(bind, towards)}:{port}"


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
            print(f"{server_url(args.bind, port)}{path}", file=sys.stderr)
        print("To control a Sonos speaker, by IP address or room name:", file=sys.stderr)
        for path, (method, _) in control.ROUTES.items():
            url = f"{server_url(args.bind, port)}{path}?{control.SPEAKER}=<speaker>"
            print(f"{method} {url}", file=sys.stderr)
        sys.stderr.flush()
        with mdns.announced(args.bind, port):
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
