"""Where each color of noise is served: what the server and the scripts that
send a speaker to it have to agree on."""

from noise import GENERATORS, Noise


def stream_path(color: str) -> str:
    """The path a color is served at, e.g. /brown.flac."""
    return f"/{color}.flac"


STREAMS: dict[str, type[Noise]] = {
    stream_path(color): generator for color, generator in GENERATORS.items()
}
