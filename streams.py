"""Where each color of noise is served: what the server and the scripts that
send a speaker to it have to agree on."""

from noise import GENERATORS, Noise


def stream_path(color: str) -> str:
    """The path a color is served at, e.g. /brown.flac."""
    return f"/{color}.flac"


def cover_path(color: str) -> str:
    """The path a color's cover image is served at, e.g. /brown.png."""
    return f"/{color}.png"


STREAMS: dict[str, type[Noise]] = {
    stream_path(color): generator for color, generator in GENERATORS.items()
}
COVERS: dict[str, str] = {cover_path(color): color for color in GENERATORS}
