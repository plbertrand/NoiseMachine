"""Colors of noise, each an endless source of PCM audio."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

Samples = npt.NDArray[np.float64]  # audio as floats, full scale 1.0

SAMPLE_RATE = 44100  # frames per second
SAMPLE_WIDTH = 2  # bytes per sample, 16-bit signed
CHANNELS = 1

# Output level, as the RMS of the signal where full scale is 1.0. The noise
# is Gaussian, so peaks beyond full scale (5 sigma here) are clipped; at this
# level that is about one sample in two million.
RMS_LEVEL = 0.2

# Brown noise is white noise through a leaky integrator: flat below this
# frequency and falling 6 dB per octave above it. A perfect integrator (0 Hz)
# would wander off to full scale, and the lower this is, the more of the power
# sits in rumble a small speaker cannot reproduce.
BROWN_CUTOFF_HZ = 60.0


def to_pcm(samples: Samples) -> bytes:
    """Float samples, full scale 1.0, as 16-bit little-endian PCM."""
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


class Noise(ABC):
    """A color of noise: an endless source of 16-bit mono PCM.

    A color extends this and implements read(). Successive reads continue
    the same signal, so the output is continuous however it is chunked.
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    @abstractmethod
    def read(self, n_frames: int) -> bytes:
        """The next n_frames of noise."""


class WhiteNoise(Noise):
    """Equal power at every frequency."""

    def read(self, n_frames: int) -> bytes:
        return to_pcm(self._rng.standard_normal(n_frames) * RMS_LEVEL)


class BrownNoise(Noise):
    """Power falling 6 dB per octave: y[n] = a * y[n-1] + x[n], x white.

    The filter state carries over from one read to the next.
    """

    # The closed form below scales by a**-n, which overflows a double for
    # large n; a block this size keeps it far inside the safe range.
    _BLOCK = 4096

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._a = np.exp(-2 * np.pi * BROWN_CUTOFF_HZ / SAMPLE_RATE)
        # The integrator multiplies the variance by 1 / (1 - a**2).
        self._gain = RMS_LEVEL * np.sqrt(1 - self._a**2)
        self._decay = self._a ** np.arange(1, self._BLOCK + 1)
        self._last: float = 0.0

    def read(self, n_frames: int) -> bytes:
        out = np.empty(n_frames)
        for start in range(0, n_frames, self._BLOCK):
            block = out[start : start + self._BLOCK]
            decay = self._decay[: len(block)]
            white = self._rng.standard_normal(len(block))
            # y[n] = a**n * (y[0] + sum of x[k] / a**k for k <= n)
            block[:] = decay * (self._last + np.cumsum(white / decay))
            self._last = block[-1]
        return to_pcm(out * self._gain)


# By the name each color goes by on the command line.
GENERATORS: dict[str, type[Noise]] = {
    "white": WhiteNoise,
    "brown": BrownNoise,
}
