"""Colors of noise, each an endless source of PCM audio."""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

Samples = npt.NDArray[np.float64]  # audio as floats, full scale 1.0

# Frames per second. Changing this breaks pink noise: PinkFilter's
# coefficients are tuned for 44.1 kHz.
SAMPLE_RATE = 44100
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

# Deep brown is brown through one more low-pass at this frequency, so above
# it the power falls 12 dB per octave: the same rumble with less hiss on top.
DEEP_BROWN_LOWPASS_HZ = 300.0


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


class Filter(ABC):
    """Shapes a signal: called with samples, returns as many, filtered.

    A filter extends this and implements __call__(). Its state carries over
    from one call to the next, so the output is continuous however the input
    is chunked.
    """

    @abstractmethod
    def __call__(self, x: Samples) -> Samples:
        """The next len(x) samples of output, for the next samples of input."""


class Identity(Filter):
    """The filter that changes nothing."""

    def __call__(self, x: Samples) -> Samples:
        return x


class Chain(Filter):
    """Filters in series: each one's output is the next one's input."""

    def __init__(self, *filters: Filter) -> None:
        self._filters = filters

    def __call__(self, x: Samples) -> Samples:
        for each in self._filters:
            x = each(x)
        return x


class OnePole(Filter):
    """The filter y[n] = a * y[n-1] + x[n], a low-pass when 0 < a < 1."""

    def __init__(self, a: float) -> None:
        # The closed form in __call__ scales by a**-n, which overflows a
        # double for large n; this block size keeps it below e**200.
        block = min(4096, int(200 / -np.log(abs(a))))
        self._decay = a ** np.arange(1, block + 1)
        self._last: float = 0.0

    @classmethod
    def low_pass(cls, cutoff_hz: float) -> "OnePole":
        return cls(float(np.exp(-2 * np.pi * cutoff_hz / SAMPLE_RATE)))

    def __call__(self, x: Samples) -> Samples:
        y = np.empty(len(x))
        size = len(self._decay)
        for start in range(0, len(x), size):
            block = y[start : start + size]
            decay = self._decay[: len(block)]
            # y[n] = a**n * (y[0] + sum of x[k] / a**k for k <= n)
            block[:] = decay * (self._last + np.cumsum(x[start : start + size] / decay))
            self._last = block[-1]
        return y


class PinkFilter(Filter):
    """Paul Kellet's refined pink filter: six one-pole filters in parallel,
    which together fall 3 dB per octave to within 0.05 dB above 10 Hz.

    The coefficients are for 44.1 kHz.
    """

    _POLES: tuple[tuple[float, float], ...] = (  # (a, weight)
        (0.99886, 0.0555179),
        (0.99332, 0.0750759),
        (0.96900, 0.1538520),
        (0.86650, 0.3104856),
        (0.55000, 0.5329522),
        (-0.7616, -0.0168980),
    )
    _DIRECT = 0.5362  # weight of the input itself
    _DELAYED = 0.115926  # weight of the input one sample late

    def __init__(self) -> None:
        self._poles = [(OnePole(a), weight) for a, weight in self._POLES]
        self._previous: float = 0.0

    def __call__(self, x: Samples) -> Samples:
        if len(x) == 0:
            return x
        y = self._DIRECT * x
        for pole, weight in self._poles:
            y += weight * pole(x)
        y[0] += self._DELAYED * self._previous
        y[1:] += self._DELAYED * x[:-1]
        self._previous = x[-1]
        return y


class FilteredNoise(Noise):
    """White noise through a filter, scaled to RMS_LEVEL.

    A color extends this and implements make_filter().
    """

    def __init__(self, rng: np.random.Generator) -> None:
        super().__init__(rng)
        self._filter = self.make_filter()
        # White noise of variance 1 leaves a filter with the variance of
        # the filter's impulse response, measured here on a spare filter.
        # One second is long enough for every filter in this file to settle.
        impulse = np.zeros(SAMPLE_RATE)
        impulse[0] = 1.0
        response = self.make_filter()(impulse)
        self._gain = RMS_LEVEL / np.sqrt(np.sum(response**2))

    @abstractmethod
    def make_filter(self) -> Filter:
        """A new filter for this color, in its initial state."""

    def read(self, n_frames: int) -> bytes:
        white = self._rng.standard_normal(n_frames)
        return to_pcm(self._filter(white) * self._gain)


class WhiteNoise(FilteredNoise):
    """Equal power at every frequency."""

    def make_filter(self) -> Filter:
        return Identity()


class PinkNoise(FilteredNoise):
    """Power falling 3 dB per octave: equal power in every octave."""

    def make_filter(self) -> Filter:
        return PinkFilter()


class BrownNoise(FilteredNoise):
    """Power falling 6 dB per octave above BROWN_CUTOFF_HZ."""

    def make_filter(self) -> Filter:
        return OnePole.low_pass(BROWN_CUTOFF_HZ)


class DeepBrownNoise(FilteredNoise):
    """Brown noise, falling 12 dB per octave above DEEP_BROWN_LOWPASS_HZ."""

    def make_filter(self) -> Filter:
        return Chain(
            OnePole.low_pass(BROWN_CUTOFF_HZ),
            OnePole.low_pass(DEEP_BROWN_LOWPASS_HZ),
        )


# By the name each color goes by on the command line.
GENERATORS: dict[str, type[Noise]] = {
    "white": WhiteNoise,
    "pink": PinkNoise,
    "brown": BrownNoise,
    "deep-brown": DeepBrownNoise,
}
