"""Tests for noise.py. Run with: python3 -m unittest"""

import unittest
from collections.abc import Iterator

import numpy as np

from noise import (
    BROWN_CUTOFF_HZ,
    GENERATORS,
    RMS_LEVEL,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    BrownNoise,
    Noise,
    Samples,
    WhiteNoise,
)


def samples(pcm: bytes) -> Samples:
    """16-bit PCM as floats, full scale 1.0."""
    return np.frombuffer(pcm, dtype="<i2") / 32767


def octave_slope_db(signal: Samples, low_hz: float) -> float:
    """Change in power density from low_hz to twice low_hz, in dB."""
    power = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)

    def density(center: float) -> float:
        return float(power[(freqs >= center * 0.9) & (freqs < center * 1.1)].mean())

    return float(10 * np.log10(density(2 * low_hz) / density(low_hz)))


class GeneratorTest(unittest.TestCase):
    """What every noise color has in common."""

    SECONDS = 30

    def make(self, color: str, seed: int = 0) -> Noise:
        return GENERATORS[color](np.random.default_rng(seed))

    def each_color(self) -> Iterator[str]:
        for color in GENERATORS:
            with self.subTest(color=color):
                yield color

    def test_read_returns_the_frames_asked_for(self) -> None:
        for color in self.each_color():
            noise = self.make(color)
            for n_frames in (0, 1, 4096, 4097, 10000):
                self.assertEqual(len(noise.read(n_frames)), n_frames * SAMPLE_WIDTH)

    def test_same_seed_same_output(self) -> None:
        for color in self.each_color():
            self.assertEqual(self.make(color, 1).read(10000), self.make(color, 1).read(10000))

    def test_different_seed_different_output(self) -> None:
        for color in self.each_color():
            self.assertNotEqual(self.make(color, 1).read(10000), self.make(color, 2).read(10000))

    def test_chunking_does_not_change_the_output(self) -> None:
        # The sizes straddle BrownNoise's internal block, so a filter state
        # dropped between reads or between blocks would show.
        sizes = (1, 4095, 4097, 50000, 7)
        for color in self.each_color():
            whole = self.make(color).read(sum(sizes))
            noise = self.make(color)
            self.assertEqual(b"".join(noise.read(n) for n in sizes), whole)

    def test_level(self) -> None:
        for color in self.each_color():
            signal = samples(self.make(color).read(self.SECONDS * SAMPLE_RATE))
            self.assertAlmostEqual(signal.std(), RMS_LEVEL, delta=0.05 * RMS_LEVEL)
            self.assertAlmostEqual(signal.mean(), 0, delta=0.05 * RMS_LEVEL)
            self.assertLess(np.mean(np.abs(signal) >= 1), 1e-4, "too much clipping")

    def test_level_holds_over_time(self) -> None:
        # A generator that drifted or faded would differ between its halves.
        for color in self.each_color():
            signal = samples(self.make(color).read(self.SECONDS * SAMPLE_RATE))
            first, second = np.array_split(signal, 2)
            self.assertAlmostEqual(first.std(), second.std(), delta=0.1 * RMS_LEVEL)


class SpectrumTest(unittest.TestCase):
    SECONDS = 30

    def signal(self, generator: type[Noise]) -> Samples:
        noise = generator(np.random.default_rng(0))
        return samples(noise.read(self.SECONDS * SAMPLE_RATE))

    def test_white_is_flat(self) -> None:
        signal = self.signal(WhiteNoise)
        for low_hz in (100, 1000, 8000):
            with self.subTest(low_hz=low_hz):
                self.assertAlmostEqual(octave_slope_db(signal, low_hz), 0, delta=0.5)

    def test_brown_falls_6_db_per_octave_above_the_cutoff(self) -> None:
        signal = self.signal(BrownNoise)
        for low_hz in (500, 1000, 2000):
            with self.subTest(low_hz=low_hz):
                self.assertAlmostEqual(octave_slope_db(signal, low_hz), -6, delta=0.5)

    def test_brown_levels_off_below_the_cutoff(self) -> None:
        # A perfect integrator would keep falling 6 dB per octave down here.
        signal = self.signal(BrownNoise)
        slope = octave_slope_db(signal, BROWN_CUTOFF_HZ / 8)
        self.assertAlmostEqual(slope, 0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
