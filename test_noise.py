"""Tests for noise.py. Run with: python3 -m unittest"""

import unittest
from collections.abc import Callable, Iterator

import numpy as np

from noise import (
    BROWN_CUTOFF_HZ,
    GENERATORS,
    RMS_LEVEL,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    BrownNoise,
    Chain,
    DeepBrownNoise,
    Filter,
    Identity,
    Noise,
    OnePole,
    PinkFilter,
    PinkNoise,
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

    def test_pink_falls_3_db_per_octave(self) -> None:
        signal = self.signal(PinkNoise)
        for low_hz in (50, 100, 1000, 8000):
            with self.subTest(low_hz=low_hz):
                self.assertAlmostEqual(octave_slope_db(signal, low_hz), -3, delta=0.5)

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

    def test_deep_brown_falls_12_db_per_octave_above_its_low_pass(self) -> None:
        signal = self.signal(DeepBrownNoise)
        for low_hz in (2000, 4000):
            with self.subTest(low_hz=low_hz):
                self.assertAlmostEqual(octave_slope_db(signal, low_hz), -12, delta=0.7)

    def test_deep_brown_matches_brown_in_shape_below_its_low_pass(self) -> None:
        low_hz = BROWN_CUTOFF_HZ / 8
        deep = octave_slope_db(self.signal(DeepBrownNoise), low_hz)
        self.assertAlmostEqual(deep, 0, delta=1.5)

    def test_each_color_is_darker_than_the_last(self) -> None:
        # Share of the power above 1 kHz, for signals of the same level.
        def treble(generator: type[Noise]) -> float:
            signal = self.signal(generator)
            power = np.abs(np.fft.rfft(signal)) ** 2
            freqs = np.fft.rfftfreq(len(signal), 1 / SAMPLE_RATE)
            return float(power[freqs >= 1000].sum() / power.sum())

        shares = [
            treble(g)
            for g in (WhiteNoise, PinkNoise, BrownNoise, DeepBrownNoise)
        ]
        self.assertEqual(shares, sorted(shares, reverse=True))


class FilterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.random.default_rng(0).standard_normal(10000)

    def test_identity_changes_nothing(self) -> None:
        np.testing.assert_array_equal(Identity()(self.x), self.x)

    def test_chain_applies_its_filters_in_order(self) -> None:
        chained = Chain(OnePole(0.9), PinkFilter())(self.x)
        np.testing.assert_array_equal(chained, PinkFilter()(OnePole(0.9)(self.x)))

    def test_empty_chain_changes_nothing(self) -> None:
        np.testing.assert_array_equal(Chain()(self.x), self.x)

    def test_filters_keep_their_state_between_calls(self) -> None:
        makers: dict[str, Callable[[], Filter]] = {
            "identity": Identity,
            "one pole": lambda: OnePole(0.9),
            "pink": PinkFilter,
            "chain": lambda: Chain(OnePole(0.9), PinkFilter()),
        }
        for name, make in makers.items():
            with self.subTest(name):
                whole = make()(self.x)
                chunked = make()
                parts = [chunked(part) for part in np.split(self.x, [1, 500, 4097])]
                np.testing.assert_allclose(np.concatenate(parts), whole, rtol=1e-9, atol=1e-9)

    def test_filters_do_not_change_their_input(self) -> None:
        makers: list[Callable[[], Filter]] = [Identity, PinkFilter, lambda: OnePole(0.9)]
        for make in makers:
            before = self.x.copy()
            make()(self.x)
            np.testing.assert_array_equal(self.x, before)


class OnePoleTest(unittest.TestCase):
    def test_matches_the_recurrence(self) -> None:
        # Poles from each end of the range in use, one of them negative.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10000)
        for a in (0.99886, 0.55, -0.7616):
            with self.subTest(a=a):
                expected = np.empty(len(x))
                last = 0.0
                for n, value in enumerate(x):
                    last = a * last + value
                    expected[n] = last
                np.testing.assert_allclose(OnePole(a)(x), expected, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
