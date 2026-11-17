"""Tests for generate_noise.py. Run with: python3 -m unittest"""

import argparse
import io
import os
import pty
import subprocess
import sys
import unittest
import wave
from pathlib import Path

import generate_noise as gn

SCRIPT = Path(__file__).with_name("generate_noise.py")


def run_cli(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, timeout=60
    )


def read_wav(data: bytes) -> tuple[wave._wave_params, bytes]:
    """(params, frames) of WAV bytes, parsed by the standard library."""
    with wave.open(io.BytesIO(data)) as wav:
        return wav.getparams(), wav.readframes(wav.getnframes())


class ParseDurationTest(unittest.TestCase):
    def test_units(self) -> None:
        cases = {
            "10": 10,
            "10s": 10,
            "5m": 300,
            "12h": 43200,
            "1.5h": 5400,
            "0.5": 0.5,
            " 10S ": 10,
        }
        for text, seconds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(gn.parse_duration(text), seconds)

    def test_rejects(self) -> None:
        for text in ("", "abc", "s", "10x", "1h30m", "-5", "0", "0s", "1e3", "10 s"):
            with self.subTest(text=text):
                with self.assertRaises(argparse.ArgumentTypeError):
                    gn.parse_duration(text)


class WavHeaderTest(unittest.TestCase):
    def test_is_44_bytes(self) -> None:
        self.assertEqual(len(gn.wav_header(0)), 44)

    def test_describes_the_format(self) -> None:
        params, frames = read_wav(gn.wav_header(3) + bytes(3 * gn.SAMPLE_WIDTH))
        self.assertEqual(params.nchannels, gn.CHANNELS)
        self.assertEqual(params.sampwidth, gn.SAMPLE_WIDTH)
        self.assertEqual(params.framerate, gn.SAMPLE_RATE)
        self.assertEqual(params.nframes, 3)
        self.assertEqual(len(frames), 3 * gn.SAMPLE_WIDTH)

    def test_longest_that_fits(self) -> None:
        frame_size = gn.CHANNELS * gn.SAMPLE_WIDTH
        longest = (0xFFFFFFFF - 36) // frame_size
        gn.wav_header(longest)
        with self.assertRaises(ValueError):
            gn.wav_header(longest + 1)


class WriteWavTest(unittest.TestCase):
    def write(self, total_frames: int) -> bytes:
        out = io.BytesIO()
        gn.write_wav(out, "brown", total_frames)
        return out.getvalue()

    def test_frame_counts(self) -> None:
        # Around the chunk size, where an off-by-one would show.
        chunk = gn.CHUNK_FRAMES
        for total in (0, 1, chunk - 1, chunk, chunk + 1, 2 * chunk + 123):
            with self.subTest(total=total):
                data = self.write(total)
                params, frames = read_wav(data)
                self.assertEqual(params.nframes, total)
                self.assertEqual(len(frames), total * gn.SAMPLE_WIDTH)
                self.assertEqual(len(data), 44 + total * gn.SAMPLE_WIDTH)


class GenerateTest(unittest.TestCase):
    def test_stub_is_silence_of_the_right_length(self) -> None:
        for color in gn.COLORS:
            with self.subTest(color=color):
                self.assertEqual(gn.generate(color, 100), bytes(100 * gn.SAMPLE_WIDTH))


class CliTest(unittest.TestCase):
    def test_writes_a_wav_of_the_requested_length(self) -> None:
        result = run_cli("--duration", "2s")
        self.assertEqual(result.returncode, 0, result.stderr)
        params, _ = read_wav(result.stdout)
        self.assertEqual(params.nframes, 2 * gn.SAMPLE_RATE)
        self.assertEqual(params.framerate, gn.SAMPLE_RATE)

    def test_fractional_duration(self) -> None:
        result = run_cli("--duration", "0.5")
        params, _ = read_wav(result.stdout)
        self.assertEqual(params.nframes, gn.SAMPLE_RATE // 2)

    def test_every_color_is_accepted(self) -> None:
        for color in gn.COLORS:
            with self.subTest(color=color):
                result = run_cli("--duration", "1s", "--color", color)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_usage_errors(self) -> None:
        cases = {
            "no duration": [],
            "bad duration": ["--duration", "abc"],
            "unknown color": ["--duration", "1s", "--color", "purple"],
            "too long for a WAV": ["--duration", "14h"],
        }
        for name, args in cases.items():
            with self.subTest(name):
                result = run_cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b"")
                self.assertIn(b"error:", result.stderr)

    def test_refuses_a_terminal(self) -> None:
        controller, terminal = pty.openpty()
        try:
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--duration", "1s"],
                stdout=terminal,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        finally:
            os.close(controller)
            os.close(terminal)
        self.assertEqual(result.returncode, 2)
        self.assertIn(b"terminal", result.stderr)

    def test_reader_going_away_is_quiet(self) -> None:
        process = subprocess.Popen(
            [sys.executable, str(SCRIPT), "--duration", "1h"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None and process.stderr is not None
        process.stdout.read(100)
        process.stdout.close()
        stderr = process.stderr.read()
        process.stderr.close()
        self.assertEqual(process.wait(timeout=60), 1)
        self.assertEqual(stderr, b"")


if __name__ == "__main__":
    unittest.main()
