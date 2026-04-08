#!/usr/bin/env python3
"""Tests for split_songs.py"""
import sys
import os
import numpy as np

sys.path.insert(0, ".")
from split_songs import compute_rms, find_songs, downsample_rms

try:
    import pytest
except ImportError:
    # Fallback if pytest not available
    class Approx:
        def __init__(self, val, abs_tol):
            self.val = val
            self.abs_tol = abs_tol if abs_tol is not None else 1e-6
        def __eq__(self, other):
            return abs(other - self.val) <= self.abs_tol
        def __repr__(self):
            return f"{self.val} +/- {self.abs_tol}"

    class pytest:
        @staticmethod
        def approx(value, abs=None):
            return Approx(value, abs)


def test_compute_rms_silent():
    """Silent signal should give RMS of 0 (or very close)."""
    samples = np.zeros((48000 * 5, 2), dtype=np.float32)  # 5 seconds, stereo
    rms = compute_rms(samples, rate=48000, window_sec=1.0)
    assert rms.shape == (5,), f"Expected 5 windows, got {rms.shape}"
    assert np.all(rms < 1e-10), f"Expected near-zero RMS for silence, got {rms}"


def test_compute_rms_loud():
    """Full-scale sine wave should give RMS near 0.707 (~-3 dBFS)."""
    t = np.linspace(0, 3, 48000 * 3, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)
    rms = compute_rms(stereo, rate=48000, window_sec=1.0)
    assert rms.shape == (3,), f"Expected 3 windows, got {rms.shape}"
    assert np.allclose(rms, 0.707, atol=0.01), f"Expected ~0.707, got {rms}"


def test_compute_rms_mixed():
    """First second loud, next two silent."""
    t = np.linspace(0, 1, 48000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(48000 * 2, dtype=np.float32)
    mono = np.concatenate([sine, silence])
    stereo = np.stack([mono, mono], axis=1)
    rms = compute_rms(stereo, rate=48000, window_sec=1.0)
    assert rms[0] > 0.5, f"First window should be loud, got {rms[0]}"
    assert rms[1] < 1e-10, f"Second window should be silent, got {rms[1]}"
    assert rms[2] < 1e-10, f"Third window should be silent, got {rms[2]}"


def test_find_songs_basic():
    """Three loud blocks separated by silence -> three songs."""
    rms = np.array(
        [0.5] * 10 + [0.001] * 5 + [0.5] * 10 + [0.001] * 5 + [0.5] * 10 + [0.001] * 20,
        dtype=np.float64,
    )
    songs = find_songs(rms, min_song_windows=5, merge_gap_windows=2, threshold=0.1)
    assert len(songs) == 3, f"Expected 3 songs, got {len(songs)}: {songs}"
    assert songs[0] == (0, 10)
    assert songs[1] == (15, 25)
    assert songs[2] == (30, 40)


def test_find_songs_merges_gap():
    """Two loud blocks with small gap -> merged into one song."""
    rms = np.array(
        [0.5] * 10 + [0.001] * 3 + [0.5] * 10 + [0.001] * 20,
        dtype=np.float64,
    )
    songs = find_songs(rms, min_song_windows=5, merge_gap_windows=5, threshold=0.1)
    assert len(songs) == 1, f"Expected 1 merged song, got {len(songs)}: {songs}"
    assert songs[0] == (0, 23)


def test_find_songs_filters_short():
    """Loud block shorter than min_song_windows is discarded."""
    rms = np.array(
        [0.5] * 2 + [0.001] * 5 + [0.5] * 10 + [0.001] * 20,
        dtype=np.float64,
    )
    songs = find_songs(rms, min_song_windows=5, merge_gap_windows=2, threshold=0.1)
    assert len(songs) == 1, f"Expected 1 song (short one filtered), got {len(songs)}"
    assert songs[0] == (7, 17)


def test_downsample_rms_reduces_length():
    rms = np.ones(1000, dtype=np.float64) * 0.5
    result = downsample_rms(rms, 100, normalize=False)
    assert len(result) == 100
    assert all(abs(v - 0.5) < 0.001 for v in result)


def test_downsample_rms_normalizes_to_one():
    rms = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float64)
    result = downsample_rms(rms, 4, normalize=True)
    assert len(result) == 4
    assert max(result) == pytest.approx(1.0, abs=0.001)


def test_downsample_rms_short_input_pads():
    rms = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    result = downsample_rms(rms, 10, normalize=False)
    assert len(result) == 10


def test_downsample_rms_silent_signal():
    rms = np.zeros(500, dtype=np.float64)
    result = downsample_rms(rms, 50, normalize=True)
    assert len(result) == 50
    assert all(v == 0.0 for v in result)


if __name__ == "__main__":
    tests = [
        test_compute_rms_silent,
        test_compute_rms_loud,
        test_compute_rms_mixed,
        test_find_songs_basic,
        test_find_songs_merges_gap,
        test_find_songs_filters_short,
        test_downsample_rms_reduces_length,
        test_downsample_rms_normalizes_to_one,
        test_downsample_rms_short_input_pads,
        test_downsample_rms_silent_signal,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
