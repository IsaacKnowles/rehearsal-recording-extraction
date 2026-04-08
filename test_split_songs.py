#!/usr/bin/env python3
"""Tests for split_songs.py"""
import sys
import numpy as np

sys.path.insert(0, ".")
from split_songs import compute_rms


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


if __name__ == "__main__":
    tests = [test_compute_rms_silent, test_compute_rms_loud, test_compute_rms_mixed]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
