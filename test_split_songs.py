#!/usr/bin/env python3
"""Tests for split_songs.py"""
import json
import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, ".")
from split_songs import compute_rms, find_songs, downsample_rms, build_metadata, COLORS, write_segments_json

import pytest


def test_compute_rms_silent():
    """Silent signal should give RMS of 0 (or very close)."""
    samples = np.zeros((48000 * 5, 2), dtype=np.float32)  # 5 seconds, stereo
    rms = compute_rms(samples, rate=48000, window_sec=1.0)
    assert rms.shape == (5,), f"Expected 5 windows, got {rms.shape}"
    assert np.all(rms < 1e-10), f"Expected near-zero RMS for silence, got {rms}"


def test_compute_rms_loud():
    """Full-scale sine wave should give peak near 1.0."""
    t = np.linspace(0, 3, 48000 * 3, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    stereo = np.stack([sine, sine], axis=1)
    rms = compute_rms(stereo, rate=48000, window_sec=1.0)
    assert rms.shape == (3,), f"Expected 3 windows, got {rms.shape}"
    assert np.allclose(rms, 1.0, atol=0.01), f"Expected ~1.0, got {rms}"


def test_compute_rms_mixed():
    """First second loud, next two silent."""
    t = np.linspace(0, 1, 48000, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(48000 * 2, dtype=np.float32)
    mono = np.concatenate([sine, silence])
    stereo = np.stack([mono, mono], axis=1)
    rms = compute_rms(stereo, rate=48000, window_sec=1.0)
    assert rms[0] > 0.9, f"First window should be loud (peak ~1.0), got {rms[0]}"
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


def test_build_metadata_top_level_keys():
    rms = np.array([0.01] * 60 + [0.8] * 120 + [0.01] * 60, dtype=np.float64)
    meta = build_metadata("raw/test.wav", 48000, 2, rms, 1.0, [(60, 180)])
    assert meta["source_file"] == "raw/test.wav"
    assert meta["sample_rate"] == 48000
    assert meta["channels"] == 2
    assert meta["duration_min"] == pytest.approx(4.0, abs=0.1)
    assert len(meta["overview_rms"]) == 2000
    assert len(meta["segments"]) == 1


def test_build_metadata_segment_fields():
    rms = np.array([0.01] * 60 + [0.8] * 120 + [0.01] * 60, dtype=np.float64)
    meta = build_metadata("raw/test.wav", 48000, 2, rms, 1.0, [(60, 180)])
    s = meta["segments"][0]
    assert s["id"] == 0
    assert s["name"] == "Song 01"
    assert s["color"] == COLORS[0]
    assert s["start_min"] == pytest.approx(1.0, abs=0.01)
    assert s["end_min"] == pytest.approx(3.0, abs=0.01)
    assert s["exported"] is False
    assert "waveform" not in s
    assert "file" not in s


def test_build_metadata_color_cycles():
    n_songs = 12
    rms = np.ones(n_songs * 300, dtype=np.float64) * 0.5
    songs = [(i * 300, (i + 1) * 300) for i in range(n_songs)]
    meta = build_metadata("x.wav", 48000, 2, rms, 1.0, songs)
    assert meta["segments"][11]["color"] == COLORS[11 % len(COLORS)]


def test_write_segments_json_creates_file():
    rms = np.array([0.01] * 60 + [0.8] * 120 + [0.01] * 60, dtype=np.float64)
    meta = build_metadata("raw/test.wav", 48000, 2, rms, 1.0, [(60, 180)])
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_segments_json(tmpdir, meta)
        assert os.path.exists(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["source_file"] == "raw/test.wav"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["exported"] is False
        assert data["segments"][0]["id"] == 0




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
        test_build_metadata_top_level_keys,
        test_build_metadata_segment_fields,
        test_build_metadata_color_cycles,
        test_write_segments_json_creates_file,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
