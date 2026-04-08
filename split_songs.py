#!/usr/bin/env python3
"""Split a live rehearsal WAV recording into individual songs.

Detects songs as loud contiguous segments separated by quiet breaks.
Writes one WAV file per detected song into an output directory.

Usage:
    python3 split_songs.py raw/recording.WAV
    python3 split_songs.py raw/recording.WAV -o songs/ --preview
"""

import argparse
import os
import sys

import numpy as np
import soundfile as sf


def compute_rms(samples: np.ndarray, rate: int, window_sec: float = 1.0) -> np.ndarray:
    """Compute per-window RMS amplitude.

    Args:
        samples: shape (n_frames, n_channels), float32 or float64
        rate: sample rate in Hz
        window_sec: window size in seconds

    Returns:
        1-D array of RMS values, one per complete window
    """
    window_frames = int(rate * window_sec)
    mono = samples.mean(axis=1)  # mix to mono for analysis
    n_windows = len(mono) // window_frames
    # Reshape into (n_windows, window_frames) for fast vectorised RMS
    trimmed = mono[: n_windows * window_frames].reshape(n_windows, window_frames)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    return rms


def find_songs(
    rms: np.ndarray,
    min_song_windows: int,
    merge_gap_windows: int,
    threshold: float,
) -> list[tuple[int, int]]:
    """Detect song segments from an RMS array.

    Args:
        rms: per-window RMS values
        min_song_windows: discard segments shorter than this many windows
        merge_gap_windows: merge consecutive segments whose gap is <= this
        threshold: RMS value above which a window counts as 'loud'

    Returns:
        List of (start_window, end_window) tuples (end is exclusive).
    """
    loud = rms >= threshold

    # Collect contiguous loud runs
    segments: list[list[int]] = []
    in_seg = False
    for i, is_loud in enumerate(loud):
        if is_loud and not in_seg:
            segments.append([i, i + 1])
            in_seg = True
        elif is_loud and in_seg:
            segments[-1][1] = i + 1
        else:
            in_seg = False

    # Merge segments whose gap is small enough
    merged: list[list[int]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= merge_gap_windows:
            merged[-1][1] = seg[1]
        else:
            merged.append(list(seg))

    # Filter by minimum duration
    return [
        (s[0], s[1])
        for s in merged
        if s[1] - s[0] >= min_song_windows
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="Split rehearsal WAV into songs by volume detection"
    )
    p.add_argument("input", help="Input WAV file")
    p.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: <input_stem>_songs/)"
    )
    p.add_argument(
        "--threshold-db", type=float, default=-40.0,
        help="RMS threshold in dBFS below which a window is 'quiet' (default: -40)"
    )
    p.add_argument(
        "--min-song-minutes", type=float, default=2.0,
        help="Minimum song duration in minutes (default: 2)"
    )
    p.add_argument(
        "--merge-gap-seconds", type=float, default=20.0,
        help="Merge loud segments separated by fewer than this many quiet seconds (default: 20)"
    )
    p.add_argument(
        "--window-seconds", type=float, default=1.0,
        help="Analysis window size in seconds (default: 1.0)"
    )
    p.add_argument(
        "--pad-seconds", type=float, default=2.0,
        help="Silence padding added before/after each exported song (default: 2)"
    )
    p.add_argument(
        "--preview", action="store_true",
        help="Print detected segments without writing files"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Input: {args.input}")
    print("(scaffold only — no logic yet)")
