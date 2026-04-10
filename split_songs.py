#!/usr/bin/env python3
"""Split a live rehearsal WAV recording into individual songs.

Detects songs as loud contiguous segments separated by quiet breaks.
Writes one WAV file per detected song into an output directory.

Usage:
    python3 split_songs.py raw/recording.WAV
    python3 split_songs.py raw/recording.WAV -o songs/ --preview
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf


COLORS: list[str] = [
    "#4a9eff", "#5abb6a", "#e07840", "#b06adb", "#e0c040",
    "#40b8c8", "#e05070", "#70c870", "#c06030", "#8080e0", "#40c8a0",
]
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


def rms_to_db(rms: np.ndarray) -> np.ndarray:
    """Convert linear RMS to dBFS. Returns ~-240 dBFS for silence (clamped at 1e-12)."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12))


def downsample_rms(
    rms: np.ndarray, n_points: int, normalize: bool = True
) -> list[float]:
    """Downsample a 1-D RMS array to n_points and optionally normalise to 0–1.

    Uses block-mean downsampling. Output is always exactly n_points long
    (zero-padded if the input is shorter).
    """
    arr = rms.astype(np.float64)
    if len(arr) >= n_points:
        # Trim to a multiple of n_points, reshape, take column means
        trimmed = arr[: n_points * (len(arr) // n_points)]
        arr = trimmed.reshape(n_points, -1).mean(axis=1)
    # Zero-pad if shorter than requested
    if len(arr) < n_points:
        arr = np.pad(arr, (0, n_points - len(arr)))
    if normalize and len(arr) > 0:
        peak = arr.max()
        if peak > 0:
            arr = arr / peak
    return [round(float(v), 4) for v in arr[:n_points]]


def build_metadata(
    input_path: str,
    rate: int,
    channels: int,
    rms: np.ndarray,
    window_sec: float,
    songs: list[tuple[int, int]],
) -> dict:
    """Build the segments.json payload for the review server.

    Args:
        input_path: path to the original WAV file (stored as-is for server use)
        rate: sample rate in Hz
        channels: number of audio channels
        rms: full per-window RMS array from compute_rms()
        window_sec: window size in seconds used during analysis
        songs: list of (start_window, end_window) tuples from find_songs()

    Returns:
        Dict matching the segments.json schema.
    """
    total_sec = len(rms) * window_sec
    duration_min = total_sec / 60.0

    segment_list = []
    for i, (w_start, w_end) in enumerate(songs):
        segment_list.append({
            "id": i,
            "name": f"Song {i + 1:02d}",
            "start_min": round(w_start * window_sec / 60.0, 4),
            "end_min": round(w_end * window_sec / 60.0, 4),
            "color": COLORS[i % len(COLORS)],
            "exported": False,
        })

    return {
        "source_file": input_path,
        "duration_min": round(duration_min, 2),
        "sample_rate": rate,
        "channels": channels,
        "overview_rms": downsample_rms(rms, 2000, normalize=True),
        "segments": segment_list,
    }


def main(args: argparse.Namespace) -> None:
    # --- Load ---
    print(f"Reading {args.input} ...")
    info = sf.info(args.input)
    print(
        f"  {info.frames / info.samplerate / 60:.1f} min  |  "
        f"{info.samplerate} Hz  {info.channels}ch  {info.format}"
    )
    samples, rate = sf.read(args.input, dtype="float32", always_2d=True)

    # --- Analyse ---
    window_sec = args.window_seconds
    rms = compute_rms(samples, rate, window_sec)
    db = rms_to_db(rms)

    threshold_linear = 10 ** (args.threshold_db / 20.0)
    min_song_windows = int((args.min_song_minutes * 60) / window_sec)
    merge_gap_windows = int(args.merge_gap_seconds / window_sec)

    print(
        f"\nThreshold: {args.threshold_db} dBFS  |  "
        f"Min song: {args.min_song_minutes} min  |  "
        f"Merge gap: {args.merge_gap_seconds} s"
    )
    print(f"Recording loudness range: {db.min():.1f} to {db.max():.1f} dBFS\n")

    songs = find_songs(rms, min_song_windows, merge_gap_windows, threshold_linear)

    if not songs:
        print("No songs detected. Try lowering --threshold-db.")
        return

    print(f"Detected {len(songs)} song(s):")
    pad_frames = int(args.pad_seconds * rate)

    def to_frames(w: int) -> int:
        return int(w * window_sec * rate)

    for i, (w_start, w_end) in enumerate(songs, 1):
        f_start = to_frames(w_start)
        f_end = to_frames(w_end)
        duration_sec = (f_end - f_start) / rate
        start_min = f_start / rate / 60
        peak_db = db[w_start:w_end].max()
        print(
            f"  Song {i:02d}: {start_min:5.1f} min  "
            f"duration {duration_sec/60:.1f} min  "
            f"peak {peak_db:.1f} dBFS"
        )

    if args.preview:
        print("\n(--preview: no files written)")
        return

    # --- Export ---
    stem = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or f"{stem}_songs"
    os.makedirs(output_dir, exist_ok=True)

    for i, (w_start, w_end) in enumerate(songs, 1):
        f_start = max(0, to_frames(w_start) - pad_frames)
        f_end = min(len(samples), to_frames(w_end) + pad_frames)
        chunk = samples[f_start:f_end]

        out_path = os.path.join(output_dir, f"song_{i:02d}.wav")
        sf.write(out_path, chunk, rate, subtype="FLOAT")
        duration_sec = len(chunk) / rate
        print(f"  -> {out_path}  ({duration_sec/60:.1f} min)")

    metadata = build_metadata(
        input_path=args.input,
        rate=rate,
        channels=info.channels,
        rms=rms,
        window_sec=window_sec,
        songs=songs,
    )

    print(f"\nDone. {len(songs)} files written to {output_dir}/")


if __name__ == "__main__":
    main(parse_args())
