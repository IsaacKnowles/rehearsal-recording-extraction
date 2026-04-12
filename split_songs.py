#!/usr/bin/env python3
"""Analyse a rehearsal WAV and launch the interactive review server.

Reads the recording, computes the RMS waveform, writes segments.json with an
empty segments list, then starts a local Flask server for creating, editing,
and exporting individual songs.

Usage:
    python3 split_songs.py raw/recording.WAV
    python3 split_songs.py raw/recording.WAV -o songs/
    python3 split_songs.py raw/recording.WAV --reset
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
    """Compute per-window peak amplitude.

    Args:
        samples: shape (n_frames, n_channels), float32 or float64
        rate: sample rate in Hz
        window_sec: window size in seconds

    Returns:
        1-D array of peak absolute values, one per complete window.
        Values are in 0–1.0 for float32 audio (matching the native sample range).
    """
    window_frames = int(rate * window_sec)
    mono = samples.mean(axis=1)  # mix to mono for analysis
    n_windows = len(mono) // window_frames
    # Reshape into (n_windows, window_frames) for fast vectorised peak
    trimmed = mono[: n_windows * window_frames].reshape(n_windows, window_frames)
    return np.max(np.abs(trimmed.astype(np.float64)), axis=1)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse a rehearsal recording and launch the interactive review server.")
    p.add_argument("input", help="Path to input WAV file")
    p.add_argument("-o", "--output-dir", default=None, help="Output directory (default: <stem>_songs/)")
    p.add_argument("--port", type=int, default=5123, help="Review server port (default: 5123)")
    p.add_argument("--reset", action="store_true", help="Re-analyse recording even if segments.json already exists")
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
    if len(arr) != n_points:
        x_in = np.linspace(0, 1, len(arr))
        x_out = np.linspace(0, 1, n_points)
        arr = np.interp(x_out, x_in, arr)
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
        "overview_rms": downsample_rms(rms, 2000, normalize=False),
        "segments": segment_list,
    }


def write_segments_json(output_dir: str, metadata: dict) -> str:
    """Write segments.json to output_dir. Returns the path written."""
    path = os.path.join(output_dir, "segments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, separators=(",", ":"))
    return path


def launch_server(output_dir: str, input_path: str, port: int) -> None:
    """Import and start the review server. Blocks until Ctrl+C."""
    import threading
    import webbrowser
    import review_server

    url = f"http://localhost:{port}"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    review_server.run(output_dir=output_dir, source_wav=input_path, port=port)


def main(args: argparse.Namespace) -> None:
    stem = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or f"{stem}_songs"
    segments_json = os.path.join(output_dir, "segments.json")

    # Session recovery: reuse existing segments.json unless --reset is passed
    if os.path.exists(segments_json) and not args.reset:
        print(f"Loading existing session from {segments_json}")
        print("  (pass --reset to re-analyse recording)")
        launch_server(output_dir, args.input, args.port)
        return

    # --- Load ---
    print(f"Reading {args.input} ...")
    info = sf.info(args.input)
    print(
        f"  {info.frames / info.samplerate / 60:.1f} min  |  "
        f"{info.samplerate} Hz  {info.channels}ch  {info.format}"
    )
    samples, rate = sf.read(args.input, dtype="float32", always_2d=True)

    # --- Analyse ---
    window_sec = 1.0
    rms = compute_rms(samples, rate, window_sec)

    os.makedirs(output_dir, exist_ok=True)
    metadata = build_metadata(args.input, rate, info.channels, rms, window_sec, [])
    json_path = write_segments_json(output_dir, metadata)
    print(f"\n  -> {json_path}  (segments)")

    launch_server(output_dir, args.input, args.port)


if __name__ == "__main__":
    main(parse_args())
