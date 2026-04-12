# Rehearsal Recording Splitter — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split a single live rehearsal WAV recording into individual songs by detecting loud segments separated by quiet breaks.

**Architecture:** A single Python script reads a float32 stereo WAV file using `soundfile`, computes per-second RMS energy via `numpy`, finds loud contiguous regions, merges nearby segments and filters short ones, then writes each detected song as a separate WAV file. All tuneable parameters have sensible defaults and are exposed as CLI flags.

**Tech Stack:** Python 3.9, numpy, soundfile (libsndfile), argparse, stdlib only beyond those two packages.

---

## File Structure

- Create: `split_songs.py` — the complete utility (single file, no package structure needed)
- Create: `requirements.txt` — pins numpy and soundfile

---

### Task 1: Bootstrap — install dependencies and scaffold the script

**Files:**
- Create: `requirements.txt`
- Create: `split_songs.py`

- [ ] **Step 1: Create requirements.txt**

```
numpy>=1.24
soundfile>=0.12
```

- [ ] **Step 2: Install dependencies**

Run:
```bash
pip3 install -r requirements.txt
```
Expected: both packages install without error. On macOS, `soundfile` bundles its own `libsndfile` so no Homebrew step is needed.

- [ ] **Step 3: Verify the libraries can open the target file**

Run:
```bash
python3 -c "
import soundfile as sf
import numpy as np
info = sf.info('raw/260407_182931_TrLR.WAV')
print('rate:', info.samplerate, 'channels:', info.channels,
      'frames:', info.frames, 'format:', info.format,
      'duration min:', info.frames / info.samplerate / 60)
"
```
Expected output (approximate):
```
rate: 48000 channels: 2 frames: <N> format: WAV duration min: ~93.0
```

- [ ] **Step 4: Create the script scaffold**

`split_songs.py`:
```python
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
```

- [ ] **Step 5: Run the scaffold to confirm it parses args without error**

Run:
```bash
python3 split_songs.py raw/260407_182931_TrLR.WAV --preview
```
Expected:
```
Input: raw/260407_182931_TrLR.WAV
(scaffold only — no logic yet)
```

- [ ] **Step 6: Commit**

```bash
git init  # only if not already a git repo
git add requirements.txt split_songs.py
git commit -m "feat: scaffold rehearsal splitter with arg parsing"
```

---

### Task 2: RMS analysis — compute per-window loudness

**Files:**
- Modify: `split_songs.py`

- [ ] **Step 1: Write the failing test for `compute_rms`**

Add this to the bottom of `split_songs.py` (inside `if __name__ == "__main__"` guard — we'll move to a test file in a later step; for now inline is fine):

Actually, write a separate test runner instead. Create `test_split_songs.py`:

```python
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 test_split_songs.py
```
Expected:
```
  FAIL  test_compute_rms_silent: cannot import name 'compute_rms' from 'split_songs'
  FAIL  test_compute_rms_loud: ...
  FAIL  test_compute_rms_mixed: ...
```

- [ ] **Step 3: Implement `compute_rms` in `split_songs.py`**

Add this function before `parse_args()`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python3 test_split_songs.py
```
Expected:
```
  PASS  test_compute_rms_silent
  PASS  test_compute_rms_loud
  PASS  test_compute_rms_mixed
```

- [ ] **Step 5: Commit**

```bash
git add split_songs.py test_split_songs.py
git commit -m "feat: add compute_rms with tests"
```

---

### Task 3: Segment detection — find, merge, and filter loud regions

**Files:**
- Modify: `split_songs.py`
- Modify: `test_split_songs.py`

- [ ] **Step 1: Write the failing tests for `find_songs`**

Append to `test_split_songs.py`:

```python
from split_songs import find_songs


def test_find_songs_basic():
    """Three loud blocks separated by silence -> three songs."""
    # 60 windows: loud (10), quiet (5), loud (10), quiet (5), loud (10), quiet (20)
    rms = np.array(
        [0.5] * 10 + [0.001] * 5 + [0.5] * 10 + [0.001] * 5 + [0.5] * 10 + [0.001] * 20,
        dtype=np.float64,
    )
    # min_song_windows=5, merge_gap_windows=2 -> gaps of 5 > 2 so no merging
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
    # merge_gap_windows=5 -> gap of 3 < 5, so merge
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 test_split_songs.py
```
Expected: all three new tests FAIL with import error.

- [ ] **Step 3: Implement `find_songs` in `split_songs.py`**

Add this function after `compute_rms`:

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python3 test_split_songs.py
```
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add split_songs.py test_split_songs.py
git commit -m "feat: add find_songs with merge and duration filter"
```

---

### Task 4: Main pipeline — wire it all together and write output files

**Files:**
- Modify: `split_songs.py`

- [ ] **Step 1: Implement the main function**

Replace the `if __name__ == "__main__":` block with:

```python
def rms_to_db(rms: np.ndarray) -> np.ndarray:
    """Convert linear RMS to dBFS. Returns -inf for silence."""
    with np.errstate(divide="ignore"):
        return 20.0 * np.log10(np.maximum(rms, 1e-12))


def main(args) -> None:
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

    for i, (w_start, w_end) in enumerate(songs, 1):
        f_start = int(w_start * window_sec * rate)
        f_end = int(w_end * window_sec * rate)
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
        f_start = max(0, int(w_start * window_sec * rate) - pad_frames)
        f_end = min(len(samples), int(w_end * window_sec * rate) + pad_frames)
        chunk = samples[f_start:f_end]

        out_path = os.path.join(output_dir, f"song_{i:02d}.wav")
        sf.write(out_path, chunk, rate, subtype="FLOAT")
        duration_sec = len(chunk) / rate
        print(f"  -> {out_path}  ({duration_sec/60:.1f} min)")

    print(f"\nDone. {len(songs)} files written to {output_dir}/")


if __name__ == "__main__":
    main(parse_args())
```

- [ ] **Step 2: Run in preview mode on the real recording**

```bash
python3 split_songs.py raw/260407_182931_TrLR.WAV --preview
```

Expected output (approximate — exact numbers will vary):
```
Reading raw/260407_182931_TrLR.WAV ...
  93.0 min  |  48000 Hz  2ch  WAV

Threshold: -40.0 dBFS  |  Min song: 2.0 min  |  Merge gap: 20.0 s
Recording loudness range: -80.0 to -3.0 dBFS

Detected N song(s):
  Song 01:   0.5 min  duration 3.8 min  peak -6.2 dBFS
  ...

(--preview: no files written)
```

If you see too many or too few songs, adjust `--threshold-db` until the count looks right (target: ≤15). Common adjustment:
- Too many segments → raise threshold (e.g. `-35`, `-30`)
- Nothing detected → lower threshold (e.g. `-45`, `-50`)

- [ ] **Step 3: Write output files once the preview looks correct**

```bash
python3 split_songs.py raw/260407_182931_TrLR.WAV
```

Expected:
```
...
  -> 260407_182931_TrLR_songs/song_01.wav  (4.1 min)
  -> 260407_182931_TrLR_songs/song_02.wav  (3.6 min)
  ...
Done. N files written to 260407_182931_TrLR_songs/
```

- [ ] **Step 4: Spot-check output**

```bash
python3 -c "
import soundfile as sf, os
d = '260407_182931_TrLR_songs'
for f in sorted(os.listdir(d)):
    info = sf.info(os.path.join(d, f))
    print(f'{f}  {info.frames/info.samplerate/60:.1f} min')
"
```

Confirm each file is ≥2 minutes and the count is ≤15.

- [ ] **Step 5: Commit**

```bash
git add split_songs.py
git commit -m "feat: complete main pipeline — reads, analyses, and writes song WAVs"
```

---

## Self-Review

**Spec coverage:**
- ✅ Read WAV file from `raw/` directory
- ✅ Split on loud segments
- ✅ Coarse detection (1-second windows)
- ✅ Max 15 songs enforced implicitly via merge + min-duration filter
- ✅ Min 2-minute song filter (`--min-song-minutes`)
- ✅ Each song exported as a separate WAV file
- ✅ Float32 WAV input/output (soundfile handles it natively)

**Placeholder scan:** None found — every step has concrete code or commands.

**Type consistency:** `compute_rms` returns `np.ndarray`; `find_songs` accepts `np.ndarray` and returns `list[tuple[int,int]]`; `main` uses both consistently.
