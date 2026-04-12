# Segment Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace auto-export with an interactive review server — `split_songs.py` detects songs, writes `segments.json`, and starts a local Flask server that serves a browser UI for editing segments and exporting individual WAVs on demand.

**Architecture:** Three files: `split_songs.py` (detection only), `review_server.py` (Flask HTTP server), `review_ui.html` (self-contained frontend). The UI talks to the server via `fetch()`; every change is persisted to `segments.json` so sessions survive restarts.

**Tech Stack:** Python 3.9, numpy, soundfile, Flask 2+; vanilla HTML/CSS/Canvas/JS in the frontend.

**Prerequisite:** Prior plan (`2026-04-08-rehearsal-splitter.md` and `2026-04-08-review-ui.md`) must be complete — `split_songs.py` must exist with all functions working.

---

## File Structure

- **Modify:** `split_songs.py` — rework `build_metadata` output schema; remove `_HTML_TEMPLATE`, `generate_html`; add `write_segments_json`, `launch_server`; update `main()` and `parse_args()`
- **Modify:** `test_split_songs.py` — update `build_metadata` tests to new schema; remove `generate_html` test; add `write_segments_json` test
- **Modify:** `requirements.txt` — add `flask>=2.0`
- **Create:** `review_server.py` — Flask app: serve UI, stream audio, read/write state, export WAV
- **Create:** `test_review_server.py` — pytest tests for all server endpoints
- **Create:** `review_ui.html` — complete interactive frontend

---

### Task 1: Rework `build_metadata` and clean up HTML generation

The `build_metadata` output schema changes to match `segments.json`. `_HTML_TEMPLATE`, `generate_html`, and their test are removed since `review_ui.html` replaces them.

**Files:**
- Modify: `split_songs.py`
- Modify: `test_split_songs.py`

- [ ] **Step 1: Update the `build_metadata` tests to the new schema**

In `test_split_songs.py`, replace the three `test_build_metadata_*` tests and remove `test_generate_html_creates_file_with_embedded_data`. The updated tests are:

```python
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
```

Also update the top-level import line — remove `generate_html`:

```python
from split_songs import compute_rms, find_songs, downsample_rms, build_metadata, COLORS
```

And update the `tests` list at the bottom: replace old `test_build_metadata_*` entries and remove `test_generate_html_creates_file_with_embedded_data`, then add the new test names:

```python
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
]
```

- [ ] **Step 2: Run tests — confirm they fail (build_metadata tests fail, generate_html test gone)**

```bash
python3 test_split_songs.py
```

Expected: `FAIL  test_build_metadata_top_level_keys`, `FAIL  test_build_metadata_segment_fields`, `FAIL  test_build_metadata_color_cycles` (old schema still in place). The other 10 tests should PASS.

- [ ] **Step 3: Update `build_metadata` in `split_songs.py`**

Replace the existing `build_metadata` function:

```python
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
```

- [ ] **Step 4: Remove `_HTML_TEMPLATE` and `generate_html` from `split_songs.py`**

Delete the `_HTML_TEMPLATE = r"""..."""` constant (currently lines ~25–438) and the `generate_html` function (the ~10 lines after it). These are replaced by `review_ui.html`.

- [ ] **Step 5: Run tests — confirm all 13 pass**

```bash
python3 test_split_songs.py
```

Expected: all 13 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add split_songs.py test_split_songs.py
git commit -m "feat: rework build_metadata to segments.json schema, remove HTML generation"
```

---

### Task 2: Add `write_segments_json`, `launch_server`, update `main()`

**Files:**
- Modify: `split_songs.py`
- Modify: `test_split_songs.py`

- [ ] **Step 1: Write the failing test for `write_segments_json`**

Append to `test_split_songs.py` (also add `write_segments_json` to the import line and `tests` list):

```python
from split_songs import compute_rms, find_songs, downsample_rms, build_metadata, COLORS, write_segments_json
```

```python
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
```

Add it to the `tests` list at the bottom of `test_split_songs.py`.

- [ ] **Step 2: Run test — confirm it fails**

```bash
python3 test_split_songs.py
```

Expected: `FAIL  test_write_segments_json_creates_file: cannot import name 'write_segments_json'`

- [ ] **Step 3: Add `write_segments_json` and `launch_server` to `split_songs.py`**

Add these two functions after `build_metadata`:

```python
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
```

- [ ] **Step 4: Update `parse_args` in `split_songs.py`**

Remove `--pad-seconds` (no longer needed — WAV export is gone). Add `--port` and `--reset`:

```python
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split rehearsal recording into songs.")
    p.add_argument("input", help="Path to input WAV file")
    p.add_argument("-o", "--output-dir", default=None, help="Output directory (default: <stem>_songs/)")
    p.add_argument("--threshold-db", type=float, default=-40.0, help="Loud/quiet threshold in dBFS (default: -40)")
    p.add_argument("--min-song-minutes", type=float, default=2.0, help="Minimum song duration in minutes (default: 2)")
    p.add_argument("--merge-gap-seconds", type=float, default=20.0, help="Merge segments closer than this (default: 20s)")
    p.add_argument("--window-seconds", type=float, default=1.0, help="RMS window size in seconds (default: 1.0)")
    p.add_argument("--port", type=int, default=5123, help="Review server port (default: 5123)")
    p.add_argument("--preview", action="store_true", help="Print detected songs without writing files or starting server")
    p.add_argument("--reset", action="store_true", help="Re-run detection even if segments.json already exists")
    return p.parse_args()
```

- [ ] **Step 5: Update `main()` in `split_songs.py`**

Replace the entire `main()` function:

```python
def main(args: argparse.Namespace) -> None:
    stem = os.path.splitext(os.path.basename(args.input))[0]
    output_dir = args.output_dir or f"{stem}_songs"
    segments_json = os.path.join(output_dir, "segments.json")

    # Session recovery: skip detection if segments.json already exists
    if os.path.exists(segments_json) and not args.reset:
        print(f"Loading existing session from {segments_json}")
        print("  (pass --reset to re-run detection)")
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

    def to_frames(w: int) -> int:
        return int(w * window_sec * rate)

    print(f"Detected {len(songs)} song(s):")
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

    os.makedirs(output_dir, exist_ok=True)
    metadata = build_metadata(args.input, rate, info.channels, rms, window_sec, songs)
    json_path = write_segments_json(output_dir, metadata)
    print(f"\n  -> {json_path}  (segments)")

    launch_server(output_dir, args.input, args.port)
```

Also add `import sys` back to the top of `split_songs.py` (needed by `launch_server` via `review_server` import chain — actually it's not needed directly, but keep imports clean). Actually `sys` is not needed; skip it.

- [ ] **Step 6: Run tests — confirm all 14 pass**

```bash
python3 test_split_songs.py
```

Expected: all 14 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add split_songs.py test_split_songs.py
git commit -m "feat: add write_segments_json, launch_server, update main to start review server"
```

---

### Task 3: Create `review_server.py` — Flask server with state and audio endpoints

**Files:**
- Create: `review_server.py`
- Create: `test_review_server.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Add Flask to requirements.txt**

```
numpy>=1.24
soundfile>=0.12
pytest>=7.0
flask>=2.0
```

Install it:

```bash
pip3 install flask
```

- [ ] **Step 2: Write the failing tests**

Create `test_review_server.py`:

```python
#!/usr/bin/env python3
"""Tests for review_server.py"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, ".")
import review_server


@pytest.fixture
def server_env(tmp_path):
    """Set up a temp output_dir with segments.json and a short WAV."""
    # Write a 5-second stereo WAV
    wav_path = str(tmp_path / "source.wav")
    samples = (np.random.randn(48000 * 5, 2) * 0.1).astype(np.float32)
    sf.write(wav_path, samples, 48000, subtype="FLOAT")

    # Write a minimal segments.json
    state = {
        "source_file": wav_path,
        "duration_min": 5 / 60,
        "sample_rate": 48000,
        "channels": 2,
        "overview_rms": [0.1] * 2000,
        "segments": [
            {"id": 0, "name": "Song 01", "start_min": 0.0, "end_min": 5/60,
             "color": "#4a9eff", "exported": False}
        ],
    }
    json_path = str(tmp_path / "segments.json")
    with open(json_path, "w") as f:
        json.dump(state, f)

    review_server.OUTPUT_DIR = str(tmp_path)
    review_server.SOURCE_WAV = wav_path
    return {"output_dir": str(tmp_path), "wav_path": wav_path, "state": state}


@pytest.fixture
def client(server_env):
    review_server.app.config["TESTING"] = True
    with review_server.app.test_client() as c:
        yield c


def test_get_state_returns_json(client, server_env):
    resp = client.get("/state")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data["source_file"] == server_env["wav_path"]
    assert len(data["segments"]) == 1


def test_put_state_writes_file(client, server_env):
    new_state = server_env["state"].copy()
    new_state["segments"][0]["name"] = "Renamed Song"
    resp = client.put(
        "/state",
        data=json.dumps(new_state),
        content_type="application/json",
    )
    assert resp.status_code == 200
    json_path = os.path.join(server_env["output_dir"], "segments.json")
    with open(json_path) as f:
        saved = json.load(f)
    assert saved["segments"][0]["name"] == "Renamed Song"


def test_audio_full_request_returns_200(client):
    resp = client.get("/audio")
    assert resp.status_code == 200
    assert resp.content_type == "audio/wav"


def test_audio_range_request_returns_206(client):
    resp = client.get("/audio", headers={"Range": "bytes=0-1023"})
    assert resp.status_code == 206
    assert b"Content-Range" in resp.headers.get("Content-Range", "").encode() or \
           resp.headers.get("Content-Range", "").startswith("bytes 0-1023/")
    assert len(resp.data) == 1024
```

- [ ] **Step 3: Run tests — confirm they fail**

```bash
pytest test_review_server.py -v
```

Expected: all 4 tests FAIL with `ModuleNotFoundError: No module named 'review_server'`.

- [ ] **Step 4: Create `review_server.py`**

```python
#!/usr/bin/env python3
"""HTTP review server for rehearsal recordings.

Usage (direct):
    python3 review_server.py <output_dir> <source_wav> <port>

Called programmatically from split_songs.py via review_server.run().
"""

import json
import os
import sys

from flask import Flask, Response, jsonify, request, send_file

app = Flask(__name__)

OUTPUT_DIR: str = ""
SOURCE_WAV: str = ""


def _segments_path() -> str:
    return os.path.join(OUTPUT_DIR, "segments.json")


def _ui_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_ui.html")


@app.route("/")
def index():
    return send_file(_ui_path(), mimetype="text/html")


@app.route("/state", methods=["GET"])
def get_state():
    with open(_segments_path(), encoding="utf-8") as f:
        return Response(f.read(), mimetype="application/json")


@app.route("/state", methods=["PUT"])
def put_state():
    data = request.get_json(force=True)
    with open(_segments_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, separators=(",", ":"))
    return jsonify({"ok": True})


@app.route("/audio")
def audio():
    file_size = os.path.getsize(SOURCE_WAV)
    range_header = request.headers.get("Range")

    if not range_header:
        response = send_file(SOURCE_WAV, mimetype="audio/wav")
        response.headers["Accept-Ranges"] = "bytes"
        return response

    # Parse "bytes=start-end"
    byte_range = range_header.replace("bytes=", "")
    start_str, _, end_str = byte_range.partition("-")
    start = int(start_str)
    end = int(end_str) if end_str else file_size - 1
    length = end - start + 1

    with open(SOURCE_WAV, "rb") as f:
        f.seek(start)
        chunk = f.read(length)

    return Response(
        chunk,
        status=206,
        mimetype="audio/wav",
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(length),
        },
    )


def run(output_dir: str, source_wav: str, port: int = 5123) -> None:
    """Start the Flask server. Blocks until Ctrl+C."""
    global OUTPUT_DIR, SOURCE_WAV
    OUTPUT_DIR = output_dir
    SOURCE_WAV = source_wav
    print(f"Review server: http://localhost:{port}  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: review_server.py <output_dir> <source_wav> <port>")
        sys.exit(1)
    run(output_dir=sys.argv[1], source_wav=sys.argv[2], port=int(sys.argv[3]))
```

- [ ] **Step 5: Run tests — confirm all 4 pass**

```bash
pytest test_review_server.py -v
```

Expected: all 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add review_server.py test_review_server.py requirements.txt
git commit -m "feat: add review_server with state and audio streaming endpoints"
```

---

### Task 4: Add export endpoint to `review_server.py`

**Files:**
- Modify: `review_server.py`
- Modify: `test_review_server.py`

- [ ] **Step 1: Write the failing test**

Append to `test_review_server.py`:

```python
def test_export_writes_wav_and_marks_exported(client, server_env):
    resp = client.post(
        "/export/song_01.wav",
        data=json.dumps({"segment_id": 0, "start_min": 0.0, "end_min": 5 / 60}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    result = json.loads(resp.data)
    assert os.path.exists(result["path"])

    # Verify exported flag was saved
    json_path = os.path.join(server_env["output_dir"], "segments.json")
    with open(json_path) as f:
        saved = json.load(f)
    assert saved["segments"][0]["exported"] is True

    # Verify it's a valid WAV
    import soundfile as sf
    data, rate = sf.read(result["path"])
    assert rate == 48000
    assert len(data) > 0
```

- [ ] **Step 2: Run test — confirm it fails**

```bash
pytest test_review_server.py::test_export_writes_wav_and_marks_exported -v
```

Expected: FAIL with 404 (route not found).

- [ ] **Step 3: Add the export endpoint to `review_server.py`**

Add this import at the top of `review_server.py` (after the existing imports):

```python
import soundfile as sf
```

Add this route after the `audio()` function:

```python
@app.route("/export/<filename>", methods=["POST"])
def export_segment(filename: str):
    """Export one segment as a WAV file.

    Body: {"segment_id": int, "start_min": float, "end_min": float}
    Returns: {"path": str}
    """
    body = request.get_json(force=True)
    start_min = float(body["start_min"])
    end_min = float(body["end_min"])
    segment_id = int(body["segment_id"])

    samples, rate = sf.read(SOURCE_WAV, dtype="float32", always_2d=True)
    start_frame = int(start_min * 60 * rate)
    end_frame = min(len(samples), int(end_min * 60 * rate))
    chunk = samples[start_frame:end_frame]

    out_path = os.path.join(OUTPUT_DIR, filename)
    sf.write(out_path, chunk, rate, subtype="FLOAT")

    # Mark segment as exported in segments.json
    with open(_segments_path(), encoding="utf-8") as f:
        state = json.load(f)
    for seg in state["segments"]:
        if seg["id"] == segment_id:
            seg["exported"] = True
            break
    with open(_segments_path(), "w", encoding="utf-8") as f:
        json.dump(state, f, separators=(",", ":"))

    return jsonify({"path": out_path})
```

- [ ] **Step 4: Run all server tests — confirm all 5 pass**

```bash
pytest test_review_server.py -v
```

Expected: all 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add review_server.py test_review_server.py
git commit -m "feat: add export endpoint to review_server"
```

---

### Task 5: Create `review_ui.html` — complete interactive frontend

This replaces the old `_HTML_TEMPLATE`. It loads state from `GET /state`, plays audio via `<audio src="/audio">`, persists every change with `PUT /state`, and exports via `POST /export/<filename>`.

**Files:**
- Create: `review_ui.html`

- [ ] **Step 1: Create `review_ui.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rehearsal Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;user-select:none}
canvas{display:block}
/* ── header ── */
.app-header{padding:14px 20px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:10}
.app-header h1{font-size:15px;font-weight:600;color:#fff}
.app-header .meta{color:#555;font-size:12px}
.badge{margin-left:auto;background:#1e3a1e;color:#6cbb6c;border:1px solid #2e5a2e;border-radius:4px;padding:2px 9px;font-size:11px;font-weight:600}
/* ── overview ── */
.overview-section{padding:14px 20px 12px;background:#121212;border-bottom:1px solid #1e1e1e}
.section-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#444;margin-bottom:6px}
.overview-wrap{position:relative;height:68px;border-radius:4px;overflow:hidden;background:#080808;cursor:crosshair}
#ovCanvas{width:100%;height:100%}
#ovGhost{position:absolute;top:0;height:100%;background:#ffffff15;border-left:2px solid #fff4;border-right:2px solid #fff4;pointer-events:none;display:none}
.overview-hint{margin-top:5px;font-size:10px;color:#383838}
.overview-times{display:flex;justify-content:space-between;margin-top:3px;color:#383838;font-size:10px;font-variant-numeric:tabular-nums}
/* ── songs section ── */
.songs-section{padding:14px 20px 80px}
.add-btn{display:inline-flex;align-items:center;gap:6px;background:#1a2a1a;border:1px dashed #2e5a2e;color:#5abb6a;border-radius:5px;padding:6px 14px;font-size:12px;cursor:pointer;margin-bottom:14px;transition:all .15s}
.add-btn:hover{background:#1e3a1e}
/* ── card ── */
.song-card{background:#161616;border:1px solid #242424;border-radius:6px;margin-bottom:10px;overflow:visible;transition:border-color .15s}
.song-card.focused{border-color:#4a9eff55}
.card-header{display:flex;align-items:center;padding:9px 12px;gap:10px;border-bottom:1px solid #1e1e1e}
.song-num{width:20px;height:20px;border-radius:3px;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.song-name{background:transparent;border:none;color:#ccc;font-size:13px;font-weight:500;flex:1;outline:none;font-family:inherit;user-select:text}
.song-name:focus{background:#1e1e1e;border-radius:3px;padding:2px 5px;margin:-2px -5px}
.btn-export{background:#1a3a1a;border:1px solid #2e6a2e;color:#5abb6a;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer;white-space:nowrap;transition:all .12s}
.btn-export:hover{background:#1e4a1e}
.btn-export.exported{background:#111;border-color:#2a3a2a;color:#3a6a3a;cursor:pointer}
.btn-del{width:24px;height:24px;background:none;border:1px solid #2a2a2a;border-radius:4px;color:#555;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;flex-shrink:0}
.btn-del:hover{border-color:#7a3030;color:#e06060;background:#1e1010}
.card-body{display:grid;grid-template-columns:1fr 1fr}
/* ── context col ── */
.context-col{border-right:1px solid #1e1e1e;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
.ctx-label{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#383838}
.ctx-outer{position:relative}
.ctx-wrap{height:56px;background:#080808;border-radius:3px;overflow:hidden;position:relative}
.ctx-canvas{width:100%;height:100%}
.seg-region{position:absolute;top:0;bottom:0;pointer-events:none}
.seg-fill{position:absolute;inset:0;border-radius:1px}
.seg-border{position:absolute;inset:0;border-radius:1px;border:1.5px solid transparent}
.seg-body{position:absolute;top:0;bottom:0;left:10px;right:10px;cursor:grab;pointer-events:all}
.seg-body:active{cursor:grabbing}
.handle{position:absolute;top:0;height:56px;width:18px;cursor:ew-resize;display:flex;align-items:center;justify-content:center;z-index:20}
.handle-bar{width:3px;height:22px;border-radius:2px}
.handle.left{transform:translateX(-9px)}
.handle.right{transform:translateX(-9px)}
/* ── time fields ── */
.time-row{display:flex;align-items:center;gap:8px}
.time-row label{font-size:9px;color:#444;text-transform:uppercase;letter-spacing:.06em;white-space:nowrap;width:28px}
.time-sep{color:#333;font-size:11px}
.time-field{display:flex;align-items:center;border:1px solid #2a2a2a;border-radius:4px;overflow:hidden;background:#0e0e0e}
.time-field:focus-within{border-color:#4a9eff55}
.time-input{background:transparent;border:none;color:#aaa;font-size:11px;font-family:monospace;width:46px;padding:3px 4px;text-align:center;outline:none;user-select:text}
.time-input:focus{color:#ccc}
.steppers{display:flex;flex-direction:column;border-left:1px solid #222}
.step-btn{width:16px;height:13px;background:none;border:none;color:#444;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:8px;padding:0;transition:color .1s,background .1s}
.step-btn:hover{color:#aaa;background:#1e1e1e}
.step-btn+.step-btn{border-top:1px solid #222}
/* ── bottom row ── */
.bottom-row{display:flex;align-items:center;justify-content:space-between}
.status-saved{font-size:10px;color:#444;display:flex;align-items:center;gap:4px}
.status-dot{width:6px;height:6px;border-radius:50%;background:#5abb6a;flex-shrink:0}
.key-hint{font-size:9px;color:#333;display:flex;align-items:center;gap:3px}
.key-hint kbd{background:#1a1a1a;border:1px solid #2a2a2a;border-radius:3px;padding:1px 4px;font-size:9px;color:#444;font-family:monospace}
.song-card.focused .key-hint{color:#555}
.song-card.focused .key-hint kbd{border-color:#4a9eff44;color:#4a9eff88}
/* ── playback col ── */
.playback-col{padding:10px 12px;display:flex;flex-direction:column;gap:8px;justify-content:center}
.wave-wrap{position:relative;height:44px;background:#0a0a0a;border-radius:3px;overflow:hidden;cursor:pointer}
.playhead{position:absolute;top:0;bottom:0;width:1.5px;pointer-events:none;left:0}
.transport{display:flex;align-items:center;gap:8px}
.btn-play{width:26px;height:26px;border-radius:50%;background:#1e3050;border:none;color:#4a9eff;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;padding-left:1px;transition:background .12s}
.btn-play:hover{background:#2a4060}
.btn-play.playing{background:#1e3020;color:#5abb6a}
.seek-wrap{flex:1;height:3px;background:#222;border-radius:2px;cursor:pointer}
.seek-fill{height:100%;border-radius:2px;pointer-events:none}
.time-disp{color:#444;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap;min-width:72px;text-align:right}
/* ── delete warning ── */
.del-warn{background:#1e1010;border:1px solid #5a2020;border-radius:4px;padding:8px 12px;font-size:11px;color:#e06060;margin-bottom:10px;display:flex;align-items:center;justify-content:space-between;gap:10px}
.del-warn-btns{display:flex;gap:6px}
.del-confirm{background:#3a1a1a;border:1px solid #6a2a2a;color:#e06060;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer}
.del-cancel{background:none;border:1px solid #333;color:#666;border-radius:4px;padding:3px 10px;font-size:11px;cursor:pointer}
</style>
</head>
<body>

<div class="app-header">
  <h1 id="hdrFilename"></h1>
  <span class="meta" id="hdrMeta"></span>
  <span class="badge" id="hdrBadge"></span>
</div>

<div class="overview-section">
  <div class="section-label">Full Recording</div>
  <div class="overview-wrap" id="ovWrap">
    <canvas id="ovCanvas"></canvas>
    <div id="ovGhost"></div>
  </div>
  <div class="overview-hint">Click + drag to add a new segment</div>
  <div class="overview-times" id="ovTimes"></div>
</div>

<div class="songs-section">
  <button class="add-btn" id="addBtn">＋ Add segment</button>
  <div id="cardsContainer"></div>
</div>

<audio id="mainAudio" src="/audio" preload="auto"></audio>

<script>
// ── State ────────────────────────────────────────────────────
let S = null;          // full state object from /state
let nextId = 0;
let focusedId = null;
let playingId = null;
let saveTimer = null;
const audio = document.getElementById('mainAudio');

// ── Boot ─────────────────────────────────────────────────────
fetch('/state').then(r => r.json()).then(state => {
  S = state;
  nextId = S.segments.reduce((m, s) => Math.max(m, s.id), -1) + 1;
  initHeader();
  initOverview();
  initCards();
});

function initHeader() {
  const fname = S.source_file.split(/[\\/]/).pop();
  document.getElementById('hdrFilename').textContent = fname;
  document.getElementById('hdrMeta').textContent =
    `${S.duration_min.toFixed(1)} min · ${(S.sample_rate/1000).toFixed(0)}kHz · ${S.channels}ch`;
  document.getElementById('hdrBadge').textContent = `${S.segments.length} segments`;
  // time labels
  const times = document.getElementById('ovTimes');
  const steps = 5;
  for (let i = 0; i <= steps; i++) {
    const m = (S.duration_min * i / steps);
    const span = document.createElement('span');
    span.textContent = minToMMSS(m);
    times.appendChild(span);
  }
}

// ── Persistence ──────────────────────────────────────────────
function saveState() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    fetch('/state', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(S),
    });
  }, 200);
}

// ── Time utils ───────────────────────────────────────────────
function minToFrac(m) { return m / S.duration_min; }
function fracToMin(f) { return f * S.duration_min; }

function minToMMSS(m) {
  const s = Math.round(m * 60);
  return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;
}
function mmssToMin(str) {
  const p = str.split(':');
  if (p.length !== 2) return null;
  const m = parseInt(p[0]), s = parseInt(p[1]);
  if (isNaN(m) || isNaN(s)) return null;
  return (m * 60 + s) / 60;
}

const STEP_MIN = 1 / 60; // 1 second in minutes

// ── Focus ────────────────────────────────────────────────────
function setFocus(id) {
  if (focusedId !== null) document.getElementById(`card${focusedId}`)?.classList.remove('focused');
  focusedId = id;
  if (id !== null) document.getElementById(`card${id}`)?.classList.add('focused');
}

// ── Arrow keys ───────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (focusedId === null) return;
  if (document.activeElement?.tagName === 'INPUT') return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const seg = S.segments.find(s => s.id === focusedId);
  if (!seg) return;
  const delta = (e.shiftKey ? 10 : 1) * STEP_MIN;
  const dur = seg.end_min - seg.start_min;
  if (e.key === 'ArrowLeft') {
    seg.start_min = Math.max(0, seg.start_min - delta);
    seg.end_min = seg.start_min + dur;
    if (seg.end_min > S.duration_min) { seg.end_min = S.duration_min; seg.start_min = S.duration_min - dur; }
  } else {
    seg.end_min = Math.min(S.duration_min, seg.end_min + delta);
    seg.start_min = seg.end_min - dur;
    if (seg.start_min < 0) { seg.start_min = 0; seg.end_min = dur; }
  }
  refreshSeg(seg.id);
  saveState();
});

// ── Drawing helpers ──────────────────────────────────────────
function setupCanvas(canvas, wrap) {
  const dpr = devicePixelRatio || 1;
  const W = wrap.clientWidth, H = wrap.clientHeight;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return [ctx, W, H];
}

function drawRMSBars(ctx, rms, W, H, color) {
  const mid = H / 2, bw = W / rms.length;
  ctx.fillStyle = color;
  for (let i = 0; i < rms.length; i++) {
    const bh = rms[i] * mid * 0.88;
    ctx.fillRect(i * bw, mid - bh, Math.max(1, bw - 0.5), bh * 2);
  }
}

// ── Overview ─────────────────────────────────────────────────
function initOverview() {
  drawOverview();
  setupOverviewInteraction();
}

function drawOverview() {
  const wrap = document.getElementById('ovWrap');
  const canvas = document.getElementById('ovCanvas');
  const [ctx, W, H] = setupCanvas(canvas, wrap);
  ctx.fillStyle = '#080808'; ctx.fillRect(0, 0, W, H);
  drawRMSBars(ctx, S.overview_rms, W, H, '#2a2a2a');
  for (const seg of S.segments) {
    const sx = minToFrac(seg.start_min) * W;
    const sw = (minToFrac(seg.end_min) - minToFrac(seg.start_min)) * W;
    ctx.fillStyle = seg.color + (seg.id === focusedId ? '44' : (seg.exported ? '35' : '28'));
    ctx.fillRect(sx, 0, sw, H);
    ctx.fillStyle = seg.color + (seg.exported ? 'aa' : 'cc');
    ctx.fillRect(sx, 0, 2, H);
    ctx.fillRect(sx + sw - 2, 0, 2, H);
  }
}

function setupOverviewInteraction() {
  const ovWrap = document.getElementById('ovWrap');
  const ghost = document.getElementById('ovGhost');
  let dragging = false, startFrac = 0;
  ovWrap.addEventListener('mousedown', e => {
    const r = ovWrap.getBoundingClientRect();
    startFrac = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    dragging = true;
    ghost.style.display = 'block';
    ghost.style.left = (startFrac * 100) + '%';
    ghost.style.width = '0';
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    const r = ovWrap.getBoundingClientRect();
    const cur = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const l = Math.min(startFrac, cur), rr = Math.max(startFrac, cur);
    ghost.style.left = (l * 100) + '%';
    ghost.style.width = ((rr - l) * 100) + '%';
  });
  document.addEventListener('mouseup', e => {
    if (!dragging) return;
    dragging = false;
    ghost.style.display = 'none';
    const r = ovWrap.getBoundingClientRect();
    const end = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const l = Math.min(startFrac, end), rr = Math.max(startFrac, end);
    if (rr - l > 0.005) addSegment(fracToMin(l), fracToMin(rr));
  });
}

// ── Cards ────────────────────────────────────────────────────
function initCards() {
  document.getElementById('cardsContainer').innerHTML = '';
  S.segments.forEach(seg => buildCard(seg));
  document.getElementById('hdrBadge').textContent = `${S.segments.length} segments`;
}

function buildCard(seg) {
  const card = document.createElement('div');
  card.className = 'song-card';
  card.id = `card${seg.id}`;
  const exportLabel = seg.exported ? '✓ Exported' : '▼ Export WAV';
  card.innerHTML = `
    <div class="card-header">
      <div class="song-num" style="background:${seg.color}22;color:${seg.color}">${S.segments.indexOf(seg)+1}</div>
      <input class="song-name" id="name${seg.id}" value="${seg.name}" />
      <button class="btn-export${seg.exported?' exported':''}" id="export${seg.id}">${exportLabel}</button>
      <button class="btn-del" id="del${seg.id}">✕</button>
    </div>
    <div class="card-body">
      <div class="context-col">
        <div class="ctx-label">Position — drag to move · handles to resize</div>
        <div class="ctx-outer" style="position:relative">
          <div class="ctx-wrap" id="ctxWrap${seg.id}">
            <canvas class="ctx-canvas" id="ctxCanvas${seg.id}"></canvas>
            <div class="seg-region" id="region${seg.id}" style="left:0;width:0;pointer-events:none">
              <div class="seg-fill" style="background:${seg.color}18"></div>
              <div class="seg-border" style="border-color:${seg.color}"></div>
              <div class="seg-body" id="segBody${seg.id}"></div>
            </div>
          </div>
          <div class="handle left" id="lh${seg.id}" style="position:absolute;top:0">
            <div class="handle-bar" style="background:${seg.color}"></div>
          </div>
          <div class="handle right" id="rh${seg.id}" style="position:absolute;top:0">
            <div class="handle-bar" style="background:${seg.color}"></div>
          </div>
        </div>
        <div class="time-row">
          <label>Start</label>
          <div class="time-field">
            <input class="time-input" id="si${seg.id}" value="${minToMMSS(seg.start_min)}" />
            <div class="steppers">
              <button class="step-btn" data-seg="${seg.id}" data-field="start" data-dir="1">▲</button>
              <button class="step-btn" data-seg="${seg.id}" data-field="start" data-dir="-1">▼</button>
            </div>
          </div>
          <span class="time-sep">→</span>
          <label>End</label>
          <div class="time-field">
            <input class="time-input" id="ei${seg.id}" value="${minToMMSS(seg.end_min)}" />
            <div class="steppers">
              <button class="step-btn" data-seg="${seg.id}" data-field="end" data-dir="1">▲</button>
              <button class="step-btn" data-seg="${seg.id}" data-field="end" data-dir="-1">▼</button>
            </div>
          </div>
        </div>
        <div class="bottom-row">
          <div class="status-saved"><span class="status-dot"></span>Autosaved</div>
          <div class="key-hint"><kbd>←</kbd><kbd>→</kbd> 1s &nbsp;<kbd>⇧</kbd>10s</div>
        </div>
      </div>
      <div class="playback-col">
        <div class="wave-wrap" id="waveWrap${seg.id}" style="cursor:pointer">
          <canvas id="waveCanvas${seg.id}" style="width:100%;height:44px"></canvas>
          <div class="playhead" id="ph${seg.id}" style="background:${seg.color}"></div>
        </div>
        <div class="transport">
          <button class="btn-play" id="play${seg.id}">▶</button>
          <div class="seek-wrap" id="seekWrap${seg.id}">
            <div class="seek-fill" id="seekFill${seg.id}" style="background:${seg.color};width:0%"></div>
          </div>
          <div class="time-disp" id="td${seg.id}">0:00 / ${minToMMSS(seg.end_min - seg.start_min)}</div>
        </div>
      </div>
    </div>`;

  document.getElementById('cardsContainer').appendChild(card);

  // Focus on click
  card.addEventListener('mousedown', e => {
    if (!e.target.closest('input') && !e.target.closest('button')) setFocus(seg.id);
  });

  // Name change
  document.getElementById(`name${seg.id}`).addEventListener('change', e => {
    seg.name = e.target.value; saveState();
  });

  // Time inputs
  document.getElementById(`si${seg.id}`).addEventListener('change', e => {
    const v = mmssToMin(e.target.value);
    if (v !== null && v >= 0 && v < seg.end_min - STEP_MIN) {
      seg.start_min = v; refreshSeg(seg.id); saveState();
    } else { e.target.value = minToMMSS(seg.start_min); }
  });
  document.getElementById(`ei${seg.id}`).addEventListener('change', e => {
    const v = mmssToMin(e.target.value);
    if (v !== null && v > seg.start_min + STEP_MIN && v <= S.duration_min) {
      seg.end_min = v; refreshSeg(seg.id); saveState();
    } else { e.target.value = minToMMSS(seg.end_min); }
  });

  // Steppers
  card.querySelectorAll('.step-btn').forEach(btn => {
    let iv = null;
    function step() {
      const field = btn.dataset.field, dir = parseInt(btn.dataset.dir);
      setFocus(seg.id);
      if (field === 'start') {
        const nv = seg.start_min + dir * STEP_MIN;
        if (nv >= 0 && nv < seg.end_min - STEP_MIN) { seg.start_min = nv; refreshSeg(seg.id); saveState(); }
      } else {
        const nv = seg.end_min + dir * STEP_MIN;
        if (nv > seg.start_min + STEP_MIN && nv <= S.duration_min) { seg.end_min = nv; refreshSeg(seg.id); saveState(); }
      }
    }
    btn.addEventListener('mousedown', e => { e.preventDefault(); step(); iv = setInterval(step, 120); });
    document.addEventListener('mouseup', () => { clearInterval(iv); iv = null; });
  });

  // Export
  document.getElementById(`export${seg.id}`).addEventListener('click', () => {
    const filename = `song_${String(seg.id + 1).padStart(2,'0')}.wav`;
    fetch(`/export/${filename}`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({segment_id: seg.id, start_min: seg.start_min, end_min: seg.end_min}),
    }).then(r => r.json()).then(() => {
      seg.exported = true;
      const btn = document.getElementById(`export${seg.id}`);
      btn.textContent = '✓ Exported';
      btn.classList.add('exported');
      drawOverview();
    });
  });

  // Delete
  document.getElementById(`del${seg.id}`).addEventListener('click', () => {
    if (seg.exported) {
      // Show inline warning
      const warn = document.createElement('div');
      warn.className = 'del-warn';
      warn.innerHTML = `<span>This won't delete the exported WAV file.</span>
        <div class="del-warn-btns">
          <button class="del-cancel">Cancel</button>
          <button class="del-confirm">Delete segment</button>
        </div>`;
      card.insertAdjacentElement('afterend', warn);
      warn.querySelector('.del-cancel').onclick = () => warn.remove();
      warn.querySelector('.del-confirm').onclick = () => { warn.remove(); removeSegment(seg.id); };
    } else {
      removeSegment(seg.id);
    }
  });

  // Play/pause
  document.getElementById(`play${seg.id}`).addEventListener('click', () => togglePlay(seg.id));

  // Seek via waveform click
  document.getElementById(`waveWrap${seg.id}`).addEventListener('click', e => {
    const frac = e.offsetX / e.currentTarget.clientWidth;
    seekSeg(seg.id, frac);
  });

  // Seek via seek bar click
  document.getElementById(`seekWrap${seg.id}`).addEventListener('click', e => {
    const frac = e.offsetX / e.currentTarget.clientWidth;
    seekSeg(seg.id, frac);
  });

  // Drag handles
  setTimeout(() => {
    refreshSeg(seg.id);
    drawMiniWave(seg.id);
    setupDrags(seg);
  }, 0);
}

function removeSegment(segId) {
  if (playingId === segId) stopAudio();
  if (focusedId === segId) setFocus(null);
  S.segments = S.segments.filter(s => s.id !== segId);
  document.getElementById(`card${segId}`)?.remove();
  saveState();
  drawOverview();
  renumber();
}

function renumber() {
  S.segments.forEach((seg, i) => {
    const el = document.querySelector(`#card${seg.id} .song-num`);
    if (el) el.textContent = i + 1;
  });
  document.getElementById('hdrBadge').textContent = `${S.segments.length} segments`;
}

// ── Refresh a segment's visuals ──────────────────────────────
function refreshSeg(segId) {
  const seg = S.segments.find(s => s.id === segId); if (!seg) return;
  drawCtxCanvas(segId);
  updateHandles(segId);
  updateTimeInputs(segId);
  drawOverview();
}

function drawCtxCanvas(segId) {
  const seg = S.segments.find(s => s.id === segId); if (!seg) return;
  const wrap = document.getElementById(`ctxWrap${segId}`);
  const canvas = document.getElementById(`ctxCanvas${segId}`);
  if (!wrap || !canvas) return;
  const [ctx, W, H] = setupCanvas(canvas, wrap);
  ctx.fillStyle = '#080808'; ctx.fillRect(0, 0, W, H);
  drawRMSBars(ctx, S.overview_rms, W, H, '#292929');
  const sf_ = minToFrac(seg.start_min), ef = minToFrac(seg.end_min);
  const sx = sf_ * W, sw = (ef - sf_) * W;
  ctx.fillStyle = seg.color + '22'; ctx.fillRect(sx, 0, sw, H);
  ctx.strokeStyle = seg.color; ctx.lineWidth = 1.5;
  ctx.strokeRect(sx + 0.75, 0.75, sw - 1.5, H - 1.5);
}

function drawMiniWave(segId) {
  const seg = S.segments.find(s => s.id === segId); if (!seg) return;
  const wrap = document.getElementById(`waveWrap${segId}`);
  const canvas = document.getElementById(`waveCanvas${segId}`);
  if (!wrap || !canvas) return;
  const [ctx, W, H] = setupCanvas(canvas, wrap);
  ctx.fillStyle = '#0a0a0a'; ctx.fillRect(0, 0, W, H);
  const i0 = Math.floor(minToFrac(seg.start_min) * S.overview_rms.length);
  const i1 = Math.floor(minToFrac(seg.end_min) * S.overview_rms.length);
  const slice = S.overview_rms.slice(i0, i1); if (!slice.length) return;
  const peak = Math.max(...slice, 0.01);
  const normed = slice.map(v => v / peak);
  drawRMSBars(ctx, normed, W, H, seg.color + 'aa');
}

function updateHandles(segId) {
  const seg = S.segments.find(s => s.id === segId); if (!seg) return;
  const wrap = document.getElementById(`ctxWrap${segId}`); if (!wrap) return;
  const W = wrap.clientWidth;
  const lx = minToFrac(seg.start_min) * W, rx = minToFrac(seg.end_min) * W;
  const lh = document.getElementById(`lh${segId}`);
  const rh = document.getElementById(`rh${segId}`);
  const region = document.getElementById(`region${segId}`);
  if (lh) lh.style.left = lx + 'px';
  if (rh) rh.style.left = rx + 'px';
  if (region) { region.style.left = lx + 'px'; region.style.width = (rx - lx) + 'px'; }
}

function updateTimeInputs(segId) {
  const seg = S.segments.find(s => s.id === segId); if (!seg) return;
  const si = document.getElementById(`si${segId}`);
  const ei = document.getElementById(`ei${segId}`);
  if (si && document.activeElement !== si) si.value = minToMMSS(seg.start_min);
  if (ei && document.activeElement !== ei) ei.value = minToMMSS(seg.end_min);
  const td = document.getElementById(`td${segId}`);
  if (td) td.textContent = `0:00 / ${minToMMSS(seg.end_min - seg.start_min)}`;
}

// ── Drag handles ─────────────────────────────────────────────
function setupDrags(seg) {
  const wrap = document.getElementById(`ctxWrap${seg.id}`);
  function startDrag(mode, e) {
    e.preventDefault(); e.stopPropagation();
    setFocus(seg.id);
    const rect = wrap.getBoundingClientRect();
    const dur = seg.end_min - seg.start_min;
    const startX = e.clientX, origS = seg.start_min, origE = seg.end_min;
    function onMove(ev) {
      const dx = ((ev.clientX - startX) / rect.width) * S.duration_min;
      if (mode === 'left') {
        seg.start_min = Math.max(0, Math.min(origS + dx, seg.end_min - STEP_MIN));
      } else if (mode === 'right') {
        seg.end_min = Math.max(seg.start_min + STEP_MIN, Math.min(S.duration_min, origE + dx));
      } else {
        let ns = origS + dx, ne = origE + dx;
        if (ns < 0) { ns = 0; ne = dur; }
        if (ne > S.duration_min) { ne = S.duration_min; ns = S.duration_min - dur; }
        seg.start_min = ns; seg.end_min = ne;
      }
      refreshSeg(seg.id); drawMiniWave(seg.id);
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      saveState();
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }
  document.getElementById(`lh${seg.id}`)?.addEventListener('mousedown', e => startDrag('left', e));
  document.getElementById(`rh${seg.id}`)?.addEventListener('mousedown', e => startDrag('right', e));
  const body = document.getElementById(`segBody${seg.id}`);
  if (body) { body.style.pointerEvents = 'all'; body.addEventListener('mousedown', e => startDrag('move', e)); }
}

// ── Audio playback ───────────────────────────────────────────
function stopAudio() {
  if (playingId === null) return;
  audio.pause();
  const btn = document.getElementById(`play${playingId}`);
  if (btn) { btn.textContent = '▶'; btn.classList.remove('playing'); }
  playingId = null;
}

function togglePlay(segId) {
  const seg = S.segments.find(s => s.id === segId);
  if (playingId === segId) { stopAudio(); return; }
  stopAudio();
  audio.currentTime = seg.start_min * 60;
  audio.play();
  playingId = segId;
  const btn = document.getElementById(`play${segId}`);
  if (btn) { btn.textContent = '⏸'; btn.classList.add('playing'); }
}

function seekSeg(segId, frac) {
  const seg = S.segments.find(s => s.id === segId);
  audio.currentTime = seg.start_min * 60 + frac * (seg.end_min - seg.start_min) * 60;
  if (playingId !== segId) { stopAudio(); audio.play(); playingId = segId; }
  const btn = document.getElementById(`play${segId}`);
  if (btn) { btn.textContent = '⏸'; btn.classList.add('playing'); }
}

audio.addEventListener('timeupdate', () => {
  if (playingId === null) return;
  const seg = S.segments.find(s => s.id === playingId);
  if (!seg) return;
  const segDurSec = (seg.end_min - seg.start_min) * 60;
  const elapsed = Math.max(0, audio.currentTime - seg.start_min * 60);
  if (elapsed >= segDurSec) { stopAudio(); return; }
  const frac = elapsed / segDurSec;
  const ph = document.getElementById(`ph${playingId}`);
  const sf_ = document.getElementById(`seekFill${playingId}`);
  const wrap = document.getElementById(`waveWrap${playingId}`);
  if (ph && wrap) ph.style.left = (frac * wrap.clientWidth) + 'px';
  if (sf_) sf_.style.width = (frac * 100) + '%';
  const td = document.getElementById(`td${playingId}`);
  if (td) td.textContent = `${minToMMSS(elapsed/60)} / ${minToMMSS(seg.end_min - seg.start_min)}`;
});

// ── Add segment ──────────────────────────────────────────────
function addSegment(startMin, endMin) {
  const seg = {
    id: nextId++,
    name: `Song ${String(nextId).padStart(2,'0')}`,
    start_min: Math.round(startMin * 3600) / 3600,
    end_min: Math.round(endMin * 3600) / 3600,
    color: ['#4a9eff','#5abb6a','#e07840','#b06adb','#e0c040','#40b8c8','#e05070'][S.segments.length % 7],
    exported: false,
  };
  S.segments.push(seg);
  buildCard(seg);
  saveState();
  drawOverview();
  setFocus(seg.id);
  document.getElementById('hdrBadge').textContent = `${S.segments.length} segments`;
}

document.getElementById('addBtn').addEventListener('click', () => {
  const used = S.segments.map(s => [s.start_min, s.end_min]).sort((a,b) => a[0]-b[0]);
  let best = [0, S.duration_min * 0.15];
  let prev = 0;
  for (const [s, e] of used) {
    if (s - prev > S.duration_min * 0.1) { best = [prev + S.duration_min*0.01, s - S.duration_min*0.01]; break; }
    prev = e;
  }
  if (S.duration_min - prev > S.duration_min * 0.1)
    best = [prev + S.duration_min*0.01, Math.min(S.duration_min, prev + S.duration_min*0.15)];
  addSegment(best[0], best[1]);
});

// ── Resize ───────────────────────────────────────────────────
window.addEventListener('resize', () => {
  drawOverview();
  S.segments.forEach(seg => { refreshSeg(seg.id); drawMiniWave(seg.id); });
});
</script>
</body>
</html>
```

- [ ] **Step 2: Verify `review_ui.html` is served by the running server**

With a `segments.json` present in an output directory, start the server manually to confirm the UI loads:

```bash
python3 review_server.py 260407_182931_TrLR_songs raw/260407_182931_TrLR.WAV 5123
```

Open http://localhost:5123 — confirm:
- Header shows filename, duration, channel info
- Overview waveform renders with segment bands
- Cards render with drag handles, time inputs, ▲▼ steppers
- Dragging handles and middle of segment box works, time inputs update live
- Arrow keys nudge selected segment 1 second
- Play button starts audio at segment start, seek bar updates
- Export WAV button shows "✓ Exported" after click and server writes WAV
- Delete on unexported segment removes card immediately
- Delete on exported segment shows inline warning first
- All changes persist to segments.json (check file after editing)

Kill the server when done (Ctrl+C).

- [ ] **Step 3: Run end-to-end via split_songs.py**

```bash
python3 split_songs.py raw/260407_182931_TrLR.WAV --reset
```

Expected:
```
Reading raw/260407_182931_TrLR.WAV ...
  93.4 min  |  48000 Hz  2ch  WAV

Threshold: -40.0 dBFS  |  Min song: 2.0 min  |  Merge gap: 20.0 s
Recording loudness range: ...

Detected 9 song(s):
  Song 01: ...
  ...

  -> 260407_182931_TrLR_songs/segments.json  (segments)

Review server: http://localhost:5123  (Ctrl+C to stop)
```

Browser should open automatically. Verify UI loads with 9 detected segments.

Kill server, re-run without `--reset`:

```bash
python3 split_songs.py raw/260407_182931_TrLR.WAV
```

Expected: `Loading existing session from 260407_182931_TrLR_songs/segments.json` — skips detection, server starts immediately.

- [ ] **Step 4: Run all tests**

```bash
python3 test_split_songs.py && pytest test_review_server.py -v
```

Expected: all 14 split_songs tests PASS, all 5 server tests PASS.

- [ ] **Step 5: Commit**

```bash
git add review_ui.html
git commit -m "feat: add review_ui.html — interactive segment editor frontend"
```

---

## Self-Review

**Spec coverage:**
- ✅ No auto-export — WAV export loop removed from `main()` — Task 2
- ✅ `segments.json` written on first run — Task 2 `write_segments_json`
- ✅ Session recovery: skip detection if `segments.json` exists — Task 2 `main()`
- ✅ `--reset` flag to force re-detection — Task 2 `parse_args`
- ✅ `GET /`, `GET /state`, `PUT /state`, `GET /audio`, `POST /export/<filename>` — Tasks 3–4
- ✅ HTTP range request support for audio seeking — Task 3
- ✅ Export marks `exported: true` in segments.json — Task 4
- ✅ Overview strip updates live as segments are dragged — Task 5 `drawOverview` called in `refreshSeg`
- ✅ Click+drag on overview to add segment — Task 5 `setupOverviewInteraction`
- ✅ Drag handles to resize — Task 5 `setupDrags` (left/right mode)
- ✅ Drag middle to move — Task 5 `setupDrags` (move mode)
- ✅ ▲▼ stepper buttons, 1 second each, hold to repeat — Task 5 steppers with `setInterval`
- ✅ Typed mm:ss inputs — Task 5 time input change handlers
- ✅ ← → arrow key nudge (1s / Shift 10s) — Task 5 keydown handler
- ✅ Autosave: every change calls `saveState()` — Task 5, all mutation handlers
- ✅ Export button calls `POST /export/<filename>`, shows ✓ Exported — Task 5 export handler
- ✅ Delete with no export: removes immediately — Task 5 `removeSegment`
- ✅ Delete with exported file: inline warning — Task 5 delete handler
- ✅ Plays from `/audio` (original file, no pre-exported WAV) — Task 5 `<audio src="/audio">`
- ✅ Playhead and seek bar update during playback — Task 5 `timeupdate` handler
- ✅ Seek via waveform click or seek bar click — Task 5 `seekSeg`

**Placeholder scan:** None found. All steps have complete code.

**Type consistency:** `seg.start_min` / `seg.end_min` (minutes, float) used consistently throughout Tasks 2–5. `minToFrac` / `fracToMin` conversions used only for canvas positioning. `STEP_MIN = 1/60` (1 second in minutes) used in arrow keys, steppers, and input validation.
