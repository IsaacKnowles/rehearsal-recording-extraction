# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests
pytest

# Run a single test
pytest test_split_songs.py::test_compute_rms_silent -v

# Start the tool
python3 split_songs.py raw/recording.WAV

# Install dependencies (lameenc is not in requirements.txt yet)
pip install -r requirements.txt lameenc
```

## Architecture

Three files, each with one responsibility:

- **`split_songs.py`** — Entry point. Reads the WAV, computes a per-second peak amplitude array (`compute_rms`, despite the name), downsamples it to 2000 points via `np.interp` (`downsample_rms`), writes `segments.json` with an empty segments list, then starts the server. On re-run, skips processing and loads the existing session unless `--reset` is passed.

- **`review_server.py`** — Flask HTTP server. Five endpoints: `GET /` (serves the UI), `GET /audio` (range-request streaming of the source WAV), `GET /state` / `PUT /state` (read/write `segments.json`), `POST /export/<filename>` (slices the WAV and encodes to 320kbps MP3 via `lameenc`).

- **`review_ui.html`** — Self-contained single-file frontend. No build step. Loaded from disk by the server on each `GET /`. All state lives in a JS object `S` (loaded from `GET /state` on startup) and is persisted on every change via `PUT /state`.

## Data flow

`segments.json` is the single source of truth for session state. It contains `overview_rms` (2000 absolute peak values, not normalised) and a `segments` array. The UI reads it once on load and PUTs the full object back on every change. The server also writes to it when marking a segment as exported after `POST /export`.

## Key decisions

- `overview_rms` stores **absolute peak amplitude** (not RMS, not normalised). Values can exceed 1.0 for 32-bit float WAVs recorded above 0 dBFS — the canvas just clips them at full height, matching Audacity's behaviour.
- The `downsample_rms` function uses `np.interp` to cover the full array length. The previous block-mean approach truncated the tail of the recording, causing the waveform to be time-shifted relative to the actual audio.
- Exports are MP3 only (320kbps stereo, `lameenc`). The export filename comes from the segment's name field, sanitised for the filesystem.
