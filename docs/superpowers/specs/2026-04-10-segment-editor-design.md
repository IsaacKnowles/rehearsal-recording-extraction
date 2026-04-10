# Rehearsal Splitter — Segment Editor Design

## Goal

Replace the auto-export flow with an interactive review server. Running `split_songs.py` detects songs, starts a local HTTP server, and opens a browser UI where the user adjusts segment boundaries, renames tracks, and exports individual WAV files on demand.

---

## Architecture

Three files, each with one responsibility:

- **`split_songs.py`** — detection only. Runs RMS analysis, detects songs, writes `segments.json` with initial detected segments, starts `review_server.py` as a subprocess, and opens the browser.
- **`review_server.py`** — HTTP server. Serves the UI, streams the original WAV for playback, reads/writes `segments.json`, and handles per-segment WAV export.
- **`review_ui.html`** — self-contained frontend, read from disk at server startup. Communicates with the server via `fetch()`.

No WAV files are written until the user clicks "Export WAV" on a card.

---

## File Changes

- **Modify:** `split_songs.py` — remove WAV export loop, `generate_html` call, `_HTML_TEMPLATE`, and `generate_html()`; add `write_segments_json()` and `launch_server()` at end of `main()`. `build_metadata()` is kept and reused by `write_segments_json()` — its `songs` dict format changes slightly (drops `waveform`, adds `exported: false`).
- **Create:** `review_server.py` — HTTP server, all endpoints, WAV export logic
- **Create:** `review_ui.html` — complete frontend (replaces `_HTML_TEMPLATE` in `split_songs.py`)

---

## Server API

All endpoints on `http://localhost:<port>` (default port 5123, configurable via `--port`).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves `review_ui.html` |
| `GET` | `/audio` | Streams the original WAV with HTTP range request support (enables browser seeking) |
| `GET` | `/state` | Returns `segments.json` contents as JSON |
| `PUT` | `/state` | Receives full segments JSON body, writes to `segments.json`. Called on every UI change. |
| `POST` | `/export/<filename>` | Exports one segment as WAV. Body: `{"start_min": 12.2, "end_min": 17.4}`. Returns `{"path": "..."}`. |

The server runs until the user kills it (Ctrl+C). On startup it prints the URL and opens the browser automatically via `webbrowser.open()`.

---

## Data Model (`segments.json`)

Written by `split_songs.py` on first run. Owned and updated by the UI thereafter.

```json
{
  "source_file": "raw/260407_182931_TrLR.WAV",
  "duration_min": 93.4,
  "sample_rate": 48000,
  "channels": 2,
  "overview_rms": [0.012, 0.45, 0.61, ...],
  "segments": [
    {
      "id": 0,
      "name": "Song 01",
      "start_min": 12.2,
      "end_min": 17.4,
      "color": "#4a9eff",
      "exported": false
    }
  ]
}
```

- `overview_rms` — 2000 points, normalised 0–1 (same as current `build_metadata`)
- `exported` — flips to `true` after a successful `/export` call; persisted to `segments.json`
- Per-segment `waveform` arrays are dropped — the UI slices `overview_rms` at runtime

---

## UI

### Overview strip (top, sticky)
- Full-recording waveform canvas (read-only)
- Segment bands drawn over it, updating live as the user drags
- Click + drag to define a new segment range
- Bands reflect `exported` state (slightly different fill when exported)

### Song cards
Each card has the same two-column layout as the current UI, with these changes:

**Card header:**
- Numbered badge, editable name input (unchanged)
- **Export WAV** button (green) — calls `POST /export/song_NN.wav`, then marks card as exported
- After export: button shows **✓ Exported** (muted style, still clickable to re-export)
- Delete (✕) button — removes from state, no file deletion needed unless already exported

**Left column — position in recording:**
- Context strip canvas (full recording waveform, this segment highlighted)
- Drag handles (left/right bars) on segment box — resize start/end
- Drag the middle of the segment box — moves entire segment
- Start / End time fields with ▲▼ stepper buttons (1 second per click, hold to repeat)
- Typed `m:ss` input — commit on blur/Enter
- Arrow key nudge: ← → moves whole segment 1s when card is focused; Shift multiplies by 10
- "Autosaved" indicator — every change PUTs to `/state`

**Right column — playback:**
- Mini waveform (sliced from `overview_rms`, normalised per-segment)
- Plays from `/audio` via HTTP range requests — no pre-exported WAV needed
- Seek by clicking mini waveform or seek bar

### Add segment
- **＋ Add segment** button — places a new segment in the largest available gap
- Click + drag on the overview strip — creates a segment spanning the dragged range

### Delete flow
- Deletes with no exported file: removes card and PUTs updated state. No confirmation needed.
- Deletes with an exported file: shows a small inline warning — "This won't delete the exported WAV file." — before removing.

---

## Workflow

```
python3 split_songs.py raw/recording.WAV
  → detects songs
  → writes segments.json to output_dir/
  → starts review_server.py
  → opens http://localhost:5123 in browser

User adjusts segments in browser
  → every change: PUT /state → segments.json updated

User clicks "Export WAV" on a card
  → POST /export/song_01.wav {start_min, end_min}
  → server writes output_dir/song_01.wav
  → card shows "✓ Exported"

User runs again on same recording
  → segments.json already exists → server loads existing state (session recovery)
  → detection is NOT re-run; existing segments.json is used as-is
  → pass --reset flag to force fresh detection and overwrite segments.json
```

---

## Out of Scope

- Re-running detection from the UI
- Bulk export all
- Waveform zoom / fine-grained waveform within a segment (overview slice is sufficient)
- Authentication or multi-user support
