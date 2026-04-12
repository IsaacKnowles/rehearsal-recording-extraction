# Rehearsal Splitter — Review UI Design

## Goal

After `split_songs.py` runs detection and exports WAV files, it also generates a single `index.html` in the output directory. Opening that file in Chrome gives the user a scrollable review interface where they can verify every detected song, rename tracks, listen and seek, and delete unwanted files — all without a backend server.

---

## Architecture

**No backend.** The Python script produces two kinds of output in the same directory:

- `song_01.wav`, `song_02.wav`, … — the exported audio files
- `index.html` — the self-contained review UI

The HTML file has all waveform data baked in as a `<script>` JSON block (computed by Python during analysis — no Web Audio API decoding needed at runtime). Audio playback uses native `<audio>` elements with `src` paths relative to the HTML file. Rename state persists to `localStorage` keyed by filename. Delete triggers a browser `confirm()` dialog, removes the `<audio>` element, and calls `fetch('DELETE', filename)` — which won't actually work from `file://`, so instead a shell command is shown in a small overlay for the user to run.

> **file:// constraint:** Chrome allows `<audio src="song_01.wav">` playback from `file://` without issue. `fetch()` to local paths is blocked by CORS. So delete works as: confirm dialog → card animates out → a sticky "pending deletes" banner appears listing files to remove, with a copyable `rm` command.

---

## UI Structure

### Header (sticky)
- Recording filename, duration, sample rate/channels
- Green badge: "N songs detected"

### Full-recording overview strip
- Canvas drawing the downsampled RMS waveform of the entire recording
- Each detected song region rendered as a colored band with a highlight box
- Hovering a song card below brightens that song's region in the overview
- Time labels along the bottom (0:00 … total duration)

### Song cards (scrollable list, one per detected song)
Each card has:

**Header row**
- Numbered badge (song's accent color)
- Editable name input (default: `Song 01`, `Song 02`, …) — persisted to `localStorage`
- Start time and duration (read-only)
- Delete button (✕)

**Body — two equal columns**

| Left: Position in recording | Right: Playback |
|---|---|
| Full-recording context strip (Canvas) showing the entire waveform with this song highlighted and boxed in its accent color | Song waveform (Canvas, zoomed to just this segment) with a playhead overlay |
| Timestamp label (`MM:SS – MM:SS`) | Transport: play/pause button · seek bar (clickable) · elapsed/total time |

- Clicking anywhere on either waveform seeks to that position
- Only one song plays at a time; starting a new one stops the current

### Delete flow
1. User clicks ✕ → browser `confirm("Delete 'Song 03'?\nThis will remove the file from disk.")`
2. On confirm: card animates out (fade + slide right)
3. A sticky banner at the bottom of the page accumulates filenames: `"3 files pending delete"` with a copyable shell command: `rm song_03.wav song_07.wav …`
4. Banner has a "Copy command" button and dismisses when copied

---

## Data Flow (Python → HTML)

Python embeds this JSON block in the generated HTML:

```json
{
  "filename": "260407_182931_TrLR.WAV",
  "duration_min": 93.4,
  "sample_rate": 48000,
  "channels": 2,
  "overview_rms": [0.012, 0.45, 0.61, ...],   // ~2000 points, normalised 0–1
  "songs": [
    {
      "file": "song_01.wav",
      "name": "Song 01",
      "start_min": 1.2,
      "dur_min": 4.1,
      "color": "#4a9eff",
      "waveform": [0.31, 0.55, ...]             // ~600 points, normalised 0–1
    },
    ...
  ]
}
```

`overview_rms`: downsampled to 2000 points from the per-second RMS array (one point per ~2.8 seconds at 93 min). Normalised to 0–1 relative to the recording's peak RMS.

`waveform` per song: downsampled to 600 points from the per-second RMS within the song's window. Normalised independently per song so quiet songs still show waveform detail.

Colors are assigned from a fixed 11-color palette cycling if there are more songs.

---

## File Changes

- **Modify:** `split_songs.py` — add `generate_html(output_dir, metadata)` function called at the end of `main()`
- **No new files** beyond the generated `index.html` at runtime

---

## Out of Scope

- Adjusting song boundaries (trim handles) — future feature
- Re-running detection from the UI
- Any server process
