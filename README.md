# Rehearsal Recording Extraction

Splits a rehearsal WAV recording into individually named MP3 files using a browser-based editor.

## How it works

1. Run the script on your WAV file — it reads the audio, computes a waveform overview, and opens a browser UI
2. In the browser, create segments by clicking and dragging on the overview strip, or with the **+ Add segment** button
3. Name each segment, adjust start/end times with drag handles or the time inputs, and preview playback
4. Click **Export MP3** on each card to write a named 320kbps MP3 to the output folder

## Usage

```bash
python3 split_songs.py raw/recording.WAV
```

Opens `http://localhost:5123` automatically. All session state is saved to `recording_songs/segments.json` as you work — close the browser and re-run the same command to resume.

```bash
python3 split_songs.py raw/recording.WAV --reset   # discard session, re-analyse
python3 split_songs.py raw/recording.WAV -o out/   # custom output folder
python3 split_songs.py raw/recording.WAV --port 8080
```

## Output

Exported files are written to `<stem>_songs/` next to the WAV (or the folder specified with `-o`):

```
recording_songs/
  segments.json       ← live session state
  Song_Name.mp3
  Another_Song.mp3
  ...
```

## Requirements

```bash
pip install soundfile numpy flask lameenc
```

## Running tests

```bash
pytest
```
