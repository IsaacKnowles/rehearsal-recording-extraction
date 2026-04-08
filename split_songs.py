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

import numpy as np
import soundfile as sf


COLORS: list[str] = [
    "#4a9eff", "#5abb6a", "#e07840", "#b06adb", "#e0c040",
    "#40b8c8", "#e05070", "#70c870", "#c06030", "#8080e0", "#40c8a0",
]

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Rehearsal Review</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0f0f0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px}
canvas{display:block}
/* ── header ── */
.app-header{padding:14px 20px;background:#1a1a1a;border-bottom:1px solid #2a2a2a;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:10}
.app-header h1{font-size:15px;font-weight:600;color:#fff}
.app-header .meta{color:#555;font-size:12px}
.badge{margin-left:auto;background:#1e3a1e;color:#6cbb6c;border:1px solid #2e5a2e;border-radius:4px;padding:2px 9px;font-size:11px;font-weight:600}
/* ── overview ── */
.overview-section{padding:14px 20px 12px;background:#121212;border-bottom:1px solid #1e1e1e}
.section-label{font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:#444;margin-bottom:6px}
.overview-wrap{position:relative;height:68px;border-radius:4px;overflow:hidden;background:#080808}
.overview-times{display:flex;justify-content:space-between;margin-top:4px;color:#383838;font-size:10px;font-variant-numeric:tabular-nums}
/* ── songs ── */
.songs-section{padding:14px 20px 80px}
.songs-header{margin-bottom:12px}
.songs-header h2{font-size:10px;font-weight:600;color:#444;letter-spacing:.08em;text-transform:uppercase}
/* ── card ── */
.song-card{background:#161616;border:1px solid #242424;border-radius:6px;margin-bottom:10px;overflow:hidden;transition:border-color .15s}
.song-card:hover{border-color:#333}
.card-header{display:flex;align-items:center;padding:9px 12px;gap:10px;border-bottom:1px solid #1e1e1e}
.song-num{width:20px;height:20px;border-radius:3px;font-size:10px;font-weight:700;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.song-name{background:transparent;border:none;color:#ccc;font-size:13px;font-weight:500;flex:1;outline:none;min-width:0;font-family:inherit}
.song-name:focus{background:#1e1e1e;border-radius:3px;padding:2px 5px;margin:-2px -5px}
.song-meta{color:#444;font-size:11px;white-space:nowrap}
.song-meta span{margin-left:8px}
.btn-del{width:24px;height:24px;background:none;border:1px solid #2a2a2a;border-radius:4px;color:#555;cursor:pointer;display:flex;align-items:center;justify-content:center;font-size:12px;transition:all .12s;flex-shrink:0}
.btn-del:hover{border-color:#7a3030;color:#e06060;background:#1e1010}
.card-body{display:grid;grid-template-columns:1fr 1fr}
/* ── context col ── */
.context-col{border-right:1px solid #1e1e1e;padding:10px 12px;display:flex;flex-direction:column;gap:6px}
.ctx-label{font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:#383838}
.ctx-wrap{flex:1;min-height:44px;background:#080808;border-radius:3px;overflow:hidden;cursor:pointer}
.ctx-time{font-size:9px;font-variant-numeric:tabular-nums;opacity:.65}
/* ── playback col ── */
.playback-col{padding:10px 12px;display:flex;flex-direction:column;gap:8px;justify-content:center}
.wave-wrap{position:relative;height:44px;background:#0a0a0a;border-radius:3px;overflow:hidden;cursor:pointer}
.playhead{position:absolute;top:0;bottom:0;width:1.5px;pointer-events:none}
.transport{display:flex;align-items:center;gap:8px}
.btn-play{width:26px;height:26px;border-radius:50%;background:#1e3050;border:none;color:#4a9eff;font-size:10px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;padding-left:1px;transition:background .12s}
.btn-play:hover{background:#2a4060}
.btn-play.playing{background:#1e3020;color:#5abb6a}
.seek-wrap{flex:1;height:3px;background:#222;border-radius:2px;cursor:pointer}
.seek-fill{height:100%;border-radius:2px;pointer-events:none}
.time-disp{color:#444;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap;min-width:72px;text-align:right}
/* ── delete banner ── */
#deleteBanner{display:none;position:fixed;bottom:0;left:0;right:0;background:#1e1010;border-top:1px solid #5a2020;padding:10px 20px;z-index:20;display:none;align-items:center;gap:12px}
#deleteBanner.visible{display:flex}
#deleteBanner .banner-msg{flex:1;font-size:12px;color:#e06060}
#deleteBanner code{background:#2a1010;border:1px solid #5a2020;border-radius:3px;padding:3px 8px;font-size:11px;color:#ff9a9a;font-family:monospace;max-width:60%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.btn-copy{background:#3a1a1a;border:1px solid #6a2a2a;color:#e06060;border-radius:4px;padding:4px 12px;cursor:pointer;font-size:11px;white-space:nowrap}
.btn-copy:hover{background:#4a2020}
</style>
</head>
<body>

<div class="app-header">
  <h1 id="hdrFilename"></h1>
  <span class="meta" id="hdrMeta"></span>
  <span class="badge" id="hdrBadge"></span>
</div>

<div class="overview-section">
  <div class="section-label">Full recording</div>
  <div class="overview-wrap" id="overviewWrap">
    <canvas id="overviewCanvas"></canvas>
  </div>
  <div class="overview-times" id="overviewTimes"></div>
</div>

<div class="songs-section">
  <div class="songs-header"><h2>Detected Songs</h2></div>
  <div id="songsList"></div>
</div>

<div id="deleteBanner">
  <span class="banner-msg" id="bannerMsg"></span>
  <code id="bannerCmd"></code>
  <button class="btn-copy" id="btnCopy">Copy rm command</button>
</div>

<script>
window.REHEARSAL_DATA = __REHEARSAL_DATA__;

(function () {
  const D = window.REHEARSAL_DATA;
  const songs = D.songs;
  let pendingDeletes = [];
  let playingIdx = null;

  // ── header ──────────────────────────────────────────────────────
  document.getElementById('hdrFilename').textContent = D.filename;
  document.getElementById('hdrMeta').textContent =
    `${D.duration_min.toFixed(1)} min \u00b7 ${D.sample_rate / 1000} kHz \u00b7 ${D.channels}ch`;
  document.getElementById('hdrBadge').textContent = `${songs.length} songs detected`;

  // ── overview time labels ─────────────────────────────────────────
  (function buildTimeLabels() {
    const container = document.getElementById('overviewTimes');
    const steps = 6;
    for (let i = 0; i <= steps; i++) {
      const min = (i / steps) * D.duration_min;
      const span = document.createElement('span');
      span.textContent = fmtMin(i === steps ? D.duration_min : min);
      container.appendChild(span);
    }
  })();

  // ── canvas helpers ───────────────────────────────────────────────
  function initCanvas(canvas, wrap) {
    const W = wrap.clientWidth, H = wrap.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + 'px';
    canvas.style.height = H + 'px';
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    return { ctx, W, H };
  }

  function drawWaveform(ctx, rms, W, H, color) {
    const mid = H / 2;
    for (let x = 0; x < W; x++) {
      const ri = Math.floor((x / W) * rms.length);
      const v = rms[ri] * mid * 0.88;
      ctx.fillStyle = color;
      ctx.fillRect(x, mid - v, 1, v * 2);
    }
  }

  // ── overview ─────────────────────────────────────────────────────
  function drawOverview(highlightIdx) {
    const wrap = document.getElementById('overviewWrap');
    const canvas = document.getElementById('overviewCanvas');
    const { ctx, W, H } = initCanvas(canvas, wrap);
    const mid = H / 2;

    songs.forEach((s, i) => {
      const x0 = (s.start_min / D.duration_min) * W;
      const x1 = ((s.start_min + s.dur_min) / D.duration_min) * W;
      const hl = i === highlightIdx;
      ctx.fillStyle = hl ? s.color + '28' : 'rgba(255,255,255,0.025)';
      ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.fillStyle = hl ? s.color : 'rgba(255,255,255,0.12)';
      ctx.fillRect(x0, 0, 1.5, H);
      ctx.fillRect(x0, 0, x1 - x0, 1.5);
      ctx.fillRect(x0, H - 1.5, x1 - x0, 1.5);
    });

    for (let x = 0; x < W; x++) {
      const ri = Math.floor((x / W) * D.overview_rms.length);
      const t = (x / W) * D.duration_min;
      const v = D.overview_rms[ri] * mid * 0.88;
      const si = songs.findIndex(s => t >= s.start_min && t < s.start_min + s.dur_min);
      if (si >= 0) {
        ctx.fillStyle = si === highlightIdx ? songs[si].color + 'dd' : songs[si].color + '55';
      } else {
        ctx.fillStyle = '#1e1e1e';
      }
      ctx.fillRect(x, mid - v, 1, v * 2);
    }
  }

  // ── context strip (per card) ──────────────────────────────────────
  function drawContext(canvas, wrap, songIdx) {
    const { ctx, W, H } = initCanvas(canvas, wrap);
    const mid = H / 2;
    const song = songs[songIdx];

    songs.forEach((s, i) => {
      const x0 = (s.start_min / D.duration_min) * W;
      const x1 = ((s.start_min + s.dur_min) / D.duration_min) * W;
      ctx.fillStyle = i === songIdx ? song.color + '28' : 'rgba(255,255,255,0.03)';
      ctx.fillRect(x0, 0, x1 - x0, H);
    });

    for (let x = 0; x < W; x++) {
      const ri = Math.floor((x / W) * D.overview_rms.length);
      const t = (x / W) * D.duration_min;
      const v = D.overview_rms[ri] * mid * 0.82;
      const si = songs.findIndex(s => t >= s.start_min && t < s.start_min + s.dur_min);
      ctx.fillStyle = si === songIdx ? song.color + 'cc' : si >= 0 ? '#2a2a2a' : '#181818';
      ctx.fillRect(x, mid - v, 1, v * 2);
    }

    const hx0 = (song.start_min / D.duration_min) * W;
    const hx1 = ((song.start_min + song.dur_min) / D.duration_min) * W;
    ctx.strokeStyle = song.color;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(hx0 + 0.75, 0.75, hx1 - hx0 - 1.5, H - 1.5);
  }

  // ── mini waveform ─────────────────────────────────────────────────
  function drawMiniWave(canvas, wrap, songIdx) {
    const { ctx, W, H } = initCanvas(canvas, wrap);
    drawWaveform(ctx, songs[songIdx].waveform, W, H, songs[songIdx].color + '99');
  }

  // ── time helpers ──────────────────────────────────────────────────
  function fmtMin(min) {
    const m = Math.floor(min);
    const s = Math.round((min - m) * 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  }
  function fmtSec(sec) { return fmtMin(sec / 60); }

  // ── localStorage names ────────────────────────────────────────────
  const STORE_KEY = 'rehearsal_names_' + D.filename;
  function loadNames() {
    try { return JSON.parse(localStorage.getItem(STORE_KEY) || '{}'); } catch { return {}; }
  }
  function saveName(file, name) {
    const m = loadNames(); m[file] = name; localStorage.setItem(STORE_KEY, JSON.stringify(m));
  }
  const savedNames = loadNames();

  // ── build cards ───────────────────────────────────────────────────
  function buildCards() {
    const list = document.getElementById('songsList');
    songs.forEach((song, i) => {
      const displayName = savedNames[song.file] || song.name;
      const card = document.createElement('div');
      card.className = 'song-card';
      card.id = `card${i}`;
      card.innerHTML = `
        <audio id="audio${i}" src="${song.file}" preload="none"></audio>
        <div class="card-header">
          <div class="song-num" style="color:${song.color}99;background:${song.color}18">${i + 1}</div>
          <input class="song-name" id="name${i}" value="${escHtml(displayName)}" spellcheck="false">
          <div class="song-meta">
            <span>@ ${fmtMin(song.start_min)}</span>
            <span>${fmtMin(song.dur_min)}</span>
          </div>
          <button class="btn-del" title="Delete song">&#x2715;</button>
        </div>
        <div class="card-body">
          <div class="context-col">
            <div class="ctx-label">Position in recording</div>
            <div class="ctx-wrap" id="ctxWrap${i}">
              <canvas id="ctxCanvas${i}"></canvas>
            </div>
            <div class="ctx-time" style="color:${song.color}">${fmtMin(song.start_min)} &ndash; ${fmtMin(song.start_min + song.dur_min)}</div>
          </div>
          <div class="playback-col">
            <div class="wave-wrap" id="waveWrap${i}">
              <canvas id="waveCanvas${i}"></canvas>
              <div class="playhead" id="playhead${i}" style="left:0%;background:${song.color}"></div>
            </div>
            <div class="transport">
              <button class="btn-play" id="playBtn${i}">&#x25B6;</button>
              <div class="seek-wrap" id="seekWrap${i}">
                <div class="seek-fill" id="seekFill${i}" style="width:0%;background:${song.color}"></div>
              </div>
              <div class="time-disp" id="timeDisp${i}">0:00 / ${fmtMin(song.dur_min)}</div>
            </div>
          </div>
        </div>
      `;
      list.appendChild(card);
    });
  }

  function escHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;');
  }

  // ── audio / playback ──────────────────────────────────────────────
  function updatePlayhead(i) {
    const audio = document.getElementById(`audio${i}`);
    const dur = audio.duration || (songs[i].dur_min * 60);
    const pct = dur > 0 ? (audio.currentTime / dur) * 100 : 0;
    document.getElementById(`seekFill${i}`).style.width = pct + '%';
    document.getElementById(`playhead${i}`).style.left = pct + '%';
    document.getElementById(`timeDisp${i}`).textContent =
      `${fmtSec(audio.currentTime)} / ${fmtMin(songs[i].dur_min)}`;
  }

  function stopCurrent() {
    if (playingIdx !== null) {
      const audio = document.getElementById(`audio${playingIdx}`);
      audio.pause();
      const btn = document.getElementById(`playBtn${playingIdx}`);
      btn.innerHTML = '&#x25B6;';
      btn.classList.remove('playing');
      playingIdx = null;
    }
  }

  function seekAudio(i, pct) {
    const audio = document.getElementById(`audio${i}`);
    const dur = audio.duration || (songs[i].dur_min * 60);
    audio.currentTime = (pct / 100) * dur;
    updatePlayhead(i);
  }

  function setupCard(i) {
    const audio = document.getElementById(`audio${i}`);
    const btn = document.getElementById(`playBtn${i}`);

    // timeupdate → move playhead
    audio.addEventListener('timeupdate', () => updatePlayhead(i));
    audio.addEventListener('ended', () => {
      btn.innerHTML = '&#x25B6;';
      btn.classList.remove('playing');
      playingIdx = null;
    });

    // play/pause
    btn.addEventListener('click', () => {
      if (playingIdx === i) { stopCurrent(); return; }
      stopCurrent();
      playingIdx = i;
      btn.innerHTML = '&#x23F8;';
      btn.classList.add('playing');
      audio.play();
    });

    // seek via seek bar
    document.getElementById(`seekWrap${i}`).addEventListener('click', e => {
      seekAudio(i, (e.offsetX / e.currentTarget.clientWidth) * 100);
    });

    // seek via mini waveform
    document.getElementById(`waveWrap${i}`).addEventListener('click', e => {
      seekAudio(i, (e.offsetX / e.currentTarget.clientWidth) * 100);
    });

    // seek via context strip (maps click position within full recording → within this song)
    document.getElementById(`ctxWrap${i}`).addEventListener('click', e => {
      const clickMin = (e.offsetX / e.currentTarget.clientWidth) * D.duration_min;
      const song = songs[i];
      const clampedMin = Math.max(song.start_min, Math.min(song.start_min + song.dur_min, clickMin));
      const pct = ((clampedMin - song.start_min) / song.dur_min) * 100;
      seekAudio(i, pct);
    });

    // rename → persist
    document.getElementById(`name${i}`).addEventListener('change', e => {
      saveName(songs[i].file, e.target.value);
    });

    // delete
    document.querySelector(`#card${i} .btn-del`).addEventListener('click', () => {
      const name = document.getElementById(`name${i}`).value;
      if (!confirm(`Delete "${name}"?\n\nThis will remove the file from disk.`)) return;
      if (playingIdx === i) stopCurrent();
      const card = document.getElementById(`card${i}`);
      card.style.transition = 'opacity .2s, transform .2s';
      card.style.opacity = '0';
      card.style.transform = 'translateX(8px)';
      setTimeout(() => card.remove(), 200);
      pendingDeletes.push(songs[i].file);
      updateDeleteBanner();
    });

    // hover → highlight in overview
    document.getElementById(`card${i}`).addEventListener('mouseenter', () => drawOverview(i));
    document.getElementById(`card${i}`).addEventListener('mouseleave', () => drawOverview(-1));
  }

  // ── delete banner ─────────────────────────────────────────────────
  function updateDeleteBanner() {
    const banner = document.getElementById('deleteBanner');
    if (pendingDeletes.length === 0) { banner.classList.remove('visible'); return; }
    banner.classList.add('visible');
    document.getElementById('bannerMsg').textContent =
      `${pendingDeletes.length} file${pendingDeletes.length > 1 ? 's' : ''} pending delete:`;
    const cmd = 'rm ' + pendingDeletes.join(' ');
    document.getElementById('bannerCmd').textContent = cmd;
  }

  document.getElementById('btnCopy').addEventListener('click', () => {
    const cmd = 'rm ' + pendingDeletes.join(' ');
    navigator.clipboard.writeText(cmd).then(() => {
      const btn = document.getElementById('btnCopy');
      btn.textContent = 'Copied!';
      setTimeout(() => { btn.textContent = 'Copy rm command'; }, 2000);
    });
  });

  // ── init ──────────────────────────────────────────────────────────
  buildCards();
  drawOverview(-1);
  songs.forEach((_, i) => {
    drawContext(
      document.getElementById(`ctxCanvas${i}`),
      document.getElementById(`ctxWrap${i}`),
      i
    );
    drawMiniWave(
      document.getElementById(`waveCanvas${i}`),
      document.getElementById(`waveWrap${i}`),
      i
    );
    setupCard(i);
  });

  window.addEventListener('resize', () => {
    drawOverview(-1);
    songs.forEach((_, i) => {
      drawContext(document.getElementById(`ctxCanvas${i}`), document.getElementById(`ctxWrap${i}`), i);
      drawMiniWave(document.getElementById(`waveCanvas${i}`), document.getElementById(`waveWrap${i}`), i);
    });
  });
})();
</script>
</body>
</html>"""


def generate_html(output_dir: str, metadata: dict) -> str:
    """Write index.html to output_dir with metadata embedded as JSON.

    Returns the absolute path to the written file.
    """
    import json as _json
    data_json = _json.dumps(metadata, separators=(",", ":"))
    html = _HTML_TEMPLATE.replace("__REHEARSAL_DATA__", data_json, 1)
    out_path = os.path.join(output_dir, "index.html")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


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
    if normalize:
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
    """Build the JSON payload embedded in the HTML review UI.

    Args:
        input_path: path to the original WAV file (basename used in output)
        rate: sample rate in Hz
        channels: number of audio channels
        rms: full per-window RMS array from compute_rms()
        window_sec: window size in seconds used during analysis
        songs: list of (start_window, end_window) tuples from find_songs()

    Returns:
        Dict matching the schema consumed by the HTML template.
    """
    total_sec = len(rms) * window_sec
    duration_min = total_sec / 60.0

    song_list = []
    for i, (w_start, w_end) in enumerate(songs):
        song_rms = rms[w_start:w_end]
        song_list.append({
            "file": f"song_{i + 1:02d}.wav",
            "name": f"Song {i + 1:02d}",
            "start_min": round(w_start * window_sec / 60.0, 4),
            "dur_min": round((w_end - w_start) * window_sec / 60.0, 4),
            "color": COLORS[i % len(COLORS)],
            "waveform": downsample_rms(song_rms, 600, normalize=True),
        })

    return {
        "filename": os.path.basename(input_path),
        "duration_min": round(duration_min, 2),
        "sample_rate": rate,
        "channels": channels,
        "overview_rms": downsample_rms(rms, 2000, normalize=True),
        "songs": song_list,
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

    print(f"\nDone. {len(songs)} files written to {output_dir}/")


if __name__ == "__main__":
    main(parse_args())
