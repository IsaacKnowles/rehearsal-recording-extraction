#!/usr/bin/env python3
"""HTTP review server for rehearsal recordings.

Usage (direct):
    python3 review_server.py <output_dir> <source_wav> <port>

Called programmatically from split_songs.py via review_server.run().
"""

import json
import os
import sys

import lameenc
import numpy as np
import soundfile as sf
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


def _encode_mp3(samples: np.ndarray, rate: int) -> bytes:
    """Encode float32 stereo/mono samples to MP3 bytes using lameenc."""
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(320)
    encoder.set_in_sample_rate(rate)
    encoder.set_channels(samples.shape[1] if samples.ndim > 1 else 1)
    encoder.set_quality(2)  # 2 = high quality

    pcm = (samples * 32767).clip(-32768, 32767).astype(np.int16)
    if pcm.ndim > 1:
        # lameenc expects interleaved samples
        pcm = pcm.flatten()
    return encoder.encode(pcm.tobytes()) + encoder.flush()


@app.route("/export/<filename>", methods=["POST"])
def export_segment(filename: str):
    """Export one segment as an MP3 file.

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

    safe_name = os.path.splitext(os.path.basename(filename))[0] + ".mp3"
    out_path = os.path.join(OUTPUT_DIR, safe_name)
    mp3_bytes = _encode_mp3(chunk, rate)
    with open(out_path, "wb") as f:
        f.write(mp3_bytes)

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
