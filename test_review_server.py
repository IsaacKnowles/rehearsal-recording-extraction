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
