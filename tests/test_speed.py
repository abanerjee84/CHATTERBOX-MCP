import json
import sys

import numpy as np
import pytest
from pydantic import ValidationError

from chatterbox_mcp.audio import change_tempo
from chatterbox_mcp.cli import main
from chatterbox_mcp.client import Client
from chatterbox_mcp.jobs import JobStore
from chatterbox_mcp.schema import SpeechRequest


@pytest.mark.parametrize("speed", [0.5, 0.75, 0.95, 1.25, 1.5])
def test_speed_changes_duration_without_pitch_shift(speed):
    rate = 24000
    samples = np.sin(2 * np.pi * 440 * np.arange(2 * rate) / rate).astype("float32")
    result = change_tempo(samples, speed)
    assert len(result) == round(len(samples) / speed)
    assert np.isfinite(result).all()
    frequency = np.fft.rfftfreq(len(result), 1 / rate)[
        np.argmax(np.abs(np.fft.rfft(result)))
    ]
    assert abs(frequency - 440) < 1


def test_normal_speed_preserves_exact_samples():
    samples = np.random.default_rng(42).normal(size=48000).astype("float32")
    assert change_tempo(samples, 1.0) is samples


def test_small_speed_change_does_not_smear_transients():
    # Short consonant-like bursts separated by silence expose phase-vocoder smear.
    samples = np.zeros(48000, dtype="float32")
    for index in range(8):
        start = index * 6000 + 1000
        samples[start : start + 120] = (
            np.random.default_rng(index).normal(size=120) * 0.1
        )
    result = change_tempo(samples, 0.95)
    active_input = np.count_nonzero(np.abs(samples) > 0.01 * np.max(np.abs(samples)))
    active_output = np.count_nonzero(np.abs(result) > 0.01 * np.max(np.abs(result)))
    assert active_output < 3 * active_input


def test_missing_ffmpeg_fails_with_install_hint(monkeypatch):
    monkeypatch.setattr("chatterbox_mcp.audio.shutil.which", lambda name: None)
    samples = np.ones(24000, dtype="float32")
    assert change_tempo(samples, 1.0) is samples
    with pytest.raises(RuntimeError, match="requires FFmpeg on PATH"):
        change_tempo(samples, 0.95)


def test_ffmpeg_failure_is_reported(monkeypatch):
    from subprocess import CompletedProcess

    monkeypatch.setattr("chatterbox_mcp.audio.require_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(
        "chatterbox_mcp.audio.subprocess.run",
        lambda *a, **k: CompletedProcess(a[0], 1, b"", b"filter unavailable"),
    )
    with pytest.raises(RuntimeError, match="filter unavailable"):
        change_tempo(np.ones(24000, dtype="float32"), 0.95)


@pytest.mark.parametrize("value", [0.49, 1.51, float("nan"), float("inf"), -1])
def test_speed_bounds(value):
    with pytest.raises(ValidationError):
        SpeechRequest(text="Hello", change_speed=value)


@pytest.mark.parametrize("flag", ["--change_speed", "--change-speed"])
def test_cli_passes_speed_to_service(flag, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        Client,
        "submit",
        lambda c, r: {"job_id": "a" * 32, "status": "queued", "request": r},
    )
    monkeypatch.setattr(
        sys, "argv", ["chatter-tts", "generate", "--text", "Hello", flag, "0.75"]
    )
    assert main() == 0
    assert json.loads(capsys.readouterr().out)["request"]["change_speed"] == 0.75


def test_speed_in_batch_json_and_command_default(monkeypatch, tmp_path, capsys):
    source = tmp_path / "batch.json"
    source.write_text('[{"text":"First"},{"text":"Second","change_speed":1.5}]')
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        Client,
        "submit",
        lambda c, r: {"job_id": "a" * 32, "status": "queued", "request": r},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["chatter-tts", "generate", "--request", str(source), "--change_speed", "0.5"],
    )
    assert main() == 0
    jobs = json.loads(capsys.readouterr().out)["jobs"]
    assert [job["request"]["change_speed"] for job in jobs] == [0.5, 1.5]


def test_old_job_retry_defaults_to_original_speed(tmp_path):
    store = JobStore(tmp_path, None)
    req = SpeechRequest(text="Hello", client_request_id="old-retry", seed=42)
    first = store.submit(req)
    legacy = first["request"].copy()
    legacy.pop("change_speed")
    with store.connect() as db:
        db.execute(
            "UPDATE jobs SET request=? WHERE id=?",
            (json.dumps(legacy), first["job_id"]),
        )
    assert store.submit(req)["job_id"] == first["job_id"]
