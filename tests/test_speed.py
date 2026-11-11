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


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.25, 1.5])
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
