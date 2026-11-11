import json
import sys
from pathlib import Path

import pytest

from chatterbox_mcp.cli import main
from chatterbox_mcp.client import Client
from chatterbox_mcp.schema import SpeechRequest


def test_cli_defaults_match_service_and_mcp(monkeypatch, tmp_path, capsys):
    sent = []

    def submit(client, request):
        sent.append(request)
        return {"job_id": "a" * 32, "status": "queued", "request": request}

    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(Client, "submit", submit)
    monkeypatch.setattr(sys, "argv", ["chatter-tts", "generate", "--text", "Hello"])
    assert main() == 0
    assert sent == [SpeechRequest(text="Hello").model_dump()]
    assert json.loads(capsys.readouterr().out)["status"] == "queued"


def test_legacy_reference_path_and_off_autocast(monkeypatch, tmp_path, capsys):
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"test")
    uploaded = []

    def upload(client, path, name="Reference voice"):
        uploaded.append(path)
        return {"voice_id": "a" * 32}

    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(Client, "upload_voice", upload)
    monkeypatch.setattr(
        Client,
        "submit",
        lambda c, r: {"job_id": "b" * 32, "status": "queued", "request": r},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "chatter-tts",
            "--autocast",
            "off",
            "generate",
            "--text",
            "Hello",
            "--voice",
            str(audio),
            "--exaggeration",
            "1.35",
            "--cfg-weight",
            "0.3",
        ],
    )
    assert main() == 0
    request = json.loads(capsys.readouterr().out)["request"]
    assert request["voice_id"] == "a" * 32 and request["autocast"] == "off"
    assert request["exaggeration"] == 1.35 and request["cfg_weight"] == 0.3
    assert Path(uploaded[0]).is_absolute()


def test_batch_prevalidated_before_any_job_is_submitted(monkeypatch, tmp_path, capsys):
    source = tmp_path / "batch.json"
    source.write_text(
        json.dumps(
            [
                {"text": "Valid", "out": "first.wav"},
                {"text": "Invalid", "exaggeration": 99},
            ]
        )
    )
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))

    def no_submit(*args):
        raise AssertionError("Partial batch was submitted")

    monkeypatch.setattr(Client, "submit", no_submit)
    monkeypatch.setattr(
        sys, "argv", ["chatter-tts", "generate", "--request", str(source)]
    )
    assert main() == 1
    assert "exaggeration" in capsys.readouterr().out


def test_batch_failure_returns_nonzero(monkeypatch, tmp_path, capsys):
    source = tmp_path / "batch.json"
    source.write_text(json.dumps([{"text": "One"}, {"text": "Two"}]))
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        Client,
        "submit",
        lambda c, r: {"job_id": "a" * 32, "status": "queued", "request": r},
    )
    monkeypatch.setattr(
        Client,
        "wait",
        lambda *a: {"job_id": "a" * 32, "status": "failed", "error": "Test failure"},
    )
    monkeypatch.setattr(
        sys, "argv", ["chatter-tts", "generate", "--request", str(source), "--wait"]
    )
    assert main() == 1
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 2


def test_request_file_cannot_silently_ignore_voice_flag(monkeypatch, tmp_path, capsys):
    source = tmp_path / "request.json"
    source.write_text('{"text":"Hello"}')
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["chatter-tts", "generate", "--request", str(source), "--voice", "my-voice"],
    )
    assert main() == 1
    assert "inside the JSON" in capsys.readouterr().out


def test_partially_submitted_batch_reports_existing_job_ids(
    monkeypatch, tmp_path, capsys
):
    from unittest.mock import Mock

    source = tmp_path / "batch.json"
    source.write_text('[{"text":"One"},{"text":"Two"}]')
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        Client,
        "submit",
        Mock(
            side_effect=[
                {"job_id": "a" * 32, "status": "queued"},
                RuntimeError("Queue full"),
            ]
        ),
    )
    monkeypatch.setattr(
        sys, "argv", ["chatter-tts", "generate", "--request", str(source)]
    )
    assert main() == 1
    assert (
        json.loads(capsys.readouterr().out)["submitted_jobs"][0]["job_id"] == "a" * 32
    )


def test_other_gpu_is_rejected(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CHATTER_TTS_HOME", str(tmp_path))
    monkeypatch.setattr(
        "chatterbox_mcp.compat.gpu_inventory",
        lambda: [{"uuid": "GPU-other", "name": "NVIDIA GeForce RTX 5060 Ti"}],
    )
    monkeypatch.setattr(
        sys, "argv", ["chatter-tts", "--gpu-uuid", "GPU-other", "doctor"]
    )
    assert main() == 1
    assert "RTX 3060" in capsys.readouterr().out


@pytest.mark.parametrize(
    "options",
    [
        {"exaggeration": 1.6},
        {"cfg_weight": -1},
        {"min_p": -1},
        {"temperature": float("nan")},
        {"seed": 2**32},
    ],
)
def test_chatterbox_control_bounds(options):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SpeechRequest(text="Hello", **options)
