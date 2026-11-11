import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from chatterbox_mcp.loading import load_streamed_weights


def test_streamed_loader_preserves_weights_and_integer_buffers(tmp_path, monkeypatch):
    module = torch.nn.Linear(4, 3)
    module.register_buffer("counter", torch.tensor(7, dtype=torch.int64))
    source = {k: torch.full_like(v, 2) for k, v in module.state_dict().items()}
    path = tmp_path / "weights.safetensors"
    save_file(source, str(path))

    def forbid_mmap(*a, **kw):
        raise AssertionError("Whole-checkpoint memory mapping used")

    monkeypatch.setattr("safetensors.torch.load_file", forbid_mmap)
    load_streamed_weights(module, path)
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, source[key])


def test_missing_checkpoint_weights_are_not_silently_random(tmp_path):
    module = torch.nn.Linear(4, 3)
    path = tmp_path / "weights.safetensors"
    save_file({"weight": module.weight}, str(path))
    with pytest.raises(ValueError, match="missing=.*bias"):
        load_streamed_weights(module, path)


def test_duplicate_header_keys_are_rejected(tmp_path):
    path = tmp_path / "weights.safetensors"
    header = b'{"weight":{},"weight":{}}'
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError, match="Duplicate"):
        load_streamed_weights(torch.nn.Linear(4, 3), path)


@pytest.mark.parametrize("corruption", ["truncated", "trailing", "shape"])
def test_streamed_checkpoint_validation(tmp_path, corruption):
    module = torch.nn.Linear(4, 3)
    path = tmp_path / "weights.safetensors"
    save_file(module.state_dict(), str(path))
    raw = path.read_bytes()
    if corruption == "truncated":
        raw = raw[:-1]
    if corruption == "trailing":
        raw += b"extra"
    if corruption == "shape":
        n = struct.unpack("<Q", raw[:8])[0]
        h = json.loads(raw[8 : 8 + n])
        h["weight"]["shape"] = [2, 6]
        encoded = json.dumps(h).encode()
        raw = struct.pack("<Q", len(encoded)) + encoded + raw[8 + n :]
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_streamed_weights(module, path)
