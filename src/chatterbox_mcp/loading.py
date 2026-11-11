"""Load the pinned checkpoint without full CPU model copies or file-sized mmap commitments."""

import json
import struct


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors header key: {key}")
        result[key] = value
    return result


def load_streamed_weights(module, path, *, allowed_missing=()):
    """Copy one validated safetensors tensor at a time into the final-device model."""
    import torch
    from safetensors.torch import load

    targets = module.state_dict()
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors header")
        header_size = struct.unpack("<Q", prefix)[0]
        if not 2 <= header_size <= 16 * 1024**2:
            raise ValueError("Invalid safetensors header size")
        header = json.loads(stream.read(header_size), object_pairs_hook=unique_keys)
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        missing = set(targets) - set(tensors) - set(allowed_missing)
        unexpected = set(tensors) - set(targets)
        if missing or unexpected:
            raise ValueError(
                f"Checkpoint keys do not match: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        base = 8 + header_size
        previous = 0
        for name, info in sorted(
            tensors.items(), key=lambda item: item[1]["data_offsets"][0]
        ):
            begin, end = info["data_offsets"]
            target = targets[name]
            if list(target.shape) != info["shape"]:
                raise ValueError(f"Checkpoint shape mismatch: {name}")
            size = end - begin
            if begin != previous or size < 0 or size > 256 * 1024**2:
                raise ValueError(f"Invalid or oversized checkpoint tensor: {name}")
            # Rust's safetensors decoder validates dtype, shape and data length for each tensor.
            small_header = json.dumps(
                {name: {**info, "data_offsets": [0, size]}}, separators=(",", ":")
            ).encode()
            small_header += b" " * ((-len(small_header)) % 8)
            stream.seek(base + begin)
            raw = stream.read(size)
            if len(raw) != size:
                raise ValueError(f"Truncated checkpoint tensor: {name}")
            encoded = struct.pack("<Q", len(small_header)) + small_header + raw
            tensor = load(encoded)[name]
            with torch.no_grad():
                target.copy_(tensor)
            del tensor, encoded, raw
            previous = end
        stream.seek(0, 2)
        if stream.tell() != base + previous:
            raise ValueError("Checkpoint has trailing or missing tensor data")
    return module


def load_chatterbox(path, device, progress):
    import torch

    from ._upstream.tts import (
        T3,
        ChatterboxTTS,
        Conditionals,
        EnTokenizer,
        S3Gen,
        VoiceEncoder,
    )

    components = []
    for name, factory, file, allowed, percent in (
        ("voice_encoder", VoiceEncoder, "ve.safetensors", (), 5),
        ("tts", T3, "t3_cfg.safetensors", (), 7),
        (
            "vocoder",
            S3Gen,
            "s3gen.safetensors",
            ("tokenizer._mel_filters", "tokenizer.window"),
            10,
        ),
    ):
        progress("loading_" + name, percent)
        # Initializing on CUDA avoids allocating and initializing multi-GB CPU parameter copies.
        with torch.device(device):
            module = factory()
        module = module.to(device).eval()
        load_streamed_weights(module, path / file, allowed_missing=allowed)
        components.append(module)
    ve, t3, s3gen = components
    tokenizer = EnTokenizer(str(path / "tokenizer.json"))
    conds = Conditionals.load(path / "conds.pt", map_location=device).to(device)
    return ChatterboxTTS(t3, s3gen, ve, tokenizer, device, conds=conds)
