"""Pitch-preserving final-audio tempo adjustment using FFmpeg's WSOLA filter."""

import math
import os
import shutil
import subprocess


def require_ffmpeg():
    executable = shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError(
            "Speed adjustment requires FFmpeg on PATH. Install FFmpeg, then restart "
            "chatter-tts. Use --change_speed 1 for unchanged audio."
        )
    return executable


def change_tempo(samples, speed, sample_rate=24000):
    if not math.isfinite(speed) or not 0.5 <= speed <= 1.5:
        raise ValueError("change_speed must be between 0.5 and 1.5")
    if speed == 1.0:
        return samples
    import numpy as np

    samples = np.asarray(samples, dtype="<f4")
    if samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError("Speed adjustment requires finite, nonempty mono audio")
    target = round(len(samples) / speed)
    # Flush the overlap window with silence before trimming to the target duration.
    # This preserves the last phoneme even when atempo buffers the final frame.
    filters = f"apad=pad_len=4096,atempo={speed:.12g},atrim=end_sample={target}"
    command = [
        require_ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-af",
        filters,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "pipe:1",
    ]
    try:
        process = subprocess.run(
            command,
            input=samples.tobytes(),
            capture_output=True,
            timeout=180,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "FFmpeg speed adjustment timed out after 180 seconds"
        ) from exc
    if process.returncode:
        message = process.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"FFmpeg speed adjustment failed: {message}")
    result = np.frombuffer(process.stdout, dtype="<f4").copy()
    if len(result) != target or not np.isfinite(result).all():
        raise RuntimeError("FFmpeg returned incomplete or invalid speed-adjusted audio")
    return result
