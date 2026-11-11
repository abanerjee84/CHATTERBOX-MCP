"""Final-audio tempo adjustment with the already-installed Python audio runtime."""

import math


def change_tempo(samples, speed):
    if not math.isfinite(speed) or not 0.5 <= speed <= 1.5:
        raise ValueError("change_speed must be between 0.5 and 1.5")
    if speed == 1.0:
        return samples
    import librosa

    return librosa.effects.time_stretch(samples, rate=speed)
