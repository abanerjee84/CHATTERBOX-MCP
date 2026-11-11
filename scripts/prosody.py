"""Prosody probe: F0, loudness, speaking rate. Needs the .venv (librosa).

Usage: python scripts/prosody.py <wav> [<wav> ...]
"""

import sys
import warnings
from pathlib import Path

import librosa
import numpy as np

warnings.filterwarnings("ignore")


def probe(path: str) -> dict:
    y, sr = librosa.load(path, sr=22050, mono=True)
    f0, vflag, vprob = librosa.pyin(
        y, fmin=70, fmax=330, frame_length=1024, sr=sr, fill_na=np.nan
    )
    voiced = f0[~np.isnan(f0)]
    rms = float(np.sqrt((y**2).mean()))
    env = np.abs(librosa.stft(y)).mean(axis=0)
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    return {
        "file": Path(path).name,
        "seconds": round(len(y) / sr, 2),
        "f0_mean_hz": round(float(np.median(voiced)), 1) if voiced.size else None,
        "f0_spread_hz": round(
            float(np.percentile(voiced, 85) - np.percentile(voiced, 15)), 1
        )
        if voiced.size
        else None,
        "voiced_ratio": round(float(vflag.sum()) / max(1, len(vflag)), 2),
        "rms": round(rms, 3),
        "energy_spread": round(float(env.std() / (env.mean() + 1e-9)), 3),
        "onset_rate_per_s": round(len(onsets) / max(0.1, len(y) / sr), 2),
    }


if __name__ == "__main__":
    print(
        f"{'file':34} {'sec':>5} {'f0':>6} {'spread':>6} {'rms':>6} {'dyn':>6} {'onset':>6}"
    )
    for p in sys.argv[1:]:
        r = probe(p)
        pitch = "n/a" if r["f0_mean_hz"] is None else f"{r['f0_mean_hz']:.1f}"
        spread = "n/a" if r["f0_spread_hz"] is None else f"{r['f0_spread_hz']:.1f}"
        print(
            f"{r['file'][:34]:34} {r['seconds']:5.2f} {pitch:>6} "
            f"{spread:>6} {r['rms']:6.3f} {r['energy_spread']:6.3f} "
            f"{r['onset_rate_per_s']:6.2f}"
        )
