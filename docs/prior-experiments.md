> These notes were present before this audit. Current validation is recorded in gpu-validation.md. Some setup statements here have been superseded.

# CHATTERBOX-MCP

Local **Chatterbox English TTS** (Resemble AI, MIT) behind a small CLI: `chatter-tts`.
Weights live here, one process loads them once and renders a batch. Measured on an
RTX 3060 with a hard 8 GiB allocator cap: **peak reserved 4.1 GiB**.

```powershell
E:\TOOLS\CHATTERBOX-MCP\bin\chatter-tts.cmd doctor
E:\TOOLS\CHATTERBOX-MCP\bin\chatter-tts.cmd --gpu-uuid <uuid> --vram-cap-gib 8 generate `
  --text "Alright! We did it!" --voice data\voices\baddy-neutral.wav `
  --exaggeration 1.35 --cfg-weight 0.3 --seed 403 --out data\outputs\hit.wav
E:\TOOLS\CHATTERBOX-MCP\bin\chatter-tts.cmd generate --request jobs.json   # batch, one load
```

## What you can actually control

Chatterbox has **one** expressiveness knob, not an emotion vocabulary. In
`models/t3/modules/cond_enc.py` the scalar `exaggeration` is projected by a learned
`nn.Linear(1, n_channels)` into the T3 condition: a single learned "lean into it"
direction whose magnitude you set. The *kind* of emotion has to come from the text;
the scalar sets how hard the model commits to it.

| Flag | Range | Effect (measured) |
| --- | --- | --- |
| `--exaggeration` | 0.45–1.35 used here; upstream default 0.5, "dramatic" ~0.7+ | pitch and loudness rise monotonically (see below) |
| `--cfg-weight` | 0–1, lower = slower and more deliberate | 0.3 for shouting lines, 0.5 for calm |
| `--temperature` | 0.8 default | sampling randomness |
| `--repetition-penalty` | 1.2 default | guards against repeats |
| `--voice` | reference wav, 5–15 s clean speech | sets identity; its prosody bleeds in |
| `--seed` | int | repeatable per software/hardware |
| `--vram-cap-gib` | default 8 | caps the CUDA caching allocator; overruns raise OOM instead of spilling |

## Measurements, RTX 3060, 8 GiB cap, fp32

| Run | peak reserved | device memory held | audio | wall |
| --- | --- | --- | --- | --- |
| 5 emotional lines, one load | 4 172 MiB | 4 210 MiB | 7.7–10.4 s each | 62.5 s total |
| 6 donor-transfer lines | 3 782 MiB | 3 820 MiB | 4.1–6.0 s each | 46.6 s total |
| single short line | 3 648 MiB | 3 686 MiB | 4.3 s | 6.4 s + load |

Weights are fp32 on disk (3.0 GB total: `t3_cfg` 2.13 GB, `s3gen` 1.06 GB, `ve` 5.7 MB).
`--autocast fp16` was tested and is **worse**: 4 484 MiB reserved vs 3 648 MiB, and slower
(10.7 s vs 6.4 s). Leave autocast off. Output is 24 kHz mono.

### Exaggeration as an intensity dial

One voice (`data/voices/baddy-neutral.wav`), same seeds, `f0` = median voiced pitch:

| Line | exaggeration | cfg | f0 Hz | f0 spread Hz | RMS |
| --- | --- | --- | --- | --- | --- |
| calm | 0.45 | 0.50 | 116 | 73 | 0.064 |
| sad | 0.80 | 0.40 | 131 | 42 | 0.092 |
| afraid | 1.00 | 0.35 | 157 | 134 | 0.151 |
| angry | 1.20 | 0.30 | 176 | 57 | 0.164 |
| excited | 1.35 | 0.30 | 216 | 76 | 0.204 |

RMS spans 3.2x and pitch 1.9x across the sweep, so the knob is real. Pitch *spread* is
not ordered — afraid had the widest movement (134 Hz) at a middling exaggeration, which is
the text doing the work, not the scalar.

### Donor prosody does transfer, partially

Same line, same settings (`exaggeration 0.7`, `cfg 0.4`), only the reference clip changed
(donors were the five emotional renders, so identity stayed roughly constant):

| Donor | donor f0/RMS | output f0/RMS |
| --- | --- | --- |
| neutral | 112 / 0.060 | 139 / 0.094 |
| angry | 176 / 0.164 | 174 / 0.155 |
| excited | 216 / 0.204 | 222 / 0.203 |
| afraid | 157 / 0.151 | 177 / 0.164 |
| calm | 116 / 0.064 | 120 / 0.102 |
| sad | 131 / 0.092 | 174 / 0.125 |

Four of five tracked their donor closely; the sad donor came out higher and louder than
its reference. Reference clips also carry their spoken words, which Chatterbox continues
in-context, so a donor is not a clean emotion vector.

## Setup notes

- Venv: `.venv` (CPython 3.11.14), `uv venv --python 3.11 .venv`.
- **torch 2.8.0+cu128 / torchaudio 2.8.0**, installed from `download.pytorch.org/whl/cu128`.
  Upstream pins `torch==2.6.0`; there is no Windows cu126/cu128 2.6 wheel, and cu126 does
  not cover Blackwell (sm_120). Verified working with Chatterbox 0.1.7 here.
- `gradio` and `spacy-pkuseg` were **not** installed (not needed for English TTS):
  `uv pip install --no-deps chatterbox-tts==0.1.7`, deps installed explicitly.
- Weights: only 5 files from `ResembleAI/chatterbox` are needed
  (`ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, `conds.pt`)
  — 3.0 GB, not the 13.9 GB the repo carries. Downloaded to `data/models/chatterbox`.
- `HF_HOME` points at `data/hf_cache` from `bin/chatter-tts.cmd`.

## Caveats

- Outputs carry Resemble's **Perth implicit watermark** (applied by the library by default).
- MIT on the code and weights; you still need permission to use any reference recording.
- English only for `ChatterboxTTS`; multilingual is a separate model and needs the deps skipped here.
- No emotion labels: no `--emotion angry`. If you need per-emotion control rather than one
  intensity axis, that is IndexTTS-2.5's `emo_vector`, not this.
