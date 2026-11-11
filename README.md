# CHATTERBOX-MCP

Local **Chatterbox English 500M speech generation and reference-voice cloning** through **MCP, a global CLI, and HTTP**. One hidden background worker serves every client through a persistent SQLite queue, following KANITTS2-MCP's service structure.

The supported target is **Windows x64 with an RTX 3060 12 GB**. The worker selects that card by UUID before importing CUDA and refuses to switch to another GPU. Output is **24,000 Hz mono PCM-16 WAV**, with request metadata, timing, memory measurements, and the upstream Perth watermark.

This project retains the original Chatterbox English model and its **exaggeration and CFG controls**. It uses pinned Chatterbox 0.1.7 inference sources with the existing tested PyTorch 2.8 CUDA runtime. Turbo, Nano, and multilingual models are separate upstream variants and are not exposed here.

**Validated locally:** 40 tests pass; both MCP transports generate actual speech and cloned voices. A two-segment passage produced 19.95 seconds of audio in 19.44 seconds with models resident, peaking at 3,675 MiB dedicated VRAM. A repeated default-voice request was byte-identical before and after a reference-voice job. See [validation details](docs/gpu-validation.md).

## Install

Requires Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), a CUDA 12.8-compatible NVIDIA driver, and space for Python, CUDA packages, outputs, and approximately 3.0 GiB of model weights. A CUDA Toolkit, C++ compiler, Docker, or separate web UI is not required.

In PowerShell:

```powershell
Set-Location E:\TOOLS
git clone https://github.com/abanerjee84/CHATTERBOX-MCP.git
Set-Location CHATTERBOX-MCP
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -DownloadModels -RegisterCodex
```

Setup creates a Python 3.11 environment, installs locked packages, downloads five pinned model files with SHA-256 verification, adds `chatter-tts` to your user PATH, and optionally registers `local-chatterbox` in Codex. Restart Codex to discover the six MCP tools. Open a new terminal for the PATH update.

Use `-RegisterClaudeCode` or `-RegisterGemini` to register those clients. Registration preserves unrelated settings, makes a backup, and refuses to replace a conflicting entry.

If PATH has not refreshed, use `E:\TOOLS\CHATTERBOX-MCP\bin\chatter-tts.cmd` or `.\.venv\Scripts\python.exe -m chatterbox_mcp`.

## Generate and clone

```powershell
# The packaged default speaker:
chatter-tts generate --text "Hello. This speech is generated locally." --seed 42 --out hello.wav

# Clone an existing recording:
chatter-tts generate --text "Well, that went better than expected. Let's call it a win." --reference-audio C:\audio\reference.wav --seed 403 --out cloned.wav

# Existing --voice file-path syntax also works:
chatter-tts generate --text "Alright! We did it!" --voice C:\audio\reference.wav --exaggeration 1.35 --cfg-weight 0.3 --seed 403 --out excited.wav

# Import once and reuse the returned voice_id:
chatter-tts upload-voice C:\audio\reference.wav --name "My voice"
chatter-tts voices
chatter-tts generate --text "Another sentence in the reference voice." --voice-id VOICE_ID --wait
```

`--out` waits for completion and writes a copy to that path. Without `--wait` or `--out`, submission returns a job ID immediately. Use `status JOB_ID`, `wait JOB_ID`, `cancel JOB_ID`, or `download JOB_ID --out speech.wav` afterward. `download JOB_ID --artifact metadata --out speech.json` retrieves the request/result sidecar.

References must contain **3-30 seconds** of non-silent audio, at most 32 MiB; use approximately 5-15 seconds of clear speech from one speaker for practical recordings. WAV, FLAC, OGG, and other SoundFile-supported formats work. Uploads are copied, mixed to mono, and resampled to 24 kHz. Original files remain in place. Identical normalized recordings return the same voice ID, preserving retry identity.

`--voice NAME` also resolves `data/voices/NAME.wav`. The recordings already present in this local folder, including `baddy-neutral`, remain usable. They are local references, not built-in Chatterbox presets, and are not included in Git. `--voice default` selects the packaged default speaker. `chatter-tts voices` distinguishes imported IDs, named local files, and the default speaker. This model does not provide Kani's named preset embeddings or accent tags.

## Delivery controls

| Option | Default | Accepted values / purpose |
| --- | --- | --- |
| `--exaggeration` | 0.5 | 0-1.5; learned intensity/expressiveness scale |
| `--cfg-weight` | 0.5 | 0-1; classifier-free guidance strength, also affecting delivery |
| `--temperature` | 0.8 | 0.1-2; sampling randomness |
| `--repetition-penalty` | 1.2 | 1-3 |
| `--min-p` | 0.05 | 0-1 |
| `--top-p` | 1.0 | Greater than 0, at most 1 |
| `--seed` | random | Omit or use -1 for random; resolved value is saved |
| `--autocast` | off | off, fp16, bf16; optional experimental inference context |

Exaggeration is one scalar control, not an emotion selector. Text and reference prosody also affect delivery; there is no `--emotion angry`. Higher exaggeration and CFG changes do not guarantee a particular emotion or speaking rate. Upstream suggests increasing exaggeration and lowering CFG for more dramatic delivery; see its [English model guide](https://huggingface.co/ResembleAI/chatterbox). The validated default uses FP32 without autocast.

A fixed seed improves repeatability on the same runtime and hardware. Cloning quality, pronunciation, pacing, and prosody still depend on the model and reference. References should be recordings you have permission to use.

## Batch files and compatibility

```powershell
chatter-tts generate --request examples/speech.json --wait
chatter-tts generate --request examples/batch.json
chatter-tts generate --text-file narration.txt --wait --out narration.wav
```

A request file accepts one object or the legacy JSON array of jobs. Each batch entry may include `voice` as a local path/name and `out` as a destination. The remaining fields are the speech request fields below. All entries are validated before queueing. Each entry is a separately tracked job; the worker runs one at a time. A failed or cancelled job gives the CLI a nonzero exit code.

Legacy global `--gpu-uuid`, `--vram-cap-gib`, `--model-dir`, and `--autocast` flags remain accepted before the subcommand. GPU selection accepts the UUID with or without `GPU-`, but only an available RTX 3060. Profile changes require the worker to be stopped and are saved for subsequent requests:

```powershell
chatter-tts stop
chatter-tts --vram-cap-gib 8 configure --residency resident
```

The original CLI and README were backed up locally under `data/migration-backup/` during the migration. Existing voices, weights, and output files were preserved. [Prior experiment notes](docs/prior-experiments.md) are retained separately; current validation is in [gpu-validation.md](docs/gpu-validation.md).

## MCP

```powershell
chatter-tts register codex
chatter-tts register claude-code
chatter-tts register gemini
chatter-tts mcp-config
```

For another host, adapt [mcp-config.example.json](mcp-config.example.json). The stdio adapter runs with `python -m chatterbox_mcp mcp` and starts the shared worker automatically. The Windows worker launches hidden through WMI and survives adapter disconnects. Stdio carries MCP messages; diagnostics go to stderr.

| Tool | Purpose |
| --- | --- |
| `generate_speech` | Queue text with a reference or default voice and delivery controls |
| `upload_voice` | Import an absolute local reference path and return a reusable ID |
| `list_voices` | List imported IDs, local reference names, and the default speaker |
| `get_speech_job` | Poll status, stage, errors, output paths, timing, and memory |
| `cancel_speech_job` | Cancel pending or running generation |
| `speech_service_status` | Read model readiness, selected GPU, and residency |

`generate_speech` accepts `voice_id` or `voice`, not both. `voice` resolves a local reference path/name, while `voice_id` comes from `upload_voice`. Omit both for the packaged speaker. `get_speech_job(include_audio=true)` attaches a successful WAV, up to 170 seconds and 8 MiB. For larger output, use the local path or HTTP download. The MCP client must support audio content to play an attachment.

## HTTP

The worker binds only to `127.0.0.1:8769`. Every endpoint, including `/health` and `/mcp`, requires the bearer token in local `data/config.json`. Browser requests from other origins are rejected. Keep that config and token private.

```powershell
chatter-tts start
$chatterConfig = Get-Content E:\TOOLS\CHATTERBOX-MCP\data\config.json -Raw | ConvertFrom-Json
$chatterHeaders = @{Authorization = "Bearer $($chatterConfig.token)"}
$chatterJob = Invoke-RestMethod http://127.0.0.1:8769/v1/jobs -Method Post -Headers $chatterHeaders -ContentType application/json -Body '{"text":"Hello from HTTP.","seed":42}'
Invoke-RestMethod "http://127.0.0.1:8769/v1/jobs/$($chatterJob.job_id)" -Headers $chatterHeaders
```

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Read service/inference readiness |
| POST | `/v1/jobs` | Submit a speech request; HTTP 202 |
| GET | `/v1/jobs/{id}` | Poll job |
| POST | `/v1/jobs/{id}/cancel` | Cancel job |
| GET | `/v1/jobs/{id}/audio` | Download successful WAV |
| GET | `/v1/jobs/{id}/metadata` | Download request/result JSON |
| POST | `/v1/voices?name=MyVoice` | Import raw audio bytes, not multipart or a filesystem path |
| GET | `/v1/voices` | List references and default speaker |
| POST | `/admin/shutdown` | Stop worker |
| MCP | `/mcp` | Streamable HTTP MCP using the same bearer header |

HTTP speech request:

```json
{
  "text": "Welcome. This speech was generated on this computer.",
  "voice_id": null,
  "exaggeration": 0.5,
  "cfg_weight": 0.5,
  "temperature": 0.8,
  "repetition_penalty": 1.2,
  "min_p": 0.05,
  "top_p": 1.0,
  "autocast": "off",
  "seed": -1,
  "client_request_id": "optional-unique-retry-key"
}
```

Retrying identical parameters with the same `client_request_id` returns the existing job. Reusing it with different parameters is rejected. For remote or isolated HTTP clients, upload raw reference audio first, then pass its ID; the API does not accept arbitrary server-side reference paths.

## Worker and local data

`chatter-tts start`, `stop`, and `doctor` manage or inspect the worker. `serve` runs in the foreground for diagnostics. The queue is limited to 32 active jobs and persists across restarts. Interrupted running jobs become failed; pending jobs resume. One worker lock prevents two workers from sharing a data directory.

The default is to unload models after each job, allowing FLUX3-MCP, KANITTS2-MCP, and YUE2-MCP to use the card. Their workers are independent; this project does not stop them automatically. For repeated narration:

```powershell
chatter-tts stop
chatter-tts configure --residency resident
chatter-tts start
# Restore per-job model release:
chatter-tts stop
chatter-tts configure --residency unload
```

The CUDA allocator budget defaults to 8 GiB. T3, S3Gen, and voice-encoder parameters and buffers are audited on CUDA. CPU work handles loading, tokenization, reference decoding/resampling, watermark processing, file I/O, and metrics. No CPU model-offload fallback is configured. Allocator limits do not cap all CUDA driver allocations.

Model loading initializes the modules on the selected CUDA device and streams checkpoint tensors individually. This avoids full CPU parameter copies and whole-checkpoint memory mappings, which can exhaust Windows commit memory even with free GPU VRAM. The largest tensor in the pinned checkpoint is about 32 MiB; temporary decoding buffers can coexist. Loading still requires host memory for Python and the CUDA runtime, and cold starts can take longer on a busy machine.

Voice conditions are reset to the packaged speaker before and after every job. A reference supplied by one client therefore does not become another client's default voice. Long text is split into segments of at most 280 characters, preferring sentence boundaries, with 150 ms joins. Prosody can change at joins. Input is limited to 12,000 characters. Cancellation is checked between stages and on T3 forward passes; loading, reference processing, or a running S3Gen operation must finish before cancellation is observed.

Set `CHATTER_TTS_HOME` to change the data directory. The legacy `CHATTERBOX_HOME` remains an alias for the project root, with data beneath its `data` folder. Default files:

```text
data/config.json       Token, GPU UUID, port, allocator budget, model path, residency
data/models/chatterbox Pinned five-file English checkpoint
data/voices/           Existing named references and imported voice IDs
data/outputs/          Job WAVs and metadata
data/jobs.sqlite3      Persistent queue
data/service.log       Worker diagnostics
```

All model files, references, outputs, credentials, environments, and caches are ignored by Git. Model downloads resume by verified ranges. If `doctor` reports missing files, rerun `chatter-tts setup-models`. If startup fails, inspect `data/service.log` and check whether another process owns port 8769.

Autostart waits for an existing worker to finish starting or shutting down instead of launching additional workers against its lock. It monitors newly launched workers and includes their latest startup-log error when they exit or fail readiness checks. A held live lock is never deleted to force another worker through.

## Runtime and verification

The upstream package pins Torch 2.6 and includes web/multilingual dependencies unrelated to this English service. This project bundles the required English inference sources from the verified `chatterbox-tts==0.1.7` wheel, retains their notices, and declares the tested runtime dependencies directly. Its replacement package initializer avoids eager imports of unused applications. `uv pip check` passes; no dependency override or `--no-deps` installation is required.

[Source hashes](docs/upstream-source.json), [model hashes](docs/model-manifest.json), and [locked dependencies](requirements.lock.txt) record the installation inputs. Generated speech still uses upstream's Perth watermarker.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/test.ps1
# Opt-in real GPU validation, including persistent-worker voice isolation:
chatter-tts stop
chatter-tts configure --residency resident
.\.venv\Scripts\python.exe scripts/validate_mcp_live.py
```

The live script generates speech, clones a synthetic reference, retrieves MCP audio, verifies default-voice isolation by WAV hash, renders longer text through HTTP MCP, and cancels a running job. It writes `docs/live-mcp-validation.json`. Restore `--residency unload` after validation if you want the default memory policy.

The code and English model weights are MIT-licensed upstream; individual source files retain their own third-party notices, including Apache-2.0 sections. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Recordings and weights are not redistributed in this repository.
