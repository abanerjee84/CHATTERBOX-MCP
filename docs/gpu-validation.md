# GPU and integration validation

Validated 2026-11-11 on Windows x64, RTX 3060 12 GB, driver 610.88, Python 3.11.14, PyTorch 2.8.0+cu128, Transformers 5.2.0, and the pinned Chatterbox 0.1.7 English inference sources. Model inference uses FP32 with autocast off. The RTX 5060 Ti is excluded through CUDA visibility before importing Torch.

| Operation | Audio | Backend time | Peak reserved | Driver peak dedicated |
| --- | ---: | ---: | ---: | ---: |
| Stdio MCP default speech, including loading | 5.76 s | 20.813 s | 3,388 MiB | 3,519 MiB |
| Stdio MCP reference cloning, warm | 4.84 s | 5.469 s | 3,388 MiB | 3,519 MiB |
| Default speech after cloning, warm | 5.76 s | 5.875 s | 3,388 MiB | 3,519 MiB |
| HTTP MCP reference cloning, two segments | 19.95 s | 19.437 s | 3,544 MiB | 3,675 MiB |

The first and third requests used identical text and seed and produced identical WAV hashes. This checks that a resident worker restores the default speaker after a cloned-voice request. The longer passage reused the synthetic reference and ran across two segments. Both short successful requests were retrieved as MCP audio attachments and checked for finite, non-silent 24 kHz samples.

A real running T3 generation was cancelled through MCP in 0.437 seconds after the cancel request. The cancelled job had no successful result. Model loading, reference processing, and a running S3Gen operation remain cancellation boundaries rather than instant interruptions.

Measurements used resident mode and an 8 GiB CUDA allocator budget. Every registered T3, S3Gen, and voice-encoder parameter/buffer was verified on CUDA; the observed audit counted 2,801 tensors. The CUDA driver sampler recorded 78 MiB of shared memory. That counter includes driver mappings and is not proof of model offloading. Driver sampling can miss short peaks.

Backend elapsed time includes loading when applicable, inference, watermarking, saving, and metric collection. It excludes CLI launch, public worker startup, and queue time. These are individual example runs, not latency percentiles or controlled comparisons between models. Synthetic-reference cloning validates the conditioning pipeline; voice identity similarity and perceptual quality were not scored.

The original CLI was also exercised before migration with the existing `baddy-neutral.wav` reference: 4.12 seconds of audio in 6.08 seconds generation time, 19.94 seconds including loading, and 3,622 MiB peak reserved memory. Those original CLI measurements used its own metric format. The prior experiment notes were preserved separately and are not presented as new measurements.

The 40 automated tests pass, covering both actual MCP transports, hidden-worker lifetime after disconnect, authentication and origin checks, missing-weight failures, queue serialization/persistence, retry identity, queued/running cancellation, reference upload/resampling/deduplication, request bounds, artifact retrieval, CLI default parity, legacy reference flags, batch prevalidation, and batch failure exit codes. Lint passes and `uv pip check` reports no conflicts. A wheel was built and checked for bundled model source files.

See [live-mcp-validation.json](live-mcp-validation.json) for sanitized measurements, requests, and WAV hashes. Local reference recordings, model weights, token/config files, and generated audio remain outside Git.

## Windows startup and commit-memory fix

A later real-user run recorded Windows error 1455 (paging-file/commit exhaustion) while the original loader mapped `s3gen.safetensors`, followed by worker-lock failures during repeated launch attempts. The loader now initializes modules on CUDA and reads weights tensor-by-tensor, validating each with the safetensors decoder. It does not allocate full CPU model copies or map complete checkpoint files.

The user's supplied text, local reference recording, CFG 1, temperature 1, and seed 403 were then run successfully through CLI autostart. The result was 7.60 seconds of finite, non-silent 24 kHz audio, with 3,438 MiB peak reserved and 3,569 MiB peak dedicated VRAM. Backend time including cold loading was 63.594 seconds on the busy machine. This is a different workload/system state from the earlier table, not a speed comparison. The recording is not committed.

Startup now waits for an existing worker lock to clear, watches the launched process for exit, uses bounded readiness probes, and reports new startup-log output on failure. Regression tests cover duplicate launches and error reporting, while loader tests cover numeric weight/buffer parity, missing tensors, duplicate header keys, truncation, trailing data, and shape mismatches. The current total is 40 passing tests; the dependency check is clean. See [startup-fix-validation.json](startup-fix-validation.json).

## Final output speed

`change_speed` adds pitch-preserving tempo adjustment after segment assembly. A real CLI request using the same reference, text, and seed as the startup check produced 7.60 seconds before adjustment and 5.06 seconds in the final WAV at speed 1.5. The final watermark pass can round a small number of samples at frame boundaries. The actual saved duration is reported in metadata.

The `--out` file, HTTP audio artifact, and MCP audio attachment were verified byte-identical and contained the adjusted waveform. See [speed-validation.json](speed-validation.json). Tests also check duration and pitch at speeds 0.5, 0.75, 1.25, and 1.5, exact sample preservation at 1, range validation, CLI aliases, batch defaults, MCP transport, and retries of older jobs without a speed field. The current total is 54 passing tests; lint passes.
