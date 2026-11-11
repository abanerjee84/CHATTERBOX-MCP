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

The 31 automated tests pass, covering both actual MCP transports, hidden-worker lifetime after disconnect, authentication and origin checks, missing-weight failures, queue serialization/persistence, retry identity, queued/running cancellation, reference upload/resampling/deduplication, request bounds, artifact retrieval, CLI default parity, legacy reference flags, batch prevalidation, and batch failure exit codes. Lint passes and `uv pip check` reports no conflicts. A wheel was built and checked for bundled model source files.

See [live-mcp-validation.json](live-mcp-validation.json) for sanitized measurements, requests, and WAV hashes. Local reference recordings, model weights, token/config files, and generated audio remain outside Git.
