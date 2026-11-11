# Third-party notices

The `src/chatterbox_mcp/_upstream/` folder contains the English inference modules from Resemble AI's `chatterbox-tts` 0.1.7 wheel. Wheel SHA-256: `83782500e3ad4e7c919132e9d7eb8755f29f57c5bde5ec48c655ca23a4eb113c`. Source files and individual SHA-256 values are recorded in [upstream-source.json](docs/upstream-source.json).

Original source: [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox), distributed through [PyPI 0.1.7](https://pypi.org/project/chatterbox-tts/0.1.7/). Copyright (c) 2025 Resemble AI. The MIT license is retained in [licenses/Chatterbox-MIT.txt](licenses/Chatterbox-MIT.txt). The package initializer is replaced with an import-free initializer so unrelated multilingual and voice-conversion applications are not eagerly loaded. Inference source files are otherwise copied without changes.

Individual modules include attribution to CosyVoice, Matcha-TTS, HiFi-GAN, and other components. Their original headers remain in place. Apache-2.0 text is provided in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt); those notices are not replaced by the integration's description of upstream's MIT license.

English model weights are downloaded separately from [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox), revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`. The model repository identifies its license as MIT. Only `ve.safetensors`, `t3_cfg.safetensors`, `s3gen.safetensors`, `tokenizer.json`, and `conds.pt` are used.

The upstream Perth watermarker remains enabled. Its installed package provides its own license and notices. PyTorch, Transformers, Diffusers, S3Tokenizer, librosa, and other installed dependencies retain their respective licenses.

Queue, client, registration, download, Windows launch, and service structure are adapted from the user's KANITTS2-MCP project. Reference recordings, generated audio, checkpoints, environments, caches, and tokens are not committed to this repository.
