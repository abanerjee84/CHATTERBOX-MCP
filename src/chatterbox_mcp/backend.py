import copy
import gc
import importlib.util
import json
import random
import re
import time
from contextlib import nullcontext

from .config import select_gpu
from .jobs import Cancelled
from .models import FILES
from .voices import voice_path


def split_text(text, limit=280):
    words = re.findall(r"\S+", text)
    chunks, current = [], ""
    for word in words:
        while len(word) > limit:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(word[:limit])
            word = word[limit:]
        if len(current) + len(word) + 1 > limit:
            chunks.append(current)
            current = ""
        current = (current + " " + word).strip()
        if len(current) >= limit // 2 and re.search(r"[.!?][\"']?$", current):
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def reset_voice(model, default_conditionals):
    """Every request starts from the packaged speaker, including after reference cloning."""
    model.conds = copy.deepcopy(default_conditionals)


class ChatterboxBackend:
    def __init__(self, settings):
        self.settings = settings
        self.model = self.default_conditionals = None
        self.memory = None

    def readiness(self):
        missing = [
            name
            for name in FILES
            if not (self.settings.checkpoint_dir / name).is_file()
        ]
        packages = [
            name
            for name in (
                "torch",
                "torchaudio",
                "transformers",
                "diffusers",
                "librosa",
                "s3tokenizer",
                "perth",
            )
            if importlib.util.find_spec(name) is None
        ]
        return {
            "ready": not missing and not packages and bool(self.settings.gpu_uuid),
            "loaded": self.model is not None,
            "model": "Chatterbox English 500M (0.1.7)",
            "model_dir": str(self.settings.checkpoint_dir),
            "missing_files": missing,
            "missing_packages": packages,
            "gpu_uuid": self.settings.gpu_uuid,
            "sample_rate": 24000,
            "voice_cloning_ready": not missing and not packages,
            "keep_loaded": self.settings.keep_loaded,
            "memory_budget_gib": self.settings.memory_gib,
            "watermark": "Perth (upstream default)",
            "last_memory": self.memory,
        }

    def _load(self, progress):
        ready = self.readiness()
        if not ready["ready"]:
            raise RuntimeError(
                f"Inference unavailable: {ready}. Run chatter-tts setup-models and install inference dependencies."
            )
        select_gpu(self.settings)
        import torch

        from ._upstream.tts import ChatterboxTTS

        if (
            not torch.cuda.is_available()
            or torch.cuda.device_count() != 1
            or "RTX 3060" not in torch.cuda.get_device_name(0)
        ):
            raise RuntimeError(
                "Exactly the configured RTX 3060 must be visible to CUDA"
            )
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(self.settings.memory_gib * 1024**3 / total, 0.9)
        )
        torch.cuda.reset_peak_memory_stats()
        progress("loading_models", 5)
        self.model = ChatterboxTTS.from_local(
            self.settings.checkpoint_dir, device="cuda:0"
        )
        self.default_conditionals = copy.deepcopy(self.model.conds)
        self.audit_gpu()

    def audit_gpu(self):
        tensors = [
            tensor
            for module in (self.model.t3, self.model.s3gen, self.model.ve)
            for tensor in (*module.parameters(), *module.buffers())
        ]
        wrong = [str(t.device) for t in tensors if t.device.type != "cuda"]
        if wrong:
            raise RuntimeError(
                "Model tensors are not all on CUDA: " + ", ".join(wrong[:5])
            )
        return {"all_model_tensors_on_cuda": True, "tensor_count": len(tensors)}

    def unload(self):
        self.model = self.default_conditionals = None
        gc.collect()
        import sys

        if "torch" in sys.modules:
            import torch

            torch.cuda.empty_cache()

    def generate(self, req, job_id, progress, cancelled):
        from .metrics import GPUMemorySampler

        sampler = GPUMemorySampler().start()
        started = time.monotonic()
        try:
            if self.model is None:
                self._load(progress)
            import numpy as np
            import soundfile as sf
            import torch

            torch.cuda.reset_peak_memory_stats()
            random.seed(req.seed)
            np.random.seed(req.seed)
            torch.manual_seed(req.seed)
            torch.cuda.manual_seed_all(req.seed)
            reset_voice(self.model, self.default_conditionals)
            reference = (
                voice_path(self.settings.home, req.voice_id) if req.voice_id else None
            )
            parts, waves = split_text(req.text), []

            def check_cancel(module, args):
                if cancelled.is_set():
                    raise Cancelled()

            hook = self.model.t3.tfmr.register_forward_pre_hook(check_cancel)
            dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(req.autocast)
            try:
                for index, text in enumerate(parts):
                    progress(
                        f"speaking_{index + 1}_of_{len(parts)}",
                        20 + int(65 * index / len(parts)),
                    )
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=dtype) if dtype else nullcontext(),
                    ):
                        wav = self.model.generate(
                            text,
                            audio_prompt_path=str(reference) if reference else None,
                            exaggeration=req.exaggeration,
                            cfg_weight=req.cfg_weight,
                            temperature=req.temperature,
                            repetition_penalty=req.repetition_penalty,
                            min_p=req.min_p,
                            top_p=req.top_p,
                        )
                    samples = wav.detach().float().cpu().numpy().reshape(-1)
                    if not samples.size or not np.isfinite(samples).all():
                        raise ValueError("Model returned empty or invalid audio")
                    waves.append(samples)
                    if index + 1 < len(parts):
                        waves.append(np.zeros(int(24000 * 0.15), dtype="float32"))
            finally:
                hook.remove()
                reset_voice(self.model, self.default_conditionals)
            progress("saving", 95)
            samples = np.concatenate(waves)
            path = self.settings.home / "outputs" / (job_id + ".wav")
            sf.write(path, samples, 24000, subtype="PCM_16")
            self.memory = {
                "peak_allocated_mib": round(
                    torch.cuda.max_memory_allocated() / 1024**2, 1
                ),
                "peak_reserved_mib": round(
                    torch.cuda.max_memory_reserved() / 1024**2, 1
                ),
                "gpu_uuid": self.settings.gpu_uuid,
                "gpu_name": torch.cuda.get_device_name(0),
                "residency": self.audit_gpu(),
                "windows_driver": sampler.stop(),
            }
            result = {
                "audio_path": str(path),
                "audio_url": f"/v1/jobs/{job_id}/audio",
                "sample_rate": 24000,
                "duration_seconds": round(len(samples) / 24000, 3),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "chunks": len(parts),
                "seed": req.seed,
                "voice_id": req.voice_id,
                "watermarked": True,
                "memory": self.memory,
            }
            path.with_suffix(".json").write_text(
                json.dumps({"request": req.model_dump(), "result": result}, indent=2),
                "utf-8",
            )
            return result
        except BaseException:
            self.unload()
            raise
        finally:
            sampler.stop()
            if not self.settings.keep_loaded:
                self.unload()
