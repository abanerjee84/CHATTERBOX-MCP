"""Keep existing CLI flags, reference names and batch files working through the shared worker."""

import json
import time
from pathlib import Path

from .config import gpu_inventory
from .schema import SpeechRequest


def apply_profile(args, settings, client):
    changes = {}
    if args.gpu_uuid:
        matches = [
            g
            for g in gpu_inventory()
            if g["uuid"].removeprefix("GPU-").casefold()
            == args.gpu_uuid.removeprefix("GPU-").casefold()
            and "RTX 3060" in g["name"]
        ]
        if len(matches) != 1:
            raise ValueError(
                "Select an available RTX 3060 UUID; other cards are not used"
            )
        changes["gpu_uuid"] = matches[0]["uuid"]
    if args.vram_cap_gib is not None:
        if not 0.5 <= args.vram_cap_gib <= 10.5:
            raise ValueError("VRAM cap must be 0.5-10.5 GiB")
        changes["memory_gib"] = args.vram_cap_gib
    if args.model_dir:
        changes["model_dir"] = str(Path(args.model_dir).resolve())
    if any(getattr(settings, k) != v for k, v in changes.items()):
        if client.reachable():
            raise RuntimeError(
                "Stop the worker before changing GPU, VRAM cap or model directory"
            )
        for k, v in changes.items():
            setattr(settings, k, v)
        settings.save()


def stop_service(client):
    if not client.reachable():
        return {"status": "not_running"}
    client._request("POST", "/admin/shutdown")
    deadline = time.monotonic() + 45
    while client.reachable():
        if time.monotonic() > deadline:
            raise TimeoutError("Worker is still stopping")
        time.sleep(0.2)
    from filelock import FileLock

    with FileLock(str(client.settings.home / "worker.lock"), timeout=35):
        pass
    return {"status": "stopped"}


def prepare_requests(args, client):
    if args.request:
        if args.voice is not None or args.voice_id is not None:
            raise ValueError(
                "Put voice or voice_id inside the JSON when using --request"
            )
        raw = json.loads(args.request.read_text("utf-8"))
        entries = raw if isinstance(raw, list) else [raw]
    else:
        entries = [
            {
                "text": args.text_file.read_text("utf-8")
                if args.text_file
                else args.text,
                "voice_id": args.voice_id,
                "voice": args.voice,
                "exaggeration": args.exaggeration,
                "cfg_weight": args.cfg_weight,
                "temperature": args.temperature,
                "repetition_penalty": args.repetition_penalty,
                "min_p": args.min_p,
                "top_p": args.top_p,
                "seed": args.seed,
                "client_request_id": args.request_id,
            }
        ]
    if not entries or len(entries) > 32:
        raise ValueError("Provide 1-32 speech requests")
    if len(entries) > 1 and args.out:
        raise ValueError("For batches, put a separate out path in each JSON entry")
    pending = []
    for source in entries:
        if not isinstance(source, dict):
            raise ValueError("Each request must be a JSON object")
        data = dict(source)
        out = data.pop("out", None)
        reference = data.pop("voice", None)
        data.setdefault("autocast", args.autocast or "off")
        data.setdefault(
            "change_speed", args.change_speed if args.change_speed is not None else 1.0
        )
        if reference is not None and data.get("voice_id"):
            raise ValueError("Choose reference voice or voice_id, not both")
        req = SpeechRequest.model_validate(data)
        pending.append((req, reference, args.out or (Path(out) if out else None)))
    prepared = []
    for req, reference, out in pending:
        if reference is not None:
            req = SpeechRequest.model_validate(
                req.model_dump() | {"voice_id": client.resolve_voice(reference)}
            )
        prepared.append((req, out))
    return prepared


def generate(args, client, emit):
    prepared = prepare_requests(args, client)
    jobs = []
    try:
        for req, out in prepared:
            jobs.append((client.submit(req.model_dump()), out))
    except Exception as exc:
        emit(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "submitted_jobs": [job for job, _ in jobs],
            }
        )
        return 1
    results = []
    for job, out in jobs:
        if args.wait or out:
            job = client.wait(job["job_id"], args.timeout)
        if out and job["status"] == "succeeded":
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(client.artifact(job["job_id"]))
            job["saved_copy"] = str(out.resolve())
        results.append(job)
    emit(results[0] if len(results) == 1 else {"jobs": results})
    return int(any(j["status"] in {"failed", "cancelled"} for j in results))
