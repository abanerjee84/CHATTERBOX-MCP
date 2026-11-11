import argparse
import json
import logging
import sys
from pathlib import Path

from .client import Client
from .config import gpu_inventory, load_settings


def emit(value):
    print(json.dumps(value, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(
        prog="chatter-tts",
        description="Local Chatterbox English speech and cloning",
        allow_abbrev=False,
    )
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--model-dir")
    parser.add_argument("--vram-cap-gib", type=float)
    parser.add_argument("--autocast", choices=["off", "fp16", "bf16"], default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in (
        "start",
        "stop",
        "doctor",
        "mcp",
        "install-cli",
        "setup-models",
        "voices",
        "mcp-config",
    ):
        sub.add_parser(cmd)
    serve = sub.add_parser("serve")
    serve.add_argument("--background", action="store_true", help=argparse.SUPPRESS)
    register = sub.add_parser("register")
    register.add_argument("harness", choices=["codex", "claude-code", "gemini"])
    configure = sub.add_parser("configure")
    configure.add_argument("--residency", choices=["resident", "unload"])
    voice = sub.add_parser("upload-voice")
    voice.add_argument("audio_path", type=Path)
    voice.add_argument("--name", default="Reference voice")
    gen = sub.add_parser("generate", allow_abbrev=False)
    inputs = gen.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--text")
    inputs.add_argument("--text-file", type=Path)
    inputs.add_argument(
        "--request", type=Path, help="Request object or legacy JSON array of jobs"
    )
    speakers = gen.add_mutually_exclusive_group()
    speakers.add_argument(
        "--voice", "--reference-audio", help="Local reference path/name, or default"
    )
    speakers.add_argument("--voice-id")
    gen.add_argument("--exaggeration", type=float, default=0.5)
    gen.add_argument("--cfg-weight", type=float, default=0.5)
    gen.add_argument("--min-p", type=float, default=0.05)
    gen.add_argument("--top-p", type=float, default=1.0)
    gen.add_argument(
        "--change_speed",
        "--change-speed",
        type=float,
        default=None,
        help="Final audio tempo, 0.5-1.5: lower is slower, higher is faster; preserves pitch (default 1)",
    )
    gen.add_argument(
        "--autocast", choices=["off", "fp16", "bf16"], default=argparse.SUPPRESS
    )
    gen.add_argument("--temperature", type=float, default=0.8)
    gen.add_argument("--repetition-penalty", type=float, default=1.2)
    gen.add_argument("--seed", type=int)
    gen.add_argument("--request-id")
    gen.add_argument("--wait", action="store_true")
    gen.add_argument("--timeout", type=float, default=3600)
    gen.add_argument("--out", type=Path)
    for cmd in ("status", "cancel", "wait", "download"):
        item = sub.add_parser(cmd)
        item.add_argument("job_id")
        if cmd == "wait":
            item.add_argument("--timeout", type=float, default=3600)
        if cmd == "download":
            item.add_argument("--out", type=Path, required=True)
            item.add_argument(
                "--artifact", choices=["audio", "metadata"], default="audio"
            )
    args = parser.parse_args()
    settings = load_settings()
    client = Client(settings)
    try:
        from .compat import apply_profile

        apply_profile(args, settings, client)
        if args.command == "serve":
            if args.background:
                sys.stdout = sys.stderr = (settings.home / "service.log").open(
                    "a", encoding="utf-8", buffering=1
                )
            import uvicorn
            from filelock import FileLock

            from .service import create_app

            with FileLock(str(settings.home / "worker.lock"), timeout=0):
                logging.basicConfig(level=logging.INFO)
                app = create_app(settings)
                server = uvicorn.Server(
                    uvicorn.Config(app, host="127.0.0.1", port=settings.port, workers=1)
                )
                app.state.shutdown = lambda: setattr(server, "should_exit", True)
                server.run()
        elif args.command == "mcp":
            from .mcp_adapter import create_mcp

            create_mcp(settings).run(transport="stdio")
        elif args.command == "start":
            emit(client.health())
        elif args.command == "stop":
            from .compat import stop_service

            emit(stop_service(client))
        elif args.command == "doctor":
            from .backend import ChatterboxBackend

            emit(
                {
                    "home": str(settings.home),
                    "gpus": gpu_inventory(),
                    "service_running": client.reachable(),
                    "inference": client.health()["inference"]
                    if client.reachable()
                    else ChatterboxBackend(settings).readiness(),
                }
            )
        elif args.command == "setup-models":
            from .models import setup_models

            emit(setup_models(settings))
        elif args.command == "configure":
            if client.reachable():
                raise RuntimeError("Stop the service before changing residency")
            if args.residency:
                settings.keep_loaded = args.residency == "resident"
            settings.save()
            emit({"keep_loaded": settings.keep_loaded})
        elif args.command in {"install-cli", "register", "mcp-config"}:
            from .install import install_cli, mcp_entry, register_codex, register_json

            if args.command == "install-cli":
                emit(install_cli(settings))
            elif args.command == "register":
                emit(
                    register_codex(settings)
                    if args.harness == "codex"
                    else register_json(settings, args.harness)
                )
            else:
                emit({"mcpServers": {"local-chatterbox": mcp_entry(settings)}})
        elif args.command == "upload-voice":
            emit(client.upload_voice(str(args.audio_path.resolve()), args.name))
        elif args.command == "voices":
            emit(client.voices())
        elif args.command == "generate":
            from .compat import generate

            return generate(args, client, emit)
        elif args.command in {"status", "cancel", "wait"}:
            job = getattr(
                client,
                {"status": "job", "cancel": "cancel", "wait": "wait"}[args.command],
            )(
                args.job_id,
                **({"timeout": args.timeout} if args.command == "wait" else {}),
            )
            emit(job)
            return 1 if args.command == "wait" and job["status"] != "succeeded" else 0
        elif args.command == "download":
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(client.artifact(args.job_id, args.artifact))
            emit({"saved": str(args.out.resolve())})
        return 0
    except Exception as exc:
        emit({"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
