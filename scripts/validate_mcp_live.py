"""Opt-in real GPU validation through stdio and HTTP MCP; saves a sanitized report."""

import asyncio
import base64
import hashlib
import io
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from chatterbox_mcp.config import ROOT, load_settings


def unpack(result):
    assert not result.isError, result.content
    return result.structuredContent


async def wait(session, job_id, include_audio=False):
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        response = await session.call_tool("get_speech_job", {"job_id": job_id})
        job = unpack(response)
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            assert job["status"] == "succeeded", job
            if include_audio:
                response = await session.call_tool(
                    "get_speech_job", {"job_id": job_id, "include_audio": True}
                )
                item = next(c for c in response.content if c.type == "audio")
                data = base64.b64decode(item.data)
                samples, rate = sf.read(io.BytesIO(data), dtype="float32")
                assert (
                    rate == 24000 and np.isfinite(samples).all() and len(samples) > rate
                )
                assert np.sqrt(np.mean(samples**2)) > 0.001
            return job
        await asyncio.sleep(1)
    raise TimeoutError(job_id)


def summarize(job):
    result = dict(job["result"])
    path = Path(result.pop("audio_path"))
    result.pop("audio_url")
    result["audio_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["request"] = job["request"]
    return result


async def check_cancellation(session, report):
    queued = unpack(
        await session.call_tool(
            "generate_speech",
            {
                "text": "This is a cancellation test that should stop before the passage is finished. "
                * 40,
                "seed": 76,
            },
        )
    )
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        job = unpack(
            await session.call_tool("get_speech_job", {"job_id": queued["job_id"]})
        )
        if job["stage"].startswith("speaking_"):
            break
        assert job["status"] not in {"succeeded", "failed", "cancelled"}, job
        await asyncio.sleep(0.1)
    else:
        raise TimeoutError("Cancellation test did not reach inference")
    started = time.monotonic()
    unpack(await session.call_tool("cancel_speech_job", {"job_id": queued["job_id"]}))
    for _ in range(200):
        job = unpack(
            await session.call_tool("get_speech_job", {"job_id": queued["job_id"]})
        )
        if job["status"] == "cancelled":
            assert job["result"] is None
            report["cancellation"] = {
                "status": "cancelled",
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            return
        assert job["status"] not in {"succeeded", "failed"}, job
        await asyncio.sleep(0.1)
    raise TimeoutError("Running generation did not cancel")


async def main():
    settings = load_settings()
    report = {"model": "ResembleAI/chatterbox", "runs": {}}
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "chatterbox_mcp", "mcp"],
        env={"CHATTER_TTS_HOME": str(settings.home)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=45)
        ) as session:
            await session.initialize()
            report["tools"] = sorted(t.name for t in (await session.list_tools()).tools)
            request = {
                "text": "This is a real speech generation test through the local MCP connection. The audio stays on this computer.",
                "seed": 73,
            }
            first = unpack(await session.call_tool("generate_speech", request))
            job = await wait(session, first["job_id"], include_audio=True)
            report["runs"]["stdio_speech"] = summarize(job)
            voice = unpack(
                await session.call_tool(
                    "upload_voice",
                    {
                        "audio_path": job["result"]["audio_path"],
                        "name": "Synthetic validation reference",
                    },
                )
            )
            assert voice["voice_id"] in [
                v["voice_id"]
                for v in unpack(await session.call_tool("list_voices", {}))["voices"]
            ]
            cloned = unpack(
                await session.call_tool(
                    "generate_speech",
                    {
                        "text": "This is different text spoken using a reference voice. No training was required.",
                        "voice_id": voice["voice_id"],
                        "seed": 74,
                        "exaggeration": 0.8,
                        "cfg_weight": 0.3,
                    },
                )
            )
            report["runs"]["stdio_cloning"] = summarize(
                await wait(session, cloned["job_id"], include_audio=True)
            )
            repeated = unpack(await session.call_tool("generate_speech", request))
            repeat_job = await wait(session, repeated["job_id"])
            report["runs"]["default_after_cloning"] = summarize(repeat_job)
            report["default_voice_isolated"] = (
                report["runs"]["stdio_speech"]["audio_sha256"]
                == report["runs"]["default_after_cloning"]["audio_sha256"]
            )
            assert report["default_voice_isolated"], (
                "Reference voice leaked or deterministic replay changed"
            )

    async with httpx.AsyncClient(
        headers={"Authorization": "Bearer " + settings.token}, trust_env=False
    ) as http:
        async with streamable_http_client(settings.url + "/mcp", http_client=http) as (
            read,
            write,
            _,
        ):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=45)
            ) as session:
                await session.initialize()
                long_text = " ".join(
                    [
                        "The morning was quiet, and sunlight was falling across the desk.",
                        "A notebook lay open beside a cup of coffee.",
                        "Outside, the trees moved slowly in the breeze, while a few birds called from the garden.",
                        "There was time to read, think, and finish the next page before the rest of the house woke up.",
                        "This final sentence checks that a longer passage can continue across more than one speech segment.",
                    ]
                )
                queued = unpack(
                    await session.call_tool(
                        "generate_speech",
                        {"text": long_text, "voice_id": voice["voice_id"], "seed": 75},
                    )
                )
                long_job = await wait(session, queued["job_id"])
                assert long_job["result"]["chunks"] > 1
                report["runs"]["http_long_cloning"] = summarize(long_job)
                await check_cancellation(session, report)
    target = ROOT / "docs/live-mcp-validation.json"
    target.write_text(json.dumps(report, indent=2), "utf-8")
    print(
        json.dumps(
            {
                "report": str(target),
                "runs": {
                    k: {
                        n: v[n]
                        for n in ("duration_seconds", "elapsed_seconds", "chunks")
                    }
                    for k, v in report["runs"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
