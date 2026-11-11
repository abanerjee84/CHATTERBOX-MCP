import asyncio
import base64
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import AudioContent, CallToolResult, TextContent, ToolAnnotations

from .client import Client
from .schema import SpeechRequest


def structured(value):
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(value))],
        structuredContent=value,
    )


def create_mcp(settings, autostart=True):
    client = Client(settings, autostart=autostart)
    mcp = FastMCP(
        "local-chatterbox",
        host="127.0.0.1",
        port=settings.port,
        streamable_http_path="/mcp",
        stateless_http=True,
        instructions="Local Chatterbox English 500M speech and voice cloning on RTX 3060. "
        "generate_speech returns job_id immediately; poll get_speech_job. "
        "Upload clean 3-30 second single-speaker audio and pass voice_id, or choose a local voice "
        "name/path from list_voices using voice. Without a reference the packaged default speaker is used. "
        "exaggeration is an intensity dial, not a named emotion. CFG, sampling and reference prosody "
        "influence delivery. Outputs are 24 kHz mono WAV with upstream Perth watermark. MIT models.",
    )
    write = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, openWorldHint=False
    )
    read = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

    @mcp.tool(annotations=write)
    async def generate_speech(
        text: str,
        voice_id: str | None = None,
        voice: str | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        temperature: float = 0.8,
        repetition_penalty: float = 1.2,
        min_p: float = 0.05,
        top_p: float = 1.0,
        seed: int | None = None,
        client_request_id: str | None = None,
    ) -> CallToolResult:
        """Queue speech with a cloned voice or the default speaker. Use -1/omitted for a random seed.

        voice is an existing local reference path/name; voice_id comes from upload_voice.
        Choose one. exaggeration is 0-1.5; cfg_weight is 0-1. No emotion labels or accent tags.
        """
        if voice is not None and voice_id is not None:
            raise ValueError("Choose voice or voice_id, not both")
        if voice is not None:
            voice_id = await asyncio.to_thread(client.resolve_voice, voice)
        req = SpeechRequest(
            text=text,
            voice_id=voice_id,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
            seed=seed,
            client_request_id=client_request_id,
        )
        return structured(await asyncio.to_thread(client.submit, req.model_dump()))

    @mcp.tool(annotations=write)
    async def upload_voice(
        audio_path: str, name: str = "Reference voice"
    ) -> CallToolResult:
        """Import an absolute local audio path and return a reusable voice_id. Use clean 3-30s speech."""
        return structured(
            await asyncio.to_thread(client.upload_voice, audio_path, name)
        )

    @mcp.tool(annotations=read)
    async def list_voices() -> CallToolResult:
        """List imported voice IDs and existing named local recordings. default selects packaged voice."""
        return structured(await asyncio.to_thread(client.voices))

    @mcp.tool(annotations=read)
    async def get_speech_job(
        job_id: str, include_audio: bool = False
    ) -> CallToolResult:
        """Get status, errors, audio path, duration and memory. Attach completed WAV with include_audio."""
        job = await asyncio.to_thread(client.job, job_id)
        result = structured(job)
        if include_audio and job["status"] == "succeeded":
            if job["result"]["duration_seconds"] > 170:
                raise ValueError(
                    "Use the local path or HTTP download for audio longer than 170 seconds"
                )
            data = await asyncio.to_thread(client.artifact, job_id)
            if len(data) > 8 * 1024**2:
                raise ValueError(
                    "Use the local path or HTTP download for audio larger than 8 MiB"
                )
            result.content.append(
                AudioContent(
                    type="audio",
                    data=base64.b64encode(data).decode(),
                    mimeType="audio/wav",
                )
            )
        return result

    @mcp.tool(annotations=write)
    async def cancel_speech_job(job_id: str) -> CallToolResult:
        """Cancel pending work or stop token generation at the next model forward pass."""
        return structured(await asyncio.to_thread(client.cancel, job_id))

    @mcp.tool(annotations=read)
    async def speech_service_status() -> CallToolResult:
        """Report readiness, selected GPU, allocator budget, residency and latest measurements."""
        return structured(await asyncio.to_thread(client.health))

    return mcp
