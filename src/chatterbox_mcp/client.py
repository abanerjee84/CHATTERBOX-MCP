from __future__ import annotations

import os
import subprocess
import sys
import time

import httpx
from filelock import FileLock, Timeout

from .config import Settings, runtime_environment
from .jobs import TERMINAL

STARTUP_TIMEOUT = 45


def worker_locked(settings):
    try:
        with FileLock(str(settings.home / "worker.lock"), timeout=0):
            return False
    except Timeout:
        return True


def startup_error(settings, offset=0):
    path = settings.home / "service.log"
    tail = ""
    try:
        with path.open("rb") as log:
            log.seek(0, 2)
            end = log.tell()
            log.seek(max(offset, end - 6000))
            lines = log.read().decode("utf-8", errors="replace").strip().splitlines()
            tail = "\n".join(lines[-12:])
    except OSError:
        pass
    return f"Service did not become ready. See {path}" + (
        f"\nLatest startup output:\n{tail}" if tail else ""
    )


class Client:
    def __init__(self, settings: Settings, autostart=True):
        self.settings = settings
        self.autostart = autostart

    def _request(self, method, path, **kwargs):
        with httpx.Client(
            base_url=self.settings.url,
            timeout=30,
            trust_env=False,
            headers={"Authorization": "Bearer " + self.settings.token},
        ) as http:
            response = http.request(method, path, **kwargs)
        if response.is_error:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1500]}")
        return response

    def reachable(self):
        try:
            data = self._request("GET", "/health", timeout=2).json()
            return data.get("service") == "chatter-tts-local" and data.get(
                "home"
            ) == str(self.settings.home)
        except (httpx.HTTPError, RuntimeError):
            return False

    def ensure(self):
        if self.reachable():
            return
        if not self.autostart:
            raise RuntimeError("Speech service is unavailable")
        with FileLock(
            str(self.settings.home / "startup.lock"), timeout=STARTUP_TIMEOUT + 5
        ):
            if self.reachable():
                return
            deadline = time.monotonic() + STARTUP_TIMEOUT
            while worker_locked(self.settings):
                if self.reachable():
                    return
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "Existing worker holds worker.lock but is not responding; it may be starting or shutting down. "
                        + startup_error(self.settings)
                    )
                time.sleep(0.1)
            env = runtime_environment()
            env["CHATTER_TTS_HOME"] = str(self.settings.home)
            process = None
            pid = None
            log_path = self.settings.home / "service.log"
            offset = log_path.stat().st_size if log_path.exists() else 0
            if os.name == "nt":
                from .windows import start_service

                pid = start_service(
                    [sys.executable, "-m", "chatterbox_mcp", "serve", "--background"],
                    env,
                    self.settings.home,
                )
            else:
                with (self.settings.home / "service.log").open("ab") as log:
                    process = subprocess.Popen(
                        [sys.executable, "-m", "chatterbox_mcp", "serve"],
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        start_new_session=True,
                    )
            deadline = time.monotonic() + STARTUP_TIMEOUT
            launched = time.monotonic()
            while time.monotonic() < deadline:
                if self.reachable():
                    return
                if process is not None and process.poll() is not None:
                    break
                if pid is not None and time.monotonic() - launched > 1:
                    from .windows import process_running

                    if not process_running(pid):
                        break
                time.sleep(0.2)
            raise RuntimeError(startup_error(self.settings, offset))

    def health(self):
        self.ensure()
        return self._request("GET", "/health").json()

    def submit(self, data: dict):
        self.ensure()
        return self._request("POST", "/v1/jobs", json=data).json()

    def job(self, job_id: str):
        self.ensure()
        return self._request("GET", f"/v1/jobs/{job_id}").json()

    def cancel(self, job_id: str):
        self.ensure()
        return self._request("POST", f"/v1/jobs/{job_id}/cancel").json()

    def artifact(self, job_id: str, artifact: str = "audio"):
        self.ensure()
        return self._request("GET", f"/v1/jobs/{job_id}/{artifact}").content

    def upload_voice(self, audio_path: str, name: str = "Reference voice"):
        from pathlib import Path

        from .schema import MAX_REFERENCE_BYTES

        path = Path(audio_path)
        if not path.is_absolute() or not path.is_file():
            raise ValueError("Provide an absolute local reference audio path")
        if path.stat().st_size > MAX_REFERENCE_BYTES:
            raise ValueError("Reference exceeds 32 MiB")
        self.ensure()
        return self._request(
            "POST", "/v1/voices", content=path.read_bytes(), params={"name": name}
        ).json()

    def voices(self):
        self.ensure()
        return self._request("GET", "/v1/voices").json()

    def resolve_voice(self, value: str | None):
        """Preserve legacy file paths and local voice names without confusing them with IDs."""
        from pathlib import Path

        if value is None or value == "default":
            return None
        path = Path(value)
        if not path.is_file() and path.name == value:
            candidate = (
                self.settings.home
                / "voices"
                / (value if path.suffix else value + ".wav")
            )
            if candidate.is_file():
                path = candidate
        if not path.is_file():
            raise ValueError(
                f"Unknown reference voice: {value}. Use chatter-tts voices or an existing audio path."
            )
        return self.upload_voice(str(path.resolve()), path.stem)["voice_id"]

    def wait(self, job_id: str, timeout: float = 1800):
        deadline = time.monotonic() + timeout
        while True:
            result = self.job(job_id)
            if result["status"] in TERMINAL:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Still running; resume with chatter-tts status {job_id}"
                )
            time.sleep(0.5)
