from unittest.mock import Mock

import pytest
from filelock import FileLock

from chatterbox_mcp.client import Client, startup_error
from chatterbox_mcp.config import Settings


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows hidden worker")
def test_no_duplicate_launch_while_old_worker_holds_lock(tmp_path, monkeypatch):
    client = Client(Settings(home=tmp_path, token="test"))
    monkeypatch.setattr(client, "reachable", lambda: False)
    monkeypatch.setattr("chatterbox_mcp.client.runtime_environment", lambda: {})
    launch = Mock(side_effect=AssertionError("Attempted duplicate worker launch"))
    monkeypatch.setattr("chatterbox_mcp.windows.start_service", launch)
    monkeypatch.setattr("chatterbox_mcp.client.STARTUP_TIMEOUT", 0.05, raising=False)
    with FileLock(str(tmp_path / "worker.lock")):
        with pytest.raises(RuntimeError, match=r"holds.*worker.lock"):
            client.ensure()
    launch.assert_not_called()


def test_startup_error_reports_only_new_log_output(tmp_path):
    settings = Settings(home=tmp_path)
    path = tmp_path / "service.log"
    path.write_text("Old unrelated startup error\n")
    offset = path.stat().st_size
    with path.open("a") as f:
        f.write("OSError: The paging file is too small (1455)\n")
    message = startup_error(settings, offset)
    assert "paging file is too small" in message and "Old unrelated" not in message


@pytest.mark.skipif(
    __import__("os").name != "nt", reason="Windows worker exit monitoring"
)
def test_exited_worker_reports_its_startup_error(tmp_path, monkeypatch):
    client = Client(Settings(home=tmp_path, token="test"))
    monkeypatch.setattr(client, "reachable", lambda: False)
    monkeypatch.setattr("chatterbox_mcp.client.runtime_environment", lambda: {})
    monkeypatch.setattr("chatterbox_mcp.client.STARTUP_TIMEOUT", 1.2)

    def launch(*args):
        (tmp_path / "service.log").write_text(
            "RuntimeError: specific startup failure\n"
        )
        return 123456789

    monkeypatch.setattr("chatterbox_mcp.windows.start_service", launch)
    monkeypatch.setattr("chatterbox_mcp.windows.process_running", lambda pid: False)
    with pytest.raises(RuntimeError, match="specific startup failure"):
        client.ensure()
