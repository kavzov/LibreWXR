# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Pipeline launch ownership and render-worker configuration regressions."""

import importlib
import subprocess
import sys
from unittest.mock import Mock

import pytest
import uvicorn

from librewxr.config import Settings


@pytest.fixture
def launch(tmp_path, monkeypatch):
    from librewxr import main

    for name in ("LIBREWXR_CACHE_DIR", "LIBREWXR_RENDER_ONLY", "LIBREWXR_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    config = Settings(_env_file=None, workers=1, cache_dir=str(tmp_path))
    monkeypatch.setattr(main, "settings", config)
    child = Mock(pid=12345)
    child.poll.return_value = None
    child.wait.return_value = 0
    popen = Mock(return_value=child)
    monkeypatch.setattr(main.subprocess, "Popen", popen)
    monkeypatch.setattr(main.threading, "Thread", Mock())
    monkeypatch.setattr(main.signal, "signal", Mock())
    monkeypatch.setattr(main.atexit, "register", Mock())
    monkeypatch.setattr(main.atexit, "unregister", Mock())
    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    # main replaces the supervisor for multi-worker launches; restore it
    # after each test so subsequent launches keep the original class.
    supervisor = importlib.import_module("uvicorn.main")
    monkeypatch.setattr(supervisor, "Multiprocess", supervisor.Multiprocess)
    return main, config, child, popen, run


@pytest.mark.parametrize("server_fails", [False, True])
def test_auto_launch_shares_cache_and_stops_pipeline(launch, server_fails):
    main, config, child, popen, run = launch
    if server_fails:
        run.side_effect = RuntimeError("server failed")
        with pytest.raises(RuntimeError, match="server failed"):
            main.main()
    else:
        main.main()

    args, kwargs = popen.call_args
    assert args[0] == [sys.executable, "-m", "librewxr.data_pipeline"]
    assert kwargs["env"]["LIBREWXR_CACHE_DIR"] == config.cache_dir
    assert "LIBREWXR_RENDER_ONLY" not in kwargs["env"]
    assert config.render_only is True
    assert run.call_args.kwargs["workers"] == 1
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=10)
    main.atexit.unregister.assert_called_once()


def test_auto_launch_honours_dotenv_worker_count(launch, tmp_path):
    main, _, child, _, run = launch
    env_file = tmp_path / ".env"
    env_file.write_text("LIBREWXR_WORKERS=3\n")
    main.settings = Settings(_env_file=env_file, cache_dir=str(tmp_path))
    main.main()
    assert run.call_args.kwargs["workers"] == 3
    child.terminate.assert_called_once()


def test_auto_launch_reaps_pipeline_after_shutdown_timeout(launch):
    main, _, child, _, _ = launch
    child.wait.side_effect = [subprocess.TimeoutExpired("pipeline", 10), -9]
    main.main()
    child.terminate.assert_called_once()
    child.kill.assert_called_once()
    assert child.wait.call_count == 2


def test_dedicated_renderer_does_not_spawn_pipeline(launch):
    main, config, _, popen, run = launch
    config.render_only = True
    main.main()
    popen.assert_not_called()
    assert run.call_args.kwargs["workers"] == config.workers
