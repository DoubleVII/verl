"""Exercise HTTP startup with real sockets and the installed Uvicorn."""

import asyncio
from types import SimpleNamespace

import aiohttp
import pytest
from fastapi import FastAPI

from verl.workers.rollout.utils import run_unvicorn


@pytest.mark.parametrize("loop_kind", ["asyncio", "uvloop"])
def test_http_server_remains_reachable_after_startup(loop_kind):
    async def exercise():
        app = FastAPI()

        @app.post("/release_memory_occupation")
        async def release(payload: dict):
            return {"released": payload["tags"]}

        port, task = await run_unvicorn(app, SimpleNamespace(), "127.0.0.1")
        try:
            async with aiohttp.ClientSession() as session:
                for _ in range(2):
                    async with session.post(
                        f"http://127.0.0.1:{port}/release_memory_occupation",
                        json={"tags": ["kv_cache", "weights"]},
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as response:
                        assert response.status == 200
                        assert await response.json() == {"released": ["kv_cache", "weights"]}
            assert not task.done()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    if loop_kind == "uvloop":
        pytest.importorskip("uvloop").run(exercise())
    else:
        asyncio.run(exercise())


def test_http_startup_failure_does_not_return_port(monkeypatch):
    from contextlib import asynccontextmanager

    from verl.workers.rollout import utils

    @asynccontextmanager
    async def failed_lifespan(app):
        raise RuntimeError("application startup failed")
        yield

    async def exercise():
        app = FastAPI(lifespan=failed_lifespan)
        port, sock = utils.get_free_port("127.0.0.1")
        monkeypatch.setattr(utils, "get_free_port", lambda address: (port, sock))
        with pytest.raises(RuntimeError, match="exited before startup"):
            await asyncio.wait_for(run_unvicorn(app, SimpleNamespace(), "127.0.0.1"), timeout=5)
        assert sock.fileno() == -1

    asyncio.run(exercise())


def test_http_bind_failure_retries_then_reports_error(monkeypatch):
    from unittest.mock import Mock

    from verl.workers.rollout import utils

    get_port = Mock(side_effect=OSError("cannot bind"))
    monkeypatch.setattr(utils, "get_free_port", get_port)
    with pytest.raises(RuntimeError, match="after 3 retries"):
        asyncio.run(run_unvicorn(FastAPI(), SimpleNamespace(), "127.0.0.1", max_retries=3))
    assert get_port.call_count == 3
