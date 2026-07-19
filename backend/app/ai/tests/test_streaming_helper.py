from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from app.ai import streaming_helper
from app.ai.model_router import staggered_gather
from app.ai.streaming_helper import acall_with_heartbeat, invoke_with_heartbeat


class RecordingRouter:
    def __init__(self) -> None:
        self.ainvoke_calls: list[tuple[str, dict[str, Any]]] = []
        self.ainvoke_tasks: list[asyncio.Task[Any] | None] = []
        self.invoke_calls: list[tuple[str, dict[str, Any]]] = []
        self.acall_calls: list[
            tuple[str, list[dict[str, str]], dict[str, Any] | None]
        ] = []
        self.acall_tasks: list[asyncio.Task[Any] | None] = []

    async def ainvoke(self, capability: str, payload: dict[str, Any]) -> dict:
        self.ainvoke_calls.append((capability, payload))
        self.ainvoke_tasks.append(asyncio.current_task())
        return {"content": {"ok": True}}

    def invoke(self, capability: str, payload: dict[str, Any]) -> dict:
        self.invoke_calls.append((capability, payload))
        return {"content": {"wrong_route": True}}

    async def acall_capability(
        self,
        capability: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
    ) -> str:
        self.acall_calls.append((capability, messages, response_format))
        self.acall_tasks.append(asyncio.current_task())
        return "ok"


class BlockingRouter(RecordingRouter):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def acall_capability(
        self,
        capability: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None,
    ) -> str:
        self.acall_calls.append((capability, messages, response_format))
        self.acall_tasks.append(asyncio.current_task())
        self.started.set()
        await self.release.wait()
        return "ok"


@pytest.mark.asyncio
async def test_invoke_heartbeat_does_not_wrap_router_in_another_timeout() -> None:
    router = RecordingRouter()
    events: list[str] = []
    caller_task = asyncio.current_task()

    result = await invoke_with_heartbeat(
        router,
        "generate",
        {"name": "elm"},
        heartbeat_interval=0.001,
        event_callback=lambda event, *_: events.append(event),
    )

    assert result == {"content": {"ok": True}}
    assert router.ainvoke_calls == [("generate", {"name": "elm"})]
    assert router.ainvoke_tasks == [caller_task]
    assert router.invoke_calls == []
    assert events[0] == "ai_request_start"
    assert events[-1] == "ai_request_complete"


@pytest.mark.asyncio
async def test_acall_heartbeat_awaits_router_directly() -> None:
    router = RecordingRouter()
    messages = [{"role": "user", "content": "Name a tree"}]
    response_format = {"type": "json_object"}
    caller_task = asyncio.current_task()

    result = await acall_with_heartbeat(
        router,
        "generate",
        messages,
        response_format,
        heartbeat_interval=0.001,
    )

    assert result == "ok"
    assert router.acall_calls == [("generate", messages, response_format)]
    assert router.acall_tasks == [caller_task]


def test_ordinary_heartbeat_helpers_do_not_accept_wrapper_timeouts() -> None:
    assert "timeout" not in inspect.signature(invoke_with_heartbeat).parameters
    assert "timeout" not in inspect.signature(acall_with_heartbeat).parameters


def test_streaming_helper_defines_invoke_wrapper_once() -> None:
    source = Path(streaming_helper.__file__).read_text(encoding="utf-8")
    assert source.count("async def invoke_with_heartbeat(") == 1


@pytest.mark.asyncio
async def test_acall_heartbeat_cancellation_closes_heartbeat_and_propagates() -> None:
    router = BlockingRouter()
    events: list[str] = []
    request_task = asyncio.create_task(
        acall_with_heartbeat(
            router,
            "generate",
            [{"role": "user", "content": "Name a tree"}],
            heartbeat_interval=0.001,
            event_callback=lambda event, *_: events.append(event),
        )
    )
    await router.started.wait()

    request_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request_task
    await asyncio.sleep(0)

    assert events[0] == "ai_request_start"
    assert events[-1] == "ai_request_cancelled"
    assert "ai_request_complete" not in events
    assert "ai_request_error" not in events
    assert not any(
        "send_heartbeats" in task.get_coro().__qualname__
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
    )


@pytest.mark.asyncio
async def test_staggered_gather_leaves_child_timeout_as_child_error() -> None:
    child_timeout = TimeoutError("child owns its deadline")
    events: list[str] = []

    async def timed_out_child() -> None:
        raise child_timeout

    results = await staggered_gather(
        [timed_out_child()],
        interval=0,
        max_concurrent=1,
        task_name="child",
        event_callback=lambda event, *_: events.append(event),
    )

    assert results == [child_timeout]
    assert "ai_parallel_task_error" in events
    assert "ai_parallel_task_timeout" not in events
    assert "task_timeout" not in inspect.signature(staggered_gather).parameters
