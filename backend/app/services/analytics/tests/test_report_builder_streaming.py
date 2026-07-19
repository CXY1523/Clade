from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from app.simulation.environment import ParsedPressure
from app.services.analytics.report_builder import ReportBuilder
from app.services.analytics.report_builder_v2 import ReportBuilderV2
from app.services.analytics.turn_report import TurnReportService


class InterruptedStreamRouter:
    def __init__(self, partial: str) -> None:
        self.partial = partial

    async def astream(self, capability: str, payload: dict[str, Any]):
        yield {"type": "status", "state": "receiving"}
        yield self.partial
        yield {
            "type": "status",
            "state": "interrupted",
            "reason": "outbound_timeout",
            "partial": True,
        }

    async def astream_capability(
        self,
        capability: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ):
        yield {"type": "status", "state": "receiving"}
        yield self.partial
        yield {
            "type": "status",
            "state": "interrupted",
            "reason": "outbound_timeout",
            "partial": True,
        }


class CompleteStreamRouter:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def astream(self, capability: str, payload: dict[str, Any]):
        for chunk in self.chunks:
            yield chunk
        yield {"type": "status", "state": "completed"}

    async def astream_capability(
        self,
        capability: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any] | None = None,
    ):
        for chunk in self.chunks:
            yield chunk
        yield {"type": "status", "state": "completed"}


class OrderedAwaitable:
    def __init__(self, order: list[str], label: str) -> None:
        self.order = order
        self.label = label
        self.awaited = False

    def __await__(self):
        async def complete() -> None:
            self.awaited = True
            self.order.append(self.label)

        return complete().__await__()


def _pressure() -> ParsedPressure:
    return ParsedPressure(
        kind="drought",
        intensity=4,
        affected_tiles=[1],
        narrative="持续干旱使可用水源减少",
    )


@pytest.mark.asyncio
async def test_v1_displays_partial_stream_but_returns_complete_fallback() -> None:
    partial = "UNFINISHED_V1_NARRATIVE_" + "x" * 100
    shown: list[str] = []
    events: list[tuple[str, str, str]] = []
    builder = ReportBuilder(InterruptedStreamRouter(partial))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        stream_callback=lambda chunk: shown.append(chunk),
        event_callback=lambda *event: events.append(event),
    )

    assert shown == [partial]
    assert "**环境压力**" in report
    assert partial not in report
    assert [event for event in events if event[0] == "ai_stream_interrupted"] == [
        ("ai_stream_interrupted", "回合报告 interrupted", "AI")
    ]
    assert all(partial not in message and "outbound_timeout" not in message for _, message, _ in events)


@pytest.mark.asyncio
async def test_v2_displays_partial_stream_but_returns_complete_fallback() -> None:
    partial = "UNFINISHED_V2_NARRATIVE_" + "y" * 100
    shown: list[str] = []
    events: list[tuple[str, str, str]] = []
    builder = ReportBuilderV2(InterruptedStreamRouter(partial))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        turn_index=7,
        stream_callback=lambda chunk: shown.append(chunk),
        event_callback=lambda *event: events.append(event),
    )

    assert shown == [partial]
    assert "## 🕐 第 7 回合" in report
    assert partial not in report
    assert [event for event in events if event[0] == "ai_stream_interrupted"] == [
        ("ai_stream_interrupted", "第7回合报告 interrupted", "AI")
    ]
    assert all(partial not in message and "outbound_timeout" not in message for _, message, _ in events)


class EventEmittingReportBuilder:
    def __init__(self) -> None:
        self.event_callback: Callable[[str, str, str], None] | None = None

    async def build_turn_narrative_async(self, **kwargs: Any) -> str:
        self.event_callback = kwargs.get("event_callback")
        assert self.event_callback is not None
        self.event_callback(
            "ai_stream_interrupted",
            "回合报告 interrupted",
            "AI",
        )
        return "完整报告：" + "z" * 60


@pytest.mark.asyncio
async def test_turn_report_service_bridges_builder_events(monkeypatch, tmp_path) -> None:
    settings = SimpleNamespace(
        ui_config_path=tmp_path / "missing-settings.json",
        enable_turn_report_llm=True,
    )
    monkeypatch.setattr("app.services.analytics.turn_report.get_settings", lambda: settings)
    builder = EventEmittingReportBuilder()
    events: list[tuple[str, str, str]] = []

    def emit_event(event_type: str, message: str, category: str) -> None:
        events.append((event_type, message, category))
        if event_type == "ai_stream_interrupted":
            raise RuntimeError("downstream event consumer failed")

    service = TurnReportService(
        report_builder=builder,
        environment_repository=object(),
        trophic_service=object(),
        emit_event_fn=emit_event,
    )

    report = await service.build_report(
        turn_index=3,
        mortality_results=[],
        pressures=[],
        branching_events=[],
        all_species=[],
    )

    assert report.narrative.startswith("完整报告：")
    assert builder.event_callback is not None
    assert [event for event in events if event[0] == "ai_stream_interrupted"] == [
        ("ai_stream_interrupted", "回合报告 interrupted", "AI")
    ]


@pytest.mark.parametrize(
    ("length", "uses_fallback"),
    [(0, True), (5, True), (50, True), (51, False)],
)
@pytest.mark.asyncio
async def test_v2_only_accepts_complete_narratives_longer_than_50_chars(
    length: int,
    uses_fallback: bool,
) -> None:
    narrative = "n" * length
    builder = ReportBuilderV2(CompleteStreamRouter([narrative] if narrative else []))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        turn_index=8,
    )

    if uses_fallback:
        assert "## 🕐 第 8 回合" in report
        assert report != narrative
    else:
        assert report == narrative


@pytest.mark.asyncio
async def test_v2_awaits_heartbeat_and_stream_awaitables_in_order() -> None:
    chunks = [f"chunk-{index:02d}-data-" for index in range(1, 6)]
    order: list[str] = []
    heartbeat_futures: list[asyncio.Future[None]] = []
    stream_awaitables: list[OrderedAwaitable] = []

    def heartbeat_callback(count: int) -> asyncio.Future[None]:
        order.append(f"heartbeat-called:{count}")
        future = asyncio.get_running_loop().create_future()
        heartbeat_futures.append(future)

        def complete() -> None:
            order.append(f"heartbeat-awaited:{count}")
            future.set_result(None)

        asyncio.get_running_loop().call_soon(complete)
        return future

    def stream_callback(chunk: str) -> OrderedAwaitable:
        order.append(f"stream-called:{chunk}")
        awaitable = OrderedAwaitable(order, f"stream-awaited:{chunk}")
        stream_awaitables.append(awaitable)
        return awaitable

    builder = ReportBuilderV2(CompleteStreamRouter(chunks))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        turn_index=9,
        stream_callback=stream_callback,
        heartbeat_callback=heartbeat_callback,
    )

    assert report == "".join(chunks)
    assert heartbeat_futures and all(future.done() for future in heartbeat_futures)
    assert stream_awaitables and all(item.awaited for item in stream_awaitables)
    assert order[-4:] == [
        "heartbeat-called:5",
        "heartbeat-awaited:5",
        "stream-called:chunk-05-data-",
        "stream-awaited:chunk-05-data-",
    ]
