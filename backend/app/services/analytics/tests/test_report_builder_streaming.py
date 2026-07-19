from __future__ import annotations

from typing import Any

import pytest

from app.simulation.environment import ParsedPressure
from app.services.analytics.report_builder import ReportBuilder
from app.services.analytics.report_builder_v2 import ReportBuilderV2


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
    builder = ReportBuilder(InterruptedStreamRouter(partial))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        stream_callback=lambda chunk: shown.append(chunk),
    )

    assert shown == [partial]
    assert "**环境压力**" in report
    assert partial not in report


@pytest.mark.asyncio
async def test_v2_displays_partial_stream_but_returns_complete_fallback() -> None:
    partial = "UNFINISHED_V2_NARRATIVE_" + "y" * 100
    shown: list[str] = []
    builder = ReportBuilderV2(InterruptedStreamRouter(partial))

    report = await builder.build_turn_narrative_async(
        species=[],
        pressures=[_pressure()],
        turn_index=7,
        stream_callback=lambda chunk: shown.append(chunk),
    )

    assert shown == [partial]
    assert "## 🕐 第 7 回合" in report
    assert partial not in report
