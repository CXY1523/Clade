from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger(f"{__package__}.speciation")


def generate_background_results(
    background_entries: list[dict],
    pressures: list[Any] | None,
    *,
    average_pressure: float,
    turn_index: int,
    generate_rule_based_fallback: Callable[..., dict],
) -> list[tuple[dict, dict]]:
    """Generate deterministic rule results for background species."""
    background_results: list[tuple[dict, dict]] = []

    # 构建环境压力字典（用于规则引擎）
    env_pressure_dict = {}
    if pressures:
        for p in pressures:
            if hasattr(p, "category") and hasattr(p, "intensity"):
                env_pressure_dict[p.category] = p.intensity

    for entry in background_entries:
        ctx = entry["ctx"]
        ai_content = generate_rule_based_fallback(
            parent=ctx["parent"],
            new_code=ctx["new_code"],
            survivors=ctx["population"],
            speciation_type=ctx["speciation_type"],
            average_pressure=average_pressure,
            environment_pressure=env_pressure_dict,
            turn_index=turn_index,
        )
        background_results.append((entry, ai_content))
        logger.debug(
            f"[规则分化] 背景物种 {ctx['parent'].common_name} -> {ai_content.get('common_name')} "
            f"({ai_content.get('_evolution_direction', '自然分化')})"
        )

    if background_results:
        logger.info(
            f"[规则分化] 完成 {len(background_results)} 个背景物种的规则生成"
        )

    return background_results


def partition_speciation_entries(
    entries: list[dict],
    deferred_requests: list[dict],
    *,
    max_deferred_requests: int,
    max_speciation_per_turn: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split prepared entries into background, active and deferred work."""
    background_entries: list[dict] = []
    ai_entries: list[dict] = []

    for entry in entries:
        parent = entry["ctx"]["parent"]
        if getattr(parent, "is_background", False):
            background_entries.append(entry)
        else:
            ai_entries.append(entry)

    pending = deferred_requests + ai_entries
    if len(pending) > max_deferred_requests:
        pending = pending[:max_deferred_requests]
    active_batch = pending[:max_speciation_per_turn]
    remaining_deferred = pending[max_speciation_per_turn:]

    return background_entries, active_batch, remaining_deferred
