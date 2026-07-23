from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger(f"{__package__}.speciation")


async def enhance_rule_fallback_descriptions(
    rule_fallback_species: list[tuple[Any, Any, str]],
    *,
    description_enhancer: Any,
    upsert_species: Callable[[Any], Any],
) -> None:
    """Enhance queued rule-generated species and always clear attempted work."""
    if rule_fallback_species:
        logger.info(
            f"[描述增强] 开始处理 {len(rule_fallback_species)} 个规则生成物种的描述增强"
        )
        try:
            # 将物种加入增强队列
            for species, parent, speciation_type in rule_fallback_species:
                description_enhancer.queue_for_enhancement(
                    species=species,
                    parent=parent,
                    speciation_type=speciation_type,
                    is_hybrid=False,
                )

            # 批量处理增强队列
            enhanced_list = await description_enhancer.process_queue_async(
                max_items=20,  # 每回合最多处理20个
                timeout_per_item=25.0,
            )

            # 保存增强后的物种描述
            for enhanced_species in enhanced_list:
                upsert_species(enhanced_species)

            logger.info(
                f"[描述增强] 完成 {len(enhanced_list)}/{len(rule_fallback_species)} 个物种描述增强"
            )
        except Exception as e:
            logger.error(f"[描述增强] 处理失败: {e}")
        finally:
            rule_fallback_species.clear()


async def execute_active_ai_batches(
    active_batch: list[dict],
    *,
    average_pressure: float,
    pressure_summary: str,
    map_changes: list,
    major_events: list,
    turn_index: int,
    stream_callback: Any,
    build_batch_payload: Callable[..., dict],
    call_batch_ai: Callable[..., Any],
    parse_batch_results: Callable[..., list],
    staggered_gather: Callable[..., Any],
) -> list:
    """Execute fixed-size AI batches and flatten their matched results."""
    results = []
    if active_batch:
        # 【优化】小批次 + 高并发策略
        # 每批 2 个物种，降低单次延迟
        # 同时 20 个批次并行，提高整体吞吐量
        batch_size = 2

        # 分割成多个批次
        batches = []
        for batch_start in range(0, len(active_batch), batch_size):
            batch_entries = active_batch[batch_start:batch_start + batch_size]
            batches.append(batch_entries)

        logger.info(
            f"[分化] 共 {len(batches)} 个AI批次（每批≤{batch_size}个），开始高并发执行"
        )

        async def process_batch(batch_entries: list) -> list:
            """处理单个批次"""
            batch_payload = build_batch_payload(
                batch_entries,
                average_pressure,
                pressure_summary,
                map_changes,
                major_events,
                turn_index,
            )
            # 【混合模式】传入entries用于判断是否为植物批次
            batch_results = await call_batch_ai(
                batch_payload, stream_callback, batch_entries
            )
            return parse_batch_results(batch_results, batch_entries)

        # 【优化】小批次 + 高并发：间隔更短，并发更高
        coroutines = [process_batch(batch) for batch in batches]
        batch_results_list = await staggered_gather(
            coroutines,
            interval=1.5,  # 调整批次启动间隔
            max_concurrent=20,  # 提升并发批次数
            task_name="分化批次",
            event_callback=stream_callback,  # 【新增】传递心跳回调
        )

        # 合并所有批次的结果
        for batch_idx, batch_result in enumerate(batch_results_list):
            if isinstance(batch_result, Exception):
                logger.error(
                    f"[分化] 批次 {batch_idx + 1} 失败: {batch_result}"
                )
                results.extend([batch_result] * len(batches[batch_idx]))
            else:
                success_count = len(
                    [r for r in batch_result if not isinstance(r, Exception)]
                )
                logger.info(
                    f"[分化] 批次 {batch_idx + 1} 完成，成功解析 {success_count} 个结果"
                )
                results.extend(batch_result)

    return results


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
