from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger(f"{__package__}.speciation")


def build_offspring_ai_entry(
    *,
    species: Any, new_code: str, population: int,
    offspring_index: int, num_offspring: int, offspring_tiles: list,
    cluster_pressure_data: list[dict], clusters: list,
    tile_populations: dict, tile_mortality: dict,
    mortality_gradient: float, is_isolated: bool, death_rate: float,
    candidate_data: dict | None, average_pressure: float,
    pressure_summary: str, generations: float, speciation_type: str,
    map_changes: list, major_events: list, food_chain_summary: str,
    current_pressures: list | None, current_pressure_types: list,
    organ_catalog: list[dict], turn_index: int,
    infer_biological_domain: Callable[[Any], str],
    generate_tile_context: Callable[..., str], rules: Any,
    naming_hint_generator: Any,
    summarize_organs: Callable[[Any], str],
    summarize_map_changes: Callable[[list], str],
    summarize_major_events: Callable[[list], str],
    summarize_prey_species: Callable[[Any], str],
    summarize_dormant_genes: Callable[..., str],
    organ_evolution_service: Any,
) -> dict:
    """Build one offspring AI request entry from an existing plan."""
    safe_history = []
    if species.history_highlights:
        for event in species.history_highlights[-2:]:
            safe_history.append(
                event[:80] + "..." if len(event) > 80 else event
            )

    biological_domain = infer_biological_domain(species)
    assigned_tiles = offspring_tiles[offspring_index] if (
        offspring_index < len(offspring_tiles)
    ) else set()

    if cluster_pressure_data and offspring_index < len(cluster_pressure_data):
        region_data = cluster_pressure_data[offspring_index]
        region_mortality = region_data["avg_mortality"]
        region_pressure_level = region_data["pressure_level"]
        region_population = region_data["population"]
    else:
        if assigned_tiles and tile_mortality:
            region_mortality = sum(
                tile_mortality.get(t, 0.5)
                for t in assigned_tiles
            ) / len(assigned_tiles)
        else:
            region_mortality = death_rate

        if region_mortality > 0.5:
            region_pressure_level = "高压"
        elif region_mortality > 0.3:
            region_pressure_level = "中压"
        else:
            region_pressure_level = "低压"
        region_population = population

    cluster_environment = None
    tile_environment = candidate_data.get(
        "tile_environment"
    ) if candidate_data else None
    cluster_environments = candidate_data.get(
        "cluster_environments", []
    ) if candidate_data else []
    if cluster_environments and offspring_index < len(cluster_environments):
        cluster_environment = cluster_environments[offspring_index]

    tile_context = generate_tile_context(
        assigned_tiles,
        tile_populations,
        tile_mortality,
        mortality_gradient,
        is_isolated,
        tile_environment=tile_environment,
        cluster_environment=cluster_environment,
    )

    environment_pressure_dict = {
        "temperature": 0,
        "humidity": 0,
        "salinity": 0,
    }
    if current_pressures:
        for p in current_pressures:
            if hasattr(p, "modifiers"):
                environment_pressure_dict.update(p.modifiers)

    rule_constraints = rules.preprocess(
        parent_species=species,
        offspring_index=offspring_index + 1,
        total_offspring=num_offspring,
        environment_pressure=environment_pressure_dict,
        pressure_context=pressure_summary,
    )

    naming_seed = abs(hash(
        f"{new_code}-{species.lineage_code}-{offspring_index}"
    )) % 1_000_000_007
    naming_hint_generator.set_seed(naming_seed)
    naming_hints = naming_hint_generator.generate_compact_hint()

    ai_payload = {
        "parent_lineage": species.lineage_code,
        "latin_name": species.latin_name,
        "common_name": species.common_name,
        "habitat_type": species.habitat_type,
        "biological_domain": biological_domain,
        "current_organs_summary": summarize_organs(species),
        "environment_pressure": average_pressure,
        "pressure_summary": pressure_summary,
        "evolutionary_generations": int(generations),
        "traits": species.description,
        "history_highlights": "; ".join(safe_history) if safe_history else "无",
        "survivors": population,
        "speciation_type": speciation_type,
        "map_changes_summary": summarize_map_changes(
            map_changes
        ) if map_changes else "",
        "major_events_summary": summarize_major_events(
            major_events
        ) if major_events else "",
        "parent_trophic_level": species.trophic_level,
        "offspring_index": offspring_index + 1,
        "total_offspring": num_offspring,
        "food_chain_status": food_chain_summary,
        "tile_context": tile_context,
        "region_mortality": region_mortality,
        "region_pressure_level": region_pressure_level,
        "mortality_gradient": mortality_gradient,
        "num_isolation_regions": len(clusters) if clusters else 1,
        "is_geographic_isolation": (
            is_isolated and len(clusters) >= 2 if clusters else False
        ),
        "trait_budget_summary": rule_constraints["trait_budget_summary"],
        "organ_constraints_summary": rule_constraints[
            "organ_constraints_summary"],
        "evolution_direction": rule_constraints["evolution_direction"],
        "direction_description": rule_constraints["direction_description"],
        "suggested_increases": ", ".join(rule_constraints["suggested_increases"]),
        "suggested_decreases": ", ".join(rule_constraints["suggested_decreases"]),
        "habitat_options": ", ".join(rule_constraints["habitat_options"]),
        "trophic_range": rule_constraints["trophic_range"],
        "niche_exploration_strategy": rule_constraints.get(
            "niche_exploration_strategy", "保守继承型"),
        "niche_exploration_description": rule_constraints.get(
            "niche_exploration_description", ""),
        "niche_exploration_full": rule_constraints.get(
            "niche_exploration_full", ""),
        "target_diet_focus": rule_constraints.get(
            "target_diet_focus", "与父代相同"),
        "target_body_size_trend": rule_constraints.get(
            "target_body_size_trend", "similar"),
        "target_ecological_role": rule_constraints.get(
            "target_ecological_role", ""),
        "competition_with_parent": rule_constraints.get(
            "competition_with_parent", "direct"),
        "era_summary": rule_constraints.get("era_summary", ""),
        "era_single_cap": rule_constraints.get("era_single_cap", 15),
        "era_total_cap": rule_constraints.get("era_total_cap", 100),
        "diminishing_returns_context": rule_constraints.get(
            "diminishing_returns_context", ""),
        "breakthrough_opportunities": rule_constraints.get(
            "breakthrough_opportunities", ""),
        "habitat_specialization_bonus": rule_constraints.get(
            "habitat_specialization_bonus", ""),
        "strategy_recommendation": rule_constraints.get(
            "strategy_recommendation", ""),
        "budget_usage_percent": rule_constraints.get(
            "budget_usage_percent", 0.5),
        "remaining_budget": rule_constraints.get("remaining_budget", 50),
        "diet_type": species.diet_type or "omnivore",
        "prey_species_summary": summarize_prey_species(species),
        "gene_diversity_radius": getattr(
            species, "gene_diversity_radius", 0.35) or 0.35,
        "gene_stability": getattr(
            species, "gene_stability", 0.5) or 0.5,
        "explored_directions": len(
            getattr(species, "explored_directions", []) or []
        ),
        "dormant_genes_summary": summarize_dormant_genes(
            species,
            pressure_types=current_pressure_types,
            pressure_strength=average_pressure,
        ),
        "organ_key_catalog": "\n".join(
            [
                f"- {c['organ_key']} ({c['category']})："
                f"{c['default_name']}"
                for c in organ_catalog
            ]
        ),
        "mature_organs_context": (
            organ_evolution_service.build_mature_organs_context(species)
        ),
        "naming_hints": naming_hints,
    }

    return {
        "ctx": {
            "parent": species,
            "new_code": new_code,
            "population": population,
            "ai_payload_input": ai_payload,
            "speciation_type": speciation_type,
            "assigned_tiles": assigned_tiles,
            "average_pressure": average_pressure,
        },
        "payload": ai_payload,
        "request_turn": turn_index,
    }


def prepare_active_result(
    result: Any,
    entry: dict,
    *,
    turn_index: int,
    average_pressure: float,
    environment_pressure: dict,
    queue_deferred_request: Callable[[dict], Any],
    normalize_ai_content: Callable[[Any], dict],
    generate_rule_based_fallback: Callable[..., dict],
) -> tuple[dict, dict] | None:
    """Prepare one active AI result or preserve its existing skip decision."""
    request_turn = entry.get("request_turn", -1)
    if request_turn != turn_index:
        logger.warning(
            f"[分化跳过-过期] {entry.get('ctx', {}).get('new_code', '?')}: "
            f"请求来自回合 {request_turn}，当前回合 {turn_index}，丢弃"
        )
        return None

    ctx = entry["ctx"]
    retry_count = entry.get("_retry_count", 0)
    use_fallback = False

    if isinstance(result, Exception):
        logger.error(f"[分化AI异常] {result}")
        if retry_count >= 2:
            use_fallback = True
            logger.info(
                f"[分化] 重试{retry_count}次后AI仍失败，使用规则fallback"
            )
        else:
            queue_deferred_request(entry)
            return None

    ai_content = result
    if not use_fallback and not isinstance(ai_content, dict):
        logger.warning(
            f"[分化警告] AI返回的content不是dict类型: "
            f"{type(ai_content)}, 内容: {ai_content}"
        )
        if retry_count >= 2:
            use_fallback = True
        else:
            queue_deferred_request(entry)
            return None
    if not use_fallback:
        ai_content = normalize_ai_content(ai_content)

    required_fields = ["latin_name", "common_name", "description"]
    if not use_fallback and any(
        not ai_content.get(field) for field in required_fields
    ):
        logger.warning(
            "[分化警告] AI返回缺少必要字段: %s",
            {field: ai_content.get(field) for field in required_fields},
        )
        if retry_count >= 2:
            use_fallback = True
        else:
            queue_deferred_request(entry)
            return None

    if use_fallback:
        ai_content = generate_rule_based_fallback(
            parent=ctx["parent"],
            new_code=ctx["new_code"],
            survivors=ctx["population"],
            speciation_type=ctx["speciation_type"],
            average_pressure=average_pressure,
            environment_pressure=environment_pressure,
            turn_index=turn_index,
        )
        ai_content = normalize_ai_content(ai_content)

    logger.info(
        "[分化AI返回] latin_name: %s, common_name: %s, description长度: %s",
        ai_content.get("latin_name"),
        ai_content.get("common_name"),
        len(str(ai_content.get("description", ""))),
    )
    return ctx, ai_content


def materialize_active_result(
    ctx: dict,
    ai_content: dict,
    *,
    turn_index: int,
    average_pressure: float,
    validate_and_fix: Callable[..., dict],
    create_species: Callable[..., Any],
    turn_offspring_counts: dict,
    rule_fallback_species: list[tuple[Any, Any, str]],
    random_uniform: Callable[[float, float], float],
    inherit_habitat_distribution: Callable[..., Any],
    update_genetic_distances: Callable[..., Any],
    gene_library_service_owner: Any,
    genus_repository: Any,
    process_ai_activated_genes: Callable[..., int],
    process_ai_new_dormant_genes: Callable[..., int],
    try_speciation_breakthrough: Callable[..., Any],
    check_and_trigger_plant_milestones: Callable[..., Any],
    evaluate_new_species_viability: Callable[..., dict],
    upsert_species: Callable[[Any], Any],
    log_lineage_event: Callable[[Any], Any],
    lineage_event_factory: Callable[..., Any],
    branching_event_factory: Callable[..., Any],
    utcnow: Callable[[], Any],
) -> Any:
    """Persist one prepared active result and return its branching event."""
    ai_content = validate_and_fix(
        ai_content,
        ctx["parent"],
        preprocess_result=None,
    )

    new_species = create_species(
        parent=ctx["parent"],
        new_code=ctx["new_code"],
        survivors=ctx["population"],
        turn_index=turn_index,
        ai_payload=ai_content,
        average_pressure=average_pressure,
        speciation_type=ctx["speciation_type"],
    )
    logger.info(
        f"[分化] 新物种 {new_species.common_name} "
        f"created_turn={new_species.created_turn} "
        f"(传入的turn_index={turn_index})"
    )
    new_species = upsert_species(new_species)

    parent_code = ctx["parent"].lineage_code
    try:
        turn_offspring_counts[parent_code] += 1
        ctx.turn_offspring_counts = turn_offspring_counts  # type: ignore[attr-defined]
    except Exception:
        pass
    logger.info(
        f"[分化] upsert后 {new_species.common_name} "
        f"created_turn={new_species.created_turn}"
    )
    parent_code = ctx["parent"].lineage_code
    try:
        turn_offspring_counts[parent_code] += 1
        ctx.turn_offspring_counts = turn_offspring_counts  # type: ignore[attr-defined]
    except Exception:
        pass

    if ai_content.get("_is_rule_fallback"):
        rule_fallback_species.append(
            (new_species, ctx["parent"], ctx["speciation_type"])
        )

    assigned_tiles = ctx.get("assigned_tiles", set())
    reproduction_bonus = random_uniform(0.30, 0.50)
    inherit_habitat_distribution(
        parent=ctx["parent"],
        child=new_species,
        turn_index=turn_index,
        assigned_tiles=assigned_tiles,
        reproduction_bonus=reproduction_bonus,
    )

    update_genetic_distances(new_species, ctx["parent"], turn_index)

    if ai_content.get("genetic_discoveries") and new_species.genus_code:
        gene_library_service_owner.gene_library_service.record_discovery(
            genus_code=new_species.genus_code,
            discoveries=ai_content["genetic_discoveries"],
            discoverer_code=new_species.lineage_code,
            turn=turn_index,
        )

    genus = (
        genus_repository.get_by_code(new_species.genus_code)
        if new_species.genus_code
        else None
    )
    gene_library_service_owner.gene_library_service.inherit_dormant_genes(
        ctx["parent"],
        new_species,
        genus,
    )
    upsert_species(new_species)

    ai_activated_genes = (
        ai_content.get("activated_genes", []) if ai_content else []
    )
    if ai_activated_genes:
        activated_count = process_ai_activated_genes(
            new_species,
            ai_activated_genes,
            turn_index,
        )
        if activated_count > 0:
            upsert_species(new_species)
            logger.info(
                f"[AI基因激活] {new_species.common_name} "
                f"激活了 {activated_count} 个AI指定的基因"
            )

    ai_new_genes = (
        ai_content.get("new_dormant_genes") if ai_content else None
    )
    if ai_new_genes:
        added_count = process_ai_new_dormant_genes(
            new_species,
            ai_new_genes,
            turn_index,
        )
        if added_count > 0:
            upsert_species(new_species)
            logger.info(
                f"[LLM新基因] {new_species.common_name} "
                f"获得了 {added_count} 个LLM生成的新休眠基因"
            )

    breakthrough_result = try_speciation_breakthrough(
        new_species,
        turn_index,
    )
    if breakthrough_result:
        upsert_species(new_species)
        logger.info(
            f"[分化突破] {new_species.common_name} 在分化中激活休眠基因: "
            f"{breakthrough_result}"
        )

    milestone_result = check_and_trigger_plant_milestones(
        new_species,
        turn_index,
    )
    if milestone_result:
        upsert_species(new_species)
        logger.info(
            f"[植物里程碑] {new_species.common_name} 触发里程碑: "
            f"{milestone_result.get('milestone_name', 'unknown')}"
        )

    if gene_library_service_owner._tensor_state is not None:
        species_map = {}
        if hasattr(gene_library_service_owner._tensor_state, "species_map"):
            species_map = (
                gene_library_service_owner._tensor_state.species_map
            )

        viability = evaluate_new_species_viability(
            new_species,
            gene_library_service_owner._tensor_state,
            species_map,
            turn_index,
        )

        if viability["recommendation"] == "extinct":
            new_species.morphology_stats["viability_risk"] = "critical"
            upsert_species(new_species)
            logger.warning(
                f"[新种筛选] {new_species.common_name} 被标记为高灭绝风险: "
                f"适宜度={viability['avg_suitability']:.3f}, "
                f"分布={viability['tile_count']}格"
            )
        elif viability["recommendation"] == "penalize":
            new_species.morphology_stats["viability_risk"] = "low"
            upsert_species(new_species)

    log_lineage_event(
        lineage_event_factory(
            lineage_code=ctx["new_code"],
            event_type="speciation",
            payload={
                "parent": ctx["parent"].lineage_code,
                "turn": turn_index,
            },
        )
    )

    event_desc = ai_content.get("event_description") if ai_content else None
    if not event_desc:
        event_desc = (
            f"{ctx['parent'].common_name}在压力{average_pressure:.1f}"
            f"条件下分化出{ctx['new_code']}"
        )

    reason_text = ai_content.get("reason") or ai_content.get(
        "speciation_reason"
    )
    if not reason_text:
        if ctx["speciation_type"] == "地理隔离":
            reason_text = (
                f"{ctx['parent'].common_name}因地形剧变导致种群地理隔离，"
                f"各隔离群体独立演化产生生殖隔离"
            )
        elif ctx["speciation_type"] == "极端环境特化":
            reason_text = (
                f"{ctx['parent'].common_name}在极端环境压力下，"
                f"部分种群演化出特化适应能力，与原种群形成生态分离"
            )
        elif ctx["speciation_type"] == "协同演化":
            reason_text = (
                f"{ctx['parent'].common_name}与竞争物种的生态位重叠"
                f"导致竞争排斥，促使种群分化到不同资源梯度"
            )
        else:
            reason_text = (
                f"{ctx['parent'].common_name}种群在演化压力下"
                f"发生生态位分化"
            )

    return branching_event_factory(
        parent_lineage=ctx["parent"].lineage_code,
        new_lineage=ctx["new_code"],
        description=event_desc,
        timestamp=utcnow(),
        reason=reason_text,
    )


def materialize_active_results(
    results: list,
    active_batch: list[dict],
    *,
    result_events: list[Any],
    preparation_kwargs: dict[str, Any],
    materialization_kwargs: dict[str, Any],
) -> None:
    """Prepare and materialize active results in stable zip order."""
    for result, entry in zip(results, active_batch):
        prepared = prepare_active_result(
            result,
            entry,
            **preparation_kwargs,
        )
        if prepared is None:
            continue
        ctx, ai_content = prepared
        result_events.append(
            materialize_active_result(
                ctx,
                ai_content,
                **materialization_kwargs,
            )
        )


def materialize_background_result(
    entry: dict,
    ai_content: dict,
    *,
    turn_index: int,
    average_pressure: float,
    validate_and_fix: Callable[..., dict],
    create_species: Callable[..., Any],
    rule_fallback_species: list[tuple[Any, Any, str]],
    random_uniform: Callable[[float, float], float],
    inherit_habitat_distribution: Callable[..., Any],
    gene_library_service_owner: Any,
    genus_repository: Any,
    try_speciation_breakthrough: Callable[..., Any],
    upsert_species: Callable[[Any], Any],
    log_lineage_event: Callable[[Any], Any],
    lineage_event_factory: Callable[..., Any],
    branching_event_factory: Callable[..., Any],
    utcnow: Callable[[], Any],
) -> Any:
    """Create and persist one background species result and its event."""
    ctx = entry["ctx"]

    logger.info(
        f"[规则分化结果] 背景物种: {ai_content.get('common_name')}, description长度: {len(str(ai_content.get('description', '')))}"
    )

    # 【新增】规则引擎后验证：验证并修正输出
    ai_content = validate_and_fix(
        ai_content,
        ctx["parent"],
        preprocess_result=None,
    )

    new_species = create_species(
        parent=ctx["parent"],
        new_code=ctx["new_code"],
        survivors=ctx["population"],
        turn_index=turn_index,
        ai_payload=ai_content,
        average_pressure=average_pressure,
        speciation_type=ctx["speciation_type"],
    )

    # 背景物种子代也标记为背景
    new_species.is_background = True

    logger.info(
        f"[规则分化] 新背景物种 {new_species.common_name} created_turn={new_species.created_turn}"
    )
    new_species = upsert_species(new_species)

    # 将背景物种加入增强队列（用于模板描述和向量遗传）
    rule_fallback_species.append(
        (new_species, ctx["parent"], ctx["speciation_type"])
    )

    # 处理分配地块
    assigned_tiles = ctx.get("assigned_tiles", set())

    # 【v3.1】背景物种也给予繁殖补偿
    reproduction_bonus = random_uniform(0.30, 0.50)

    inherit_habitat_distribution(
        parent=ctx["parent"],
        child=new_species,
        turn_index=turn_index,
        assigned_tiles=assigned_tiles,
        reproduction_bonus=reproduction_bonus,
    )

    # 【修复】即使没有 genus 也调用继承方法（处理新突变和额外基因）
    if (
        hasattr(gene_library_service_owner, "gene_library_service")
        and gene_library_service_owner.gene_library_service
    ):
        genus = (
            genus_repository.get_by_code(new_species.genus_code)
            if new_species.genus_code
            else None
        )
        gene_library_service_owner.gene_library_service.inherit_dormant_genes(
            ctx["parent"],
            new_species,
            genus,
        )
        upsert_species(new_species)

    # 【分化突破】背景物种仅激活已有休眠基因（不生成新基因）
    breakthrough_result = try_speciation_breakthrough(
        new_species,
        turn_index,
    )
    if breakthrough_result:
        upsert_species(new_species)
        logger.info(
            f"[分化突破-背景] {new_species.common_name} 在分化中激活休眠基因: "
            f"{breakthrough_result}"
        )

    log_lineage_event(
        lineage_event_factory(
            lineage_code=ctx["new_code"],
            event_type="speciation",
            payload={
                "parent": ctx["parent"].lineage_code,
                "turn": turn_index,
            },
        )
    )

    event_desc = (
        f"{ctx['parent'].common_name}在压力{average_pressure:.1f}条件下"
        f"分化出{ctx['new_code']}（背景物种）"
    )
    reason_text = (
        f"{ctx['parent'].common_name}种群在演化压力下发生生态位分化"
    )

    return branching_event_factory(
        parent_lineage=ctx["parent"].lineage_code,
        new_lineage=ctx["new_code"],
        description=event_desc,
        timestamp=utcnow(),
        reason=reason_text,
    )


def materialize_background_results(
    background_results: list[tuple[dict, dict]],
    *,
    result_events: list[Any],
    **kwargs: Any,
) -> None:
    """Materialize background results in their existing stable order."""
    for entry, ai_content in background_results:
        result_events.append(
            materialize_background_result(entry, ai_content, **kwargs)
        )


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
