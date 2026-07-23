from __future__ import annotations

import logging
from typing import Any, Callable


logger = logging.getLogger(f"{__package__}.speciation")


def build_batch_payload(
    entries: list[dict],
    average_pressure: float,
    pressure_summary: str,
    map_changes: list,
    major_events: list,
    turn_index: int,
    *,
    is_plant: Callable[[Any], bool],
    get_stage_name: Callable[[int], str],
    get_milestone_hints: Callable[[Any], str],
    format_competition_context: Callable[[Any, list[Any]], str],
    get_evolution_hint: Callable[[str | None], dict | None],
    naming_hint_generator: Any,
    summarize_map_changes: Callable[[list], str],
    summarize_major_events: Callable[[list], str],
    get_tradeoff_ratio: Callable[[], float],
) -> dict:
    """构建批量分化请求的 payload。"""
    # 构建物种列表文本
    species_list_parts = []
    for idx, entry in enumerate(entries):
        payload = entry["payload"]
        ctx = entry["ctx"]

        # 获取生物类群和器官摘要（可能在单独调用时已添加）
        biological_domain = payload.get("biological_domain", "protist")
        organs_summary = payload.get(
            "current_organs_summary", "无已记录的器官系统"
        )

        # 【关键】获取规则引擎约束信息
        organ_constraints = payload.get(
            "organ_constraints_summary", "无器官约束"
        )
        trait_budget = payload.get(
            "trait_budget_summary", "增加上限: +3.0, 减少下限: -1.5"
        )
        trophic_range = payload.get("trophic_range", "1.5-2.5")
        parent_trophic = payload.get("parent_trophic_level", 2.0)

        # 【新增】获取地块级信息
        tile_context = payload.get("tile_context", "未知区域")
        region_mortality = payload.get("region_mortality", 0.5)
        region_pressure_level = payload.get("region_pressure_level", "中压")
        mortality_gradient = payload.get("mortality_gradient", 0.0)
        num_isolation_regions = payload.get("num_isolation_regions", 1)
        is_geographic_isolation = payload.get("is_geographic_isolation", False)

        # 【植物演化】为植物物种添加专有上下文
        parent_species = ctx["parent"]
        parent_is_plant = is_plant(parent_species)
        plant_context = ""

        if parent_is_plant:
            # 获取植物演化阶段信息
            life_form_stage = getattr(parent_species, "life_form_stage", 0)
            growth_form = getattr(parent_species, "growth_form", "aquatic")
            stage_name = get_stage_name(life_form_stage)

            # 获取里程碑提示
            milestone_hints = get_milestone_hints(parent_species)

            # 获取植物特质摘要
            traits = parent_species.abstract_traits or {}
            plant_trait_summary = ", ".join(
                [
                    f"{k}={v:.1f}"
                    for k, v in traits.items()
                    if k
                    in [
                        "光合效率",
                        "根系发达度",
                        "保水能力",
                        "木质化程度",
                        "种子化程度",
                        "多细胞程度",
                    ]
                ]
            )

            # 【新增】获取竞争上下文
            # 从entries中收集所有父代物种作为species_list
            all_parent_species = [e["ctx"]["parent"] for e in entries]
            competition_context = format_competition_context(
                parent_species, all_parent_species
            )

            plant_context = f"""
- 【🌱植物演化信息】:
  - 当前阶段: {life_form_stage} ({stage_name})
  - 生长形式: {growth_form}
  - 植物特质: {plant_trait_summary or '无'}
  - 里程碑提示:
{milestone_hints}
- {competition_context}
- 【植物阶段约束⚠️】:
  - 阶段只能升级1级（{life_form_stage} → {life_form_stage + 1}）
  - 登陆条件(阶段2→3): 保水能力>=5.0, 耐旱性>=4.0
  - 成为树木条件: 木质化程度>=7.0, 阶段>=5"""

        # 【新增】获取 AI 演化提示（来自生态智能体）
        parent_code = payload.get("parent_lineage")
        evolution_hint = get_evolution_hint(parent_code)
        ai_evolution_context = ""
        if evolution_hint:
            ai_directions = evolution_hint.get("ai_directions", [])
            if ai_directions:
                ai_evolution_context = f"""
- 【🧠AI演化建议】（来自生态智能体评估）:
  - 建议方向: {', '.join(ai_directions[:5])}
  - 请参考这些方向设计子代的特质变化！"""
                logger.debug(
                    f"[分化] {parent_code} 使用AI演化提示: {ai_directions}"
                )

        species_info = f"""
【物种 {idx + 1}】{'🌱植物' if parent_is_plant else '🦎动物'}
- request_id: {idx}
- 父系编码: {payload.get('parent_lineage')}
- 学名: {payload.get('latin_name')}
- 俗名: {payload.get('common_name')}
- 新编码: {ctx['new_code']}
- 栖息地: {payload.get('habitat_type')}
- 生物类群: {biological_domain}
- 营养级: T{parent_trophic:.1f}（允许范围：{trophic_range}）
- 描述: {payload.get('traits', '')[:200]}
- 现有器官: {organs_summary}
- 幸存者: {payload.get('survivors', 0):,}
- 分化类型: {payload.get('speciation_type')}
- 子代编号: 第{payload.get('offspring_index', 1)}个（共{payload.get('total_offspring', 1)}个）
- 【属性预算】: {trait_budget}
- 【器官约束⚠️必须遵守current_stage】:
{organ_constraints}{plant_context}{ai_evolution_context}
- 【地块背景】: {tile_context[:150]}
- 区域死亡率: {region_mortality:.1%}（{region_pressure_level}）
- 死亡率梯度: {mortality_gradient:.1%}
- 隔离区域数: {num_isolation_regions}
- 地理隔离: {'是' if is_geographic_isolation else '否'}"""
        species_list_parts.append(species_info)

    species_list = "\n".join(species_list_parts)

    # 【修复】不再转义 species_list 中的花括号
    # prompt format(**payload) 不会递归解析 value 中的花括号
    species_list_escaped = species_list

    from ...simulation.constants import get_time_config

    time_config = get_time_config(max(turn_index, 0))
    time_context = (
        "\n=== ⏳ 时间尺度上下文 (Chronos Flow) ===\n"
        f"当前地质年代：{time_config['era_name']}\n"
        f"时间流逝速度：{time_config['years_per_turn']:,} 年/回合\n"
        f"演化指导原则：{time_config.get('evolution_guide', 'Standard')}\n"
    )

    # 【命名提示】为批量分化生成随机命名参考
    batch_naming_seed = (
        abs(hash(f"batch-{turn_index}-{len(entries)}")) % 1_000_000_007
    )
    naming_hint_generator.set_seed(batch_naming_seed)
    naming_hints = naming_hint_generator.generate_naming_prompt(
        samples_per_category=2
    )

    payload_data = {
        "average_pressure": average_pressure,
        "pressure_summary": pressure_summary,
        "map_changes_summary": (
            summarize_map_changes(map_changes)
            if map_changes
            else "无显著地形变化"
        ),
        "major_events_summary": (
            summarize_major_events(major_events) if major_events else "无重大事件"
        ),
        # 【修复】同时提供 major_events 字段
        "major_events": (
            summarize_major_events(major_events) if major_events else "无重大事件"
        ),
        "species_list": species_list_escaped,
        "batch_size": len(entries),
        "time_context": time_context,
        "naming_hints": naming_hints,
    }
    first_payload = entries[0]["payload"] if entries else {}
    payload_data.update(
        {
            # 张量系统预先计算的预算约束，LLM 只需输出增益
            "max_increase": 3.0,
            "single_max": 2.0,
            "era_caps": first_payload.get("trait_budget_summary", "依时代上限"),
            "tradeoff_ratio": get_tradeoff_ratio(),
            # 为精简版prompt提供摘要
            "parent_summary": species_list_escaped,
            "trigger_context": pressure_summary,
        }
    )
    logger.debug(f"[分化批量] Payload keys: {list(payload_data.keys())}")
    return payload_data


def normalize_ai_content(ai_content: Any) -> Any:
    """将新格式的AI输出规范化为内部通用字段。

    - 如果提供了 innovations/gains，则汇总为 trait_changes
    - 缺少 key_innovations 时从 innovations 中提取
    """
    if not isinstance(ai_content, dict):
        return ai_content

    if not ai_content.get("trait_changes"):
        aggregated: dict[str, float] = {}
        key_innovations = list(ai_content.get("key_innovations") or [])
        innovations = ai_content.get("innovations") or []

        if isinstance(innovations, list):
            for inv in innovations:
                if not isinstance(inv, dict):
                    continue
                name = inv.get("name")
                if name and name not in key_innovations:
                    key_innovations.append(name)

                gains = inv.get("gains") or {}
                if isinstance(gains, dict):
                    for trait, delta in gains.items():
                        try:
                            val = float(str(delta).replace("+", ""))
                        except (ValueError, TypeError):
                            continue
                        aggregated[trait] = aggregated.get(trait, 0.0) + val

        if aggregated:
            ai_content["trait_changes"] = aggregated
        if key_innovations and not ai_content.get("key_innovations"):
            ai_content["key_innovations"] = key_innovations

    return ai_content
