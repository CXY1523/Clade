from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ...models.species import Species

logger = logging.getLogger(f"{__package__}.speciation")

SuitabilityCalculator = Callable[[Species, Any], float]


def calculate_initial_habitat_for_child(
    child: Species,
    parent: Species,
    turn_index: int,
    assigned_tiles: set[int] | None = None,
    suitability_calculator: SuitabilityCalculator | None = None,
) -> None:
    """为没有栖息地的子代计算初始栖息地分布

    【核心改进】现在支持基于地块的分化：
    - 如果指定了 assigned_tiles，只在这些地块中选择
    - 如果未指定，则在所有合适地块中选择

    Args:
        child: 子代物种
        parent: 父代物种（用于参考）
        turn_index: 当前回合
        assigned_tiles: 分配给该子代的地块集合（可选）
        suitability_calculator: 现有服务的适宜度计算入口
    """
    from ...models.environment import HabitatPopulation
    from ...repositories.environment_repository import environment_repository

    if suitability_calculator is None:
        suitability_calculator = calculate_suitability_for_species

    logger.info(f"[栖息地计算] 为 {child.common_name} 计算初始栖息地")

    # 1. 获取所有地块
    all_tiles = environment_repository.list_tiles()
    if not all_tiles:
        logger.error(
            f"[栖息地计算] 没有可用地块，无法为 {child.common_name} 计算栖息地"
        )
        return

    # 【核心改进】如果指定了分配地块，只在这些地块中计算
    if assigned_tiles:
        all_tiles = [t for t in all_tiles if t.id in assigned_tiles]
        if not all_tiles:
            logger.warning(
                f"[栖息地计算] {child.common_name} 分配的地块在数据库中不存在，"
                f"使用全部地块"
            )
            all_tiles = environment_repository.list_tiles()

    # 2. 根据栖息地类型筛选地块
    habitat_type = getattr(child, "habitat_type", "terrestrial")
    suitable_tiles = []

    for tile in all_tiles:
        biome = tile.biome.lower()
        is_suitable = False

        if habitat_type == "marine" and ("浅海" in biome or "中层" in biome):
            is_suitable = True
        elif habitat_type == "deep_sea" and "深海" in biome:
            is_suitable = True
        elif habitat_type == "coastal" and ("海岸" in biome or "浅海" in biome):
            is_suitable = True
        elif habitat_type == "freshwater" and getattr(tile, "is_lake", False):
            is_suitable = True
        elif habitat_type == "terrestrial" and "海" not in biome:
            is_suitable = True
        elif habitat_type == "amphibious" and (
            "海岸" in biome or ("平原" in biome and tile.humidity > 0.4)
        ):
            is_suitable = True
        elif habitat_type == "aerial" and "海" not in biome and "山" not in biome:
            is_suitable = True

        if is_suitable:
            suitable_tiles.append(tile)

    if not suitable_tiles:
        logger.warning(
            f"[栖息地计算] {child.common_name} ({habitat_type}) 没有合适的地块"
        )
        # 回退：使用分配的地块或前10个地块
        suitable_tiles = all_tiles[:10] if all_tiles else []

    # 3. 计算适宜度
    tile_suitability = []
    for tile in suitable_tiles:
        suitability = suitability_calculator(child, tile)
        if suitability > 0.1:  # 只保留适宜度>0.1的地块
            tile_suitability.append((tile, suitability))

    if not tile_suitability:
        logger.warning(
            f"[栖息地计算] {child.common_name} 没有适宜度>0.1的地块，使用前10个"
        )
        tile_suitability = [(tile, 0.5) for tile in suitable_tiles[:10]]

    # 4. 选择top 10地块（如果有分配限制，可能更少）
    tile_suitability.sort(key=lambda x: x[1], reverse=True)
    max_tiles = min(10, len(tile_suitability))
    top_tiles = tile_suitability[:max_tiles]

    # 5. 归一化适宜度（总和=1.0）
    total_suitability = sum(s for _, s in top_tiles)
    if total_suitability == 0:
        total_suitability = 1.0

    # 6. 创建栖息地记录
    child_habitats = []
    for tile, raw_suitability in top_tiles:
        normalized_suitability = raw_suitability / total_suitability
        child_habitats.append(
            HabitatPopulation(
                tile_id=tile.id,
                species_id=child.id,
                population=0,
                suitability=normalized_suitability,
                turn_index=turn_index,
            )
        )

    if child_habitats:
        environment_repository.write_habitats(child_habitats)
        if assigned_tiles:
            logger.info(
                f"[基于地块分化] {child.common_name} 在分配区域内计算得到 "
                f"{len(child_habitats)} 个栖息地"
            )
        else:
            logger.info(
                f"[栖息地计算] {child.common_name} 计算得到 "
                f"{len(child_habitats)} 个栖息地"
            )


def calculate_suitability_for_species(species: Species, tile: Any) -> float:
    """计算物种对地块的适宜度 (v2.0 收紧版)

    【v2.0 改进】
    - 温度范围：5-10°C（极窄）
    - 温度惩罚：0.4/度
    - 湿度惩罚：4.0系数
    - 资源门槛：900
    """
    # 温度适应性 (v2.0 收紧)
    heat_res = species.abstract_traits.get("耐热性", 5)
    cold_res = species.abstract_traits.get("耐寒性", 5)

    # 最优温度和容忍范围
    optimal_temp = 15.0 + (heat_res - cold_res) * 2.0
    tolerance_range = (cold_res + heat_res) * 1.0  # coef=1.0，5-10°C范围

    min_temp = optimal_temp - tolerance_range / 2
    max_temp = optimal_temp + tolerance_range / 2

    if min_temp <= tile.temperature <= max_temp:
        temp_score = 1.0
    else:
        diff = min(
            abs(tile.temperature - min_temp),
            abs(tile.temperature - max_temp),
        )
        # 每度扣0.4，超过2.5度归零
        temp_score = max(0.0, 1.0 - diff * 0.4)

    if temp_score == 0.0:
        return 0.0

    # 湿度适应性 (v2.0 收紧)
    drought_pref = species.abstract_traits.get("耐旱性", 5)
    best_humidity = 1.0 - (drought_pref * 0.08)
    hum_diff = abs(tile.humidity - best_humidity)
    humidity_score = max(0.0, 1.0 - hum_diff * 4.0)  # 系数从1.0增加到4.0

    # 资源可用性 (v2.0 收紧)
    resource_score = min(1.0, tile.resources / 900.0)  # 门槛从500增加到900

    # 综合评分
    return max(
        0.0,
        temp_score * 0.35 + humidity_score * 0.25 + resource_score * 0.40,
    )
