from __future__ import annotations

import logging
from typing import Any, Callable

from ...models.species import Species

logger = logging.getLogger(f"{__package__}.speciation")


def infer_complexity_by_rules(species: Species) -> int:
    """基于规则推断复杂度等级（降级方案）"""
    description = (species.description or "").lower()
    common_name = (species.common_name or "").lower()
    organs = species.organs or {}
    body_length = species.morphology_stats.get("body_length_cm", 0.01)

    # 关键词映射
    level_keywords = {
        0: ["细菌", "杆菌", "球菌", "古菌", "原核", "bacteria", "archaea", "芽孢"],
        1: ["原生", "单细胞", "鞭毛虫", "纤毛虫", "变形虫", "眼虫", "草履虫", "protist", "amoeba"],
        2: ["团藻", "海绵", "水母", "珊瑚", "群体", "殖民", "简单多细胞", "colony"],
        3: ["扁形虫", "涡虫", "线虫", "环节", "蚯蚓", "水蛭", "组织分化"],
        4: ["节肢", "软体", "昆虫", "甲壳", "蜘蛛", "鱼", "章鱼", "蜗牛", "器官系统"],
        5: ["两栖", "爬行", "鸟", "哺乳", "脊椎", "蛙", "蜥蜴", "蛇", "恐龙", "鲸", "猫", "狗", "人"],
    }

    # 关键词匹配
    for level in range(5, -1, -1):  # 从高到低匹配
        if any(kw in description or kw in common_name for kw in level_keywords[level]):
            # 原核生物额外验证：不能有真核特征
            if level == 0:
                eukaryote_features = ["叶绿体", "线粒体", "细胞核", "内质网", "高尔基体"]
                if any(kw in description for kw in eukaryote_features):
                    continue
            return level

    # 基于器官复杂度推断
    organ_count = len([o for o in organs.values() if o.get("is_active", True)])
    if organ_count >= 5:
        return 4  # 器官级
    elif organ_count >= 3:
        return 3  # 组织级
    elif organ_count >= 1:
        return 2 if body_length > 0.1 else 1

    # 基于体型推断
    if body_length < 0.001:  # < 10微米
        return 0  # 原核生物
    elif body_length < 0.1:  # < 1毫米
        return 1  # 简单真核
    elif body_length < 1.0:  # < 1厘米
        return 2  # 简单多细胞
    elif body_length < 10.0:  # < 10厘米
        return 3  # 组织级
    else:
        return 4  # 器官级或更高


def get_complexity_constraints(complexity_level: str) -> dict:
    """获取复杂度等级的基础约束

    设计理念：允许自由演化，只限制"跳跃式"发展
    - 不限制能发展什么结构（让环境压力自然筛选）
    - 只限制原核/真核的基本分界（这是生物学硬约束）
    - 通过阶段系统保证渐进式发展
    """
    level = int(complexity_level.split("_")[1]) if "_" in complexity_level else 1

    # 极简约束：只区分原核/真核的根本差异
    if level == 0:  # 原核生物
        return {
            # 原核生物的唯一硬约束：不能有真核细胞器
            # （因为这需要内共生事件，不是渐进演化能达到的）
            "origin_type": "prokaryote",
            "hard_forbidden": ["真核鞭毛", "纤毛", "线粒体", "叶绿体", "细胞核", "内质网", "高尔基体"],
            "max_organ_stage": 4,
        }
    else:  # 真核生物（等级1-5）
        return {
            "origin_type": "eukaryote",
            "hard_forbidden": [],  # 真核生物可以自由发展任何结构
            "max_organ_stage": 4,
        }


def validate_gradual_evolution(
    organ_evolution: list,
    parent_organs: dict,
    biological_domain: str,
    get_constraints: Callable[[str], dict],
) -> tuple[bool, list]:
    """验证器官进化是否符合渐进式原则

    设计理念：最小限制，最大自由
    - 只验证"渐进式"（不能跳跃）
    - 只验证"原核/真核分界"（硬性生物学约束）
    - 其他一切都允许，让环境压力自然筛选

    返回：(是否有效, 过滤后的有效进化列表)
    """
    if not organ_evolution:
        return True, []

    valid_evolutions = []

    # 获取基础约束
    constraints = get_constraints(biological_domain)
    hard_forbidden = constraints.get("hard_forbidden", [])
    max_stage = constraints.get("max_organ_stage", 4)
    origin_type = constraints.get("origin_type", "eukaryote")

    for evo in organ_evolution:
        if not isinstance(evo, dict):
            continue

        category = evo.get("category", "")
        action = evo.get("action", "")
        current_stage = evo.get("current_stage", 0)
        target_stage = evo.get("target_stage", 0)
        structure_name = evo.get("structure_name", "")

        # === 核心验证1：阶段跳跃限制（渐进式核心） ===
        stage_jump = target_stage - current_stage
        if stage_jump > 2:
            logger.info(
                f"[渐进式] 修正跳跃: {structure_name} {current_stage}→{target_stage} "
                f"改为 →{min(current_stage + 2, max_stage)}"
            )
            target_stage = min(current_stage + 2, max_stage)
            evo["target_stage"] = target_stage

        # === 核心验证2：新器官从原基开始 ===
        if action == "initiate" and target_stage > 1:
            logger.info(f"[渐进式] 新器官从原基开始: {structure_name}")
            evo["target_stage"] = 1

        # === 核心验证3：原核/真核硬性分界 ===
        # 这是唯一的"禁止"规则，因为这需要内共生事件
        if origin_type == "prokaryote" and hard_forbidden:
            if any(f in structure_name for f in hard_forbidden):
                logger.warning(
                    f"[生物学约束] 原核生物不能发展真核结构: {structure_name} "
                    f"(需要内共生事件，非渐进演化)"
                )
                continue

        # === 验证4：enhance操作需要父代有该器官 ===
        if action == "enhance":
            if category not in parent_organs:
                # 自动转为initiate，允许发展新器官
                logger.debug(f"[器官] {category}不存在，转为新发展")
                evo["action"] = "initiate"
                evo["current_stage"] = 0
                evo["target_stage"] = 1
            else:
                # 使用父代实际阶段
                actual_stage = parent_organs[category].get("evolution_stage", 4)
                if current_stage != actual_stage:
                    evo["current_stage"] = actual_stage
                    if target_stage - actual_stage > 2:
                        evo["target_stage"] = min(actual_stage + 2, max_stage)

        valid_evolutions.append(evo)

    # 限制每次分化最多3个器官变化（放宽限制）
    if len(valid_evolutions) > 3:
        logger.info(f"[器官验证] 单次分化器官变化限制为3个")
        valid_evolutions = valid_evolutions[:3]

    return True, valid_evolutions


def process_plant_organ_changes(
    organs: dict,
    organ_changes: list,
    parent: Species,
    turn_index: int,
    plant_evolution: Any,
    plant_organ_catalog: dict,
    plant_organ_categories: dict,
) -> dict:
    """【植物混合模式】处理植物的器官变化

    支持：
    1. 里程碑必须器官（固定名称）
    2. 参考器官（预定义）
    3. 自定义器官（LLM创意）

    Args:
        organs: 继承的器官字典
        organ_changes: AI返回的器官变化列表
        parent: 父代物种
        turn_index: 当前回合

    Returns:
        更新后的器官字典（含植物专用结构）
    """
    current_stage = getattr(parent, "life_form_stage", 0)

    # 初始化或继承植物器官
    plant_organs = getattr(parent, "plant_organs", None)
    if plant_organs is None:
        plant_organs = {}
    else:
        plant_organs = dict(plant_organs)  # 深拷贝
        for cat, cat_organs in plant_organs.items():
            if isinstance(cat_organs, dict):
                plant_organs[cat] = dict(cat_organs)

    for change in organ_changes:
        if not isinstance(change, dict):
            continue

        category = change.get("category", "")
        change_type = change.get("change_type", "new")
        organ_name = change.get("organ_name", "")

        # 参数可能是新格式的 parameters 或旧格式的 parameter+delta
        parameters = change.get("parameters", {})
        if not parameters:
            # 兼容旧格式
            param_name = change.get("parameter", "")
            delta = change.get("delta", 0)
            if param_name:
                parameters = {param_name: delta}

        # 验证类别是否有效
        if category not in plant_organ_categories:
            logger.warning(f"[植物器官] 未知类别 {category}，跳过")
            continue

        cat_config = plant_organ_categories[category]
        min_stage = cat_config.get("min_stage", 0)

        # 验证阶段限制
        if current_stage < min_stage:
            logger.warning(
                f"[植物器官] {organ_name} 需要阶段{min_stage}，当前阶段{current_stage}，跳过"
            )
            continue

        # 检查是否是里程碑必须器官
        is_milestone_organ, milestone_id = (
            plant_evolution.is_milestone_required_organ(organ_name)
        )

        if change_type == "new":
            # 新增器官
            if category not in plant_organs:
                plant_organs[category] = {}

            # 使用验证系统获取修正后的参数
            valid, reason, corrected_params = plant_evolution.validate_custom_organ(
                category, organ_name, parameters, current_stage
            )

            if valid:
                plant_organs[category][organ_name] = {
                    **corrected_params,
                    "acquired_turn": turn_index,
                    "is_custom": organ_name
                    not in plant_organ_catalog.get(category, {}),
                }

                # 里程碑器官特殊标记
                if is_milestone_organ:
                    plant_organs[category][organ_name]["milestone_required"] = True
                    plant_organs[category][organ_name]["milestone_id"] = milestone_id

                organ_type = (
                    "自定义"
                    if plant_organs[category][organ_name]["is_custom"]
                    else "参考"
                )
                logger.info(
                    f"[植物器官] 新增{organ_type}器官: {organ_name} ({category})"
                )
            else:
                logger.warning(f"[植物器官] 验证失败: {reason}")

        elif change_type == "enhance":
            # 增强现有器官
            if category in plant_organs and organ_name in plant_organs[category]:
                existing = plant_organs[category][organ_name]

                # 应用参数增强
                param_ranges = cat_config.get("param_ranges", {})
                for param, delta in parameters.items():
                    current_val = existing.get(param, 0)
                    new_val = current_val + delta

                    # 范围钳制
                    if param in param_ranges:
                        min_val, max_val = param_ranges[param]
                        new_val = max(min_val, min(max_val, new_val))

                    existing[param] = new_val

                existing["modified_turn"] = turn_index
                logger.info(f"[植物器官] 增强器官: {organ_name} ({category})")
            else:
                logger.warning(
                    f"[植物器官] 增强失败: 器官 {organ_name} 不存在于 {category}"
                )

        elif change_type == "degrade":
            # 退化器官
            if category in plant_organs and organ_name in plant_organs[category]:
                # 里程碑器官不能退化
                if is_milestone_organ:
                    logger.warning(f"[植物器官] 里程碑器官 {organ_name} 不能退化")
                    continue

                existing = plant_organs[category][organ_name]
                existing["is_degraded"] = True
                existing["degraded_turn"] = turn_index
                logger.info(f"[植物器官] 退化器官: {organ_name} ({category})")

    # 将植物器官合并到通用器官字典中
    # 同时保持与动物器官系统的兼容性
    for category, cat_organs in plant_organs.items():
        if category not in organs:
            organs[category] = {}

        # 找到该类别中最高效的器官作为主器官
        if cat_organs:
            best_organ = None
            best_value = -1

            for name, data in cat_organs.items():
                if data.get("is_degraded"):
                    continue

                # 获取主要参数值作为排序依据
                cat_config = plant_organ_categories.get(category, {})
                main_param = (cat_config.get("required_params") or ["efficiency"])[0]
                value = data.get(main_param, 0)

                if value > best_value:
                    best_value = value
                    best_organ = name

            if best_organ:
                organs[category]["type"] = best_organ
                organs[category]["parameters"] = dict(cat_organs[best_organ])
                organs[category]["evolution_stage"] = 4  # 植物器官默认完善
                organs[category]["evolution_progress"] = 1.0
                organs[category]["is_active"] = True

    # 保存完整的植物器官到隐藏字段（供后续使用）
    organs["_plant_organs"] = plant_organs

    return organs


def update_capabilities(parent: Species, organs: dict) -> list[str]:
    """根据器官更新能力标签

    Args:
        parent: 父系物种
        organs: 当前器官字典

    Returns:
        能力标签列表（中文）
    """
    # 能力映射表：旧英文标签 -> 中文标签
    legacy_map = {
        "photosynthesis": "光合作用",
        "autotrophy": "自养",
        "flagellar_motion": "鞭毛运动",
        "chemical_detection": "化学感知",
        "heterotrophy": "异养",
        "chemosynthesis": "化能合成",
        "extremophile": "嗜极生物",
        "ciliary_motion": "纤毛运动",
        "limb_locomotion": "附肢运动",
        "swimming": "游泳",
        "light_detection": "感光",
        "vision": "视觉",
        "touch_sensation": "触觉",
        "aerobic_respiration": "有氧呼吸",
        "digestion": "消化",
        "armor": "盔甲",
        "spines": "棘刺",
        "venom": "毒素",
    }

    capabilities = set()

    # 继承并转换父代能力
    for cap in parent.capabilities:
        if cap in legacy_map:
            capabilities.add(legacy_map[cap])
        else:
            # 如果已经是中文或其他未映射的，直接保留
            capabilities.add(cap)

    # 根据活跃器官添加能力标签
    for category, organ_data in organs.items():
        if not organ_data.get("is_active", True):
            continue  # 跳过已退化的器官

        organ_type = organ_data.get("type", "").lower()

        # 运动能力
        if category == "locomotion":
            if (
                "flagella" in organ_type
                or "flagellum" in organ_type
                or "鞭毛" in organ_type
            ):
                capabilities.add("鞭毛运动")
            elif "cilia" in organ_type or "纤毛" in organ_type:
                capabilities.add("纤毛运动")
            elif (
                "leg" in organ_type
                or "limb" in organ_type
                or "足" in organ_type
                or "肢" in organ_type
            ):
                capabilities.add("附肢运动")
            elif "fin" in organ_type or "鳍" in organ_type:
                capabilities.add("游泳")

        # 感觉能力
        elif category == "sensory":
            if "eye" in organ_type or "ocellus" in organ_type or "眼" in organ_type:
                capabilities.add("感光")
                capabilities.add("视觉")
            elif (
                "photoreceptor" in organ_type
                or "eyespot" in organ_type
                or "光感受" in organ_type
                or "眼点" in organ_type
            ):
                capabilities.add("感光")
            elif "mechanoreceptor" in organ_type or "机械感受" in organ_type:
                capabilities.add("触觉")
            elif "chemoreceptor" in organ_type or "化学感受" in organ_type:
                capabilities.add("化学感知")

        # 代谢能力
        elif category == "metabolic":
            if (
                "chloroplast" in organ_type
                or "photosynthetic" in organ_type
                or "叶绿体" in organ_type
                or "光合" in organ_type
            ):
                capabilities.add("光合作用")
            elif "mitochondria" in organ_type or "线粒体" in organ_type:
                capabilities.add("有氧呼吸")

        # 消化能力
        elif category == "digestive":
            if organ_data.get("is_active", True):
                capabilities.add("消化")

        # 防御能力
        elif category == "defense":
            if (
                "shell" in organ_type
                or "carapace" in organ_type
                or "壳" in organ_type
                or "甲" in organ_type
            ):
                capabilities.add("盔甲")
            elif (
                "spine" in organ_type
                or "thorn" in organ_type
                or "刺" in organ_type
                or "棘" in organ_type
            ):
                capabilities.add("棘刺")
            elif "toxin" in organ_type or "毒" in organ_type:
                capabilities.add("毒素")

    return list(capabilities)
