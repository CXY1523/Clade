from __future__ import annotations

import logging

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
