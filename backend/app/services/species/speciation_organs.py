from __future__ import annotations

import logging

from ...models.species import Species

logger = logging.getLogger(f"{__package__}.speciation")


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
