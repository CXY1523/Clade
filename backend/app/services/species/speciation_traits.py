from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(f"{__package__}.speciation")


def apply_tradeoff_penalties(
    proposed_changes: dict[str, float],
    current_traits: dict[str, float],
    tradeoff_calculator: Any,
) -> dict[str, float]:
    """基于自动代价计算器为增益添加权衡代价。"""
    if not proposed_changes or not tradeoff_calculator:
        return proposed_changes

    gains = {k: v for k, v in proposed_changes.items() if v > 0}
    if not gains:
        return proposed_changes

    try:
        penalties = tradeoff_calculator.calculate_penalties(
            gains, current_traits or {}
        )
    except Exception as e:
        logger.debug(f"[权衡计算] 计算失败，跳过自动代价: {e}")
        return proposed_changes

    if not penalties:
        return proposed_changes

    merged = dict(proposed_changes)
    for trait, delta in penalties.items():
        merged[trait] = merged.get(trait, 0.0) + delta
    logger.debug(f"[权衡计算] 自动代价: {penalties}")
    return merged


def validate_trait_changes(
    old_traits: dict,
    new_traits: dict,
    trophic_level: float,
    get_attribute_limits: Callable[[float], dict],
) -> tuple[bool, str]:
    """验证属性变化是否符合营养级规则

    【修复】与 prompt 中的预算限制保持一致：
    - prompt 告诉 AI: max_increase=3.0
    - 这里的验证应该允许一定容错（考虑权衡后的净增益）
    - 硬限制改为 6.0（允许 AI 略微超出，但不能离谱）

    Returns:
        (验证是否通过, 错误信息)
    """
    # 获取营养级对应的属性上限
    limits = get_attribute_limits(trophic_level)

    # 1. 检查总和变化（与 prompt 中 max_increase=3.0 对应，但允许容错到 6.0）
    old_sum = sum(old_traits.values())
    new_sum = sum(new_traits.values())
    sum_diff = new_sum - old_sum

    # 【修改】净增益上限从 8 改为 6（prompt 要求 3，允许 2x 容错）
    # 超过 6 说明 AI 完全忽略了预算限制
    if sum_diff > 6.0:
        logger.warning(f"[属性验证] 净增益 {sum_diff:.1f} 超过上限 6.0（prompt 要求 ≤3.0）")
        return False, f"属性总和净增加{sum_diff:.1f}，超过上限6.0（建议≤3.0）"

    # 2. 检查总和是否超过营养级上限
    if new_sum > limits["total"]:
        return False, f"属性总和{new_sum:.1f}超过营养级T{trophic_level:.1f}的上限{limits['total']}"

    # 3. 检查单个属性是否超过特化上限
    above_specialized = [
        (k, v) for k, v in new_traits.items() if v > limits["specialized"]
    ]
    if above_specialized:
        return False, f"属性{above_specialized[0][0]}={above_specialized[0][1]:.1f}超过特化上限{limits['specialized']}"

    # 4. 检查超过基础上限的属性数量
    above_base_count = sum(1 for v in new_traits.values() if v > limits["base"])
    if above_base_count > 2:
        return False, f"{above_base_count}个属性超过基础上限{limits['base']}，最多允许2个"

    # 5. 检查权衡（有增必有减，除非是小幅提升）
    increases = sum(1 for k, v in new_traits.items() if v > old_traits.get(k, 0))
    decreases = sum(1 for k, v in new_traits.items() if v < old_traits.get(k, 0))

    # 【修改】放宽权衡检查：只有净增益 >4 且无任何减少时才拒绝
    if increases > 0 and decreases == 0 and sum_diff > 4.0:
        return False, f"净增益{sum_diff:.1f}但无权衡代价（需要至少一项属性降低）"

    return True, "验证通过"
