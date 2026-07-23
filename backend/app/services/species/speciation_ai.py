from __future__ import annotations

from typing import Any


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
