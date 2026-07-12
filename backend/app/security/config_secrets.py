from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..models.config import UIConfig


class UIConfigUpdateRequest(BaseModel):
    config: UIConfig
    clear_provider_api_keys: set[str] = Field(default_factory=set)


def public_ui_config(config: UIConfig) -> dict[str, Any]:
    data = config.model_dump(mode="json")
    for provider in data.get("providers", {}).values():
        provider["api_key_configured"] = bool(provider.get("api_key"))
        provider["api_key"] = ""
    for field in ("ai_api_key", "embedding_api_key"):
        data[f"{field}_configured"] = bool(data.get(field))
        data[field] = ""
    return data


def merge_ui_config_secrets(
    current: UIConfig,
    incoming: UIConfig,
    clear_provider_api_keys: set[str],
) -> UIConfig:
    data = incoming.model_dump(mode="python")
    current_providers = current.providers
    for provider_id, provider in data.get("providers", {}).items():
        if provider_id in clear_provider_api_keys:
            provider["api_key"] = None
        elif not provider.get("api_key") and provider_id in current_providers:
            provider["api_key"] = current_providers[provider_id].api_key
    for field in ("ai_api_key", "embedding_api_key"):
        if not data.get(field):
            data[field] = getattr(current, field)
    return UIConfig.model_validate(data)
