from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ..models.config import UIConfig


class UIConfigUpdateRequest(BaseModel):
    config: UIConfig
    clear_provider_api_keys: set[str] = Field(default_factory=set)


@dataclass(frozen=True)
class ProviderCredentials:
    base_url: str | None
    api_key: str
    provider_type: str


def resolve_provider_credentials(
    request: dict[str, Any],
    current: UIConfig,
) -> ProviderCredentials:
    provider_id = str(request.get("provider_id") or "")
    stored = current.providers.get(provider_id)
    base_url = request.get("base_url") or (stored.base_url if stored else None)
    api_key = request.get("api_key") or (stored.api_key if stored else None)
    provider_type = request.get("provider_type") or (
        stored.provider_type if stored else "openai"
    )
    if not api_key:
        raise ValueError("API Key is not configured")
    return ProviderCredentials(
        base_url=base_url,
        api_key=api_key,
        provider_type=provider_type,
    )


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
