from types import SimpleNamespace

from app.ai.model_router import ModelConfig, ModelRouter
from app.core.ai_router_config import configure_model_router
from app.models.config import CapabilityRouteConfig, ProviderConfig, UIConfig


def test_configure_model_router_deduplicates_provider_pool_ids() -> None:
    router = ModelRouter(
        defaults={
            "speciation": ModelConfig(
                provider="openai",
                model="existing-model",
                endpoint="/chat/completions",
            )
        }
    )
    config = UIConfig(
        providers={
            "main": ProviderConfig(
                id="main",
                name="Main",
                base_url="https://example.com/v1",
                api_key="sk-test",
                selected_models=["model-1"],
            )
        },
        load_balance_enabled=True,
        capability_routes={
            "speciation": CapabilityRouteConfig(
                provider_ids=["main", "main", "unknown", "main"]
            )
        },
    )
    settings = SimpleNamespace(
        speciation_model="fallback-model",
        embedding_provider="openai",
    )

    configure_model_router(config, router, None, settings)  # type: ignore[arg-type]

    assert router.get_provider_pools_info()["speciation"] == ["main"]
