from types import SimpleNamespace

from app.ai.model_router import ModelConfig, ModelRouter
from app.core.ai_router_config import configure_model_router
from app.models.config import CapabilityRouteConfig, ProviderConfig, UIConfig


class RecordingEmbeddingService:
    def __init__(self) -> None:
        self.network_calls = 0


def test_configure_model_router_propagates_local_endpoint_policy() -> None:
    router = ModelRouter()
    embedding_service = RecordingEmbeddingService()
    settings = SimpleNamespace(speciation_model="fallback-model", embedding_provider="openai")
    config = UIConfig(allow_local_ai_endpoints=True)

    configure_model_router(config, router, embedding_service, settings)

    assert router.allow_local_ai_endpoints is True
    assert embedding_service.allow_local_ai_endpoints is True

    config.allow_local_ai_endpoints = False
    configure_model_router(config, router, embedding_service, settings)

    assert router.allow_local_ai_endpoints is False
    assert embedding_service.allow_local_ai_endpoints is False


def test_configure_model_router_retains_legacy_embedding_source_for_safe_runtime_validation() -> None:
    router = ModelRouter()
    embedding_service = RecordingEmbeddingService()
    settings = SimpleNamespace(speciation_model="fallback-model", embedding_provider="legacy")
    config = UIConfig(
        embedding_base_url="https://public.example/v1/",
        embedding_api_key="legacy-key",
        embedding_model="legacy-model",
    )

    configure_model_router(config, router, embedding_service, settings)

    assert embedding_service.provider == "legacy"
    assert embedding_service.api_base_url == "https://public.example/v1/"
    assert embedding_service.api_key == "legacy-key"
    assert embedding_service.model == "legacy-model"
    assert embedding_service.enabled is True
    assert embedding_service.network_calls == 0


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
