from types import SimpleNamespace

from app.ai.model_router import ModelConfig, ModelRouter
from app.core.ai_router_config import configure_model_router
from app.models.config import CapabilityRouteConfig, ProviderConfig, UIConfig
from app.services.system.embedding import EmbeddingService


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


def test_configure_model_router_atomically_commits_real_embedding_service(
    monkeypatch, tmp_path,
) -> None:
    router = ModelRouter()
    embedding_service = EmbeddingService(
        provider="provider-a",
        base_url="https://provider-a.example/v1",
        api_key="key-a",
        model="model-a",
        enabled=True,
        timeout=17,
        allow_fake_embeddings=False,
        cache_dir=tmp_path,
        allow_local_ai_endpoints=True,
    )
    settings = SimpleNamespace(speciation_model="fallback-model", embedding_provider="legacy")
    config = UIConfig(
        providers={
            "embedding-b": ProviderConfig(
                id="embedding-b",
                name="Embedding B",
                type="openai",
                provider_type="openai",
                base_url="https://provider-b.example/v1",
                api_key="key-b",
                selected_models=["model-b"],
            )
        },
        embedding_provider_id="embedding-b",
        embedding_model="model-b",
        allow_local_ai_endpoints=False,
    )
    atomic_calls = []
    original = embedding_service.configure_runtime_config

    def record_atomic_call(**kwargs):
        atomic_calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(embedding_service, "configure_runtime_config", record_atomic_call)

    configure_model_router(config, router, embedding_service, settings)

    assert atomic_calls == [{
        "provider": "openai",
        "base_url": "https://provider-b.example/v1",
        "api_key": "key-b",
        "model": "model-b",
        "enabled": True,
        "allow_local_ai_endpoints": False,
    }]
    assert embedding_service.provider == "openai"
    assert embedding_service.api_base_url == "https://provider-b.example/v1"
    assert embedding_service.api_key == "key-b"
    assert embedding_service.model == "model-b"
    assert embedding_service.enabled is True
    assert embedding_service.allow_local_ai_endpoints is False
    assert embedding_service.timeout == 17
    assert embedding_service.allow_fake_embeddings is False


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


def test_configure_model_router_clears_deleted_provider_runtime_state() -> None:
    router = ModelRouter(
        defaults={
            "speciation": ModelConfig(
                provider="openai", model="existing", endpoint="/chat/completions"
            )
        }
    )
    configured = UIConfig(
        providers={
            "old": ProviderConfig(
                id="old",
                name="Old",
                base_url="https://old.example/v1",
                api_key="old-key-sentinel",
                selected_models=["old-model"],
            )
        },
        default_provider_id="old",
        load_balance_enabled=True,
        capability_routes={
            "speciation": CapabilityRouteConfig(provider_ids=["old"])
        },
    )
    settings = SimpleNamespace(
        speciation_model="fallback",
        embedding_provider="openai",
        ai_base_url=None,
        ai_api_key=None,
    )

    configure_model_router(configured, router, None, settings)  # type: ignore[arg-type]
    router._provider_latencies["old"] = 1.0
    configure_model_router(
        UIConfig(providers={}, load_balance_enabled=True),
        router,
        None,  # type: ignore[arg-type]
        settings,
    )

    assert router.api_base_url is None
    assert router.api_key is None
    assert router.get_provider_pools_info() == {}
    assert router._lb_counters == {}
    assert router._provider_latencies == {}


def test_configure_model_router_none_clears_runtime_state_before_returning() -> None:
    router = ModelRouter(
        defaults={
            "speciation": ModelConfig(
                provider="openai", model="existing", endpoint="/chat/completions"
            )
        }
    )
    configured = UIConfig(
        providers={
            "old": ProviderConfig(
                id="old",
                name="Old",
                base_url="https://old.example/v1",
                api_key="old-test-key",
                selected_models=["old-model"],
            )
        },
        default_provider_id="old",
        load_balance_enabled=True,
        capability_routes={
            "speciation": CapabilityRouteConfig(provider_ids=["old"])
        },
    )
    settings = SimpleNamespace(
        speciation_model="fallback",
        embedding_provider="openai",
        ai_base_url="https://settings.example/v1",
        ai_api_key="settings-test-key",
    )

    configure_model_router(configured, router, None, settings)  # type: ignore[arg-type]
    router._provider_latencies["old"] = 1.0

    result = configure_model_router(
        None,  # type: ignore[arg-type]
        router,
        None,  # type: ignore[arg-type]
        settings,
    )

    assert result is None
    assert router.api_base_url == "https://settings.example/v1"
    assert router.api_key == "settings-test-key"
    assert router.overrides == {}
    assert router.get_provider_pools_info() == {}
    assert router._lb_counters == {}
    assert router._provider_latencies == {}
