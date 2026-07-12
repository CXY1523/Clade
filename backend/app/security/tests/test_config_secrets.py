from app.models.config import ProviderConfig, UIConfig
from app.security.config_secrets import merge_ui_config_secrets, public_ui_config


def config_with_key(key: str = "sk-secret") -> UIConfig:
    return UIConfig(
        providers={
            "main": ProviderConfig(
                id="main",
                name="Main",
                provider_type="openai",
                api_key=key,
            )
        },
        ai_api_key="legacy-chat",
        embedding_api_key="legacy-embedding",
    )


def test_public_config_never_contains_stored_secrets() -> None:
    public = public_ui_config(config_with_key())
    assert public["providers"]["main"]["api_key"] == ""
    assert public["providers"]["main"]["api_key_configured"] is True
    assert public["ai_api_key"] == ""
    assert public["ai_api_key_configured"] is True
    assert public["embedding_api_key"] == ""
    assert public["embedding_api_key_configured"] is True
    assert "sk-secret" not in str(public)


def test_empty_key_preserves_existing_secret() -> None:
    incoming = config_with_key("")
    merged = merge_ui_config_secrets(config_with_key(), incoming, set())
    assert merged.providers["main"].api_key == "sk-secret"


def test_non_empty_key_replaces_existing_secret() -> None:
    incoming = config_with_key("sk-new")
    merged = merge_ui_config_secrets(config_with_key(), incoming, set())
    assert merged.providers["main"].api_key == "sk-new"


def test_explicit_clear_removes_provider_secret() -> None:
    incoming = config_with_key("")
    merged = merge_ui_config_secrets(config_with_key(), incoming, {"main"})
    assert merged.providers["main"].api_key is None


def test_removed_provider_is_not_reintroduced() -> None:
    incoming = UIConfig(providers={})
    merged = merge_ui_config_secrets(config_with_key(), incoming, set())
    assert merged.providers == {}
