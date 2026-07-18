import logging
from urllib.parse import quote, urlencode

import pytest

from app.ai.model_router import (
    PROVIDER_TYPE_ANTHROPIC,
    PROVIDER_TYPE_GOOGLE,
    PROVIDER_TYPE_OPENAI,
    ModelRouter,
)


class RecordingSafeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []


@pytest.fixture
def router() -> ModelRouter:
    runtime_client = RecordingSafeRuntimeClient()
    router = ModelRouter(api_key="GLOBAL", runtime_client=runtime_client)
    assert router._runtime_client is runtime_client
    return router


def test_provider_request_location_for_openai_without_version_suffix(router: ModelRouter) -> None:
    assert router._provider_request_location(
        PROVIDER_TYPE_OPENAI,
        "https://public.example/",
        "model",
        "/chat/completions",
        api_key="selected-key",
        stream=False,
    ) == ("https://public.example", "/v1/chat/completions")


def test_provider_request_location_for_openai_with_version_suffix(router: ModelRouter) -> None:
    assert router._provider_request_location(
        PROVIDER_TYPE_OPENAI,
        "https://public.example/v1/",
        "model",
        "/chat/completions",
        api_key="selected-key",
        stream=False,
    ) == ("https://public.example/v1", "/chat/completions")


def test_provider_request_location_keeps_custom_openai_endpoint(router: ModelRouter) -> None:
    assert router._provider_request_location(
        PROVIDER_TYPE_OPENAI,
        "https://public.example/v1/",
        "model",
        "/custom/chat?format=json",
        api_key="selected-key",
        stream=False,
    ) == ("https://public.example/v1", "/custom/chat?format=json")


def test_provider_request_location_for_anthropic(router: ModelRouter) -> None:
    assert router._provider_request_location(
        PROVIDER_TYPE_ANTHROPIC,
        "https://public.example/",
        "model",
        None,
        api_key="selected-key",
        stream=False,
    ) == ("https://public.example", "/messages")


@pytest.mark.parametrize(
    ("stream", "action"),
    [(False, "generateContent"), (True, "streamGenerateContent")],
)
def test_provider_request_location_for_google_escapes_model_and_key(
    router: ModelRouter, caplog: pytest.LogCaptureFixture, stream: bool, action: str
) -> None:
    model_name = "gemini/2 ?#\u2603"
    selected_api_key = "SELECTED+&="

    with caplog.at_level(logging.DEBUG):
        base_url, request_target = router._provider_request_location(
            PROVIDER_TYPE_GOOGLE,
            "https://public.example/v1/",
            model_name,
            None,
            api_key=selected_api_key,
            stream=stream,
        )

    assert base_url == "https://public.example/v1"
    assert request_target == (
        f"/models/{quote(model_name, safe='')}:{action}?"
        f"{urlencode({'key': selected_api_key})}"
    )
    assert model_name not in request_target
    assert selected_api_key not in request_target
    assert "GLOBAL" not in request_target
    assert request_target not in caplog.text
    assert selected_api_key not in caplog.text
    assert "GLOBAL" not in caplog.text
