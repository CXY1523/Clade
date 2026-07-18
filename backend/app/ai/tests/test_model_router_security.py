import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from typing import Any
from urllib.parse import quote, urlencode

import httpx
import pytest

from app.ai import model_router as model_router_module
from app.ai.model_router import (
    PROVIDER_TYPE_ANTHROPIC,
    PROVIDER_TYPE_GOOGLE,
    PROVIDER_TYPE_OPENAI,
    ModelConfig,
    ModelRouter,
    ProviderPoolConfig,
)
from app.security import (
    AI_JSON_MAX_BYTES,
    STREAM_EVENT_MAX_BYTES,
    STREAM_MAX_BYTES,
    OutboundRequestError,
)


PUBLIC_BLOCKED = "That network address is not allowed"
PUBLIC_TIMEOUT = "The upstream request timed out"
UPSTREAM_STREAM_ERROR = "AI stream response error"


class ForbiddenAsyncResponseContext:
    async def __aenter__(self) -> None:
        raise httpx.ConnectError("DIRECT_HTTPX_FORBIDDEN")

    async def __aexit__(self, *args: object) -> None:
        return None


class ForbiddenAsyncClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    async def __aenter__(self) -> "ForbiddenAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def post(self, *args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("DIRECT_HTTPX_FORBIDDEN")

    def stream(self, *args: object, **kwargs: object) -> ForbiddenAsyncResponseContext:
        return ForbiddenAsyncResponseContext()


@pytest.fixture(autouse=True)
def forbid_direct_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_post(*args: object, **kwargs: object) -> None:
        raise httpx.ConnectError("DIRECT_HTTPX_FORBIDDEN")

    async def no_backoff(_: float) -> None:
        return None

    monkeypatch.setattr(model_router_module.httpx, "post", forbidden_post)
    monkeypatch.setattr(model_router_module.httpx, "AsyncClient", ForbiddenAsyncClient)
    monkeypatch.setattr(model_router_module.asyncio, "sleep", no_backoff)


class RecordingAsyncLines(AsyncIterator[str]):
    def __init__(
        self,
        lines: list[str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        self._lines = iter(lines or [])
        self._error = error
        self._raised_error = False
        self.close_count = 0

    def __aiter__(self) -> "RecordingAsyncLines":
        return self

    async def __anext__(self) -> str:
        if self._error is not None and not self._raised_error:
            self._raised_error = True
            raise self._error
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.close_count += 1


class RecordingSafeRuntimeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.sync_results: list[dict[str, Any] | BaseException] = []
        self.async_results: list[dict[str, Any] | BaseException] = []
        self.stream_lines: list[str] = []
        self.stream_error: BaseException | None = None
        self.stream_iterator: RecordingAsyncLines | None = None
        self.on_async_call: Callable[[int], None] | None = None

    @staticmethod
    def _provider_type(headers: dict[str, str]) -> str:
        if "x-api-key" in headers:
            return PROVIDER_TYPE_ANTHROPIC
        if "Authorization" in headers:
            return PROVIDER_TYPE_OPENAI
        return PROVIDER_TYPE_GOOGLE

    def _record(self, method: str, base_url: str, **kwargs: Any) -> None:
        self.calls.append(
            {
                "method": method,
                "base_url": base_url,
                "provider_type": self._provider_type(dict(kwargs["headers"])),
                **kwargs,
            }
        )

    @staticmethod
    def _next_result(results: list[dict[str, Any] | BaseException]) -> dict[str, Any]:
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def post_json(
        self,
        base_url: str,
        *,
        request_target: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        allow_local: bool,
        read_timeout: float,
        max_bytes: int,
    ) -> dict[str, Any]:
        self._record(
            "post_json",
            base_url,
            request_target=request_target,
            headers=headers,
            json_body=json_body,
            allow_local=allow_local,
            read_timeout=read_timeout,
            max_bytes=max_bytes,
        )
        return self._next_result(self.sync_results)

    async def apost_json(
        self,
        base_url: str,
        *,
        request_target: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        allow_local: bool,
        read_timeout: float,
        max_bytes: int,
    ) -> dict[str, Any]:
        self._record(
            "apost_json",
            base_url,
            request_target=request_target,
            headers=headers,
            json_body=json_body,
            allow_local=allow_local,
            read_timeout=read_timeout,
            max_bytes=max_bytes,
        )
        if self.on_async_call is not None:
            self.on_async_call(len(self.calls))
        return self._next_result(self.async_results)

    def astream_lines(
        self,
        base_url: str,
        *,
        request_target: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
        allow_local: bool,
        max_bytes: int,
        max_event_bytes: int,
        idle_timeout: float | None = None,
        total_timeout: float | None = None,
    ) -> AsyncIterator[str]:
        self._record(
            "astream_lines",
            base_url,
            request_target=request_target,
            headers=headers,
            json_body=json_body,
            allow_local=allow_local,
            max_bytes=max_bytes,
            max_event_bytes=max_event_bytes,
            idle_timeout=idle_timeout,
            total_timeout=total_timeout,
        )
        self.stream_iterator = RecordingAsyncLines(self.stream_lines, self.stream_error)
        return self.stream_iterator


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


def _remote_router(
    runtime_client: RecordingSafeRuntimeClient,
    *,
    provider_type: str = PROVIDER_TYPE_OPENAI,
    base_url: str = "https://default.example/api",
    api_key: str = "default-key",
    timeout: int = 17,
    max_retries: int = 3,
    allow_local: bool = True,
) -> ModelRouter:
    router = ModelRouter(
        defaults={
            "generate": ModelConfig(
                provider=provider_type,
                model="default-model",
                provider_type=provider_type,
                extra_body={"temperature": 0},
            )
        },
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        runtime_client=runtime_client,
        allow_local_ai_endpoints=allow_local,
    )
    router.set_prompt("generate", "Answer for {name}")
    return router


def _assert_json_call(
    call: dict[str, Any],
    *,
    method: str,
    base_url: str,
    request_target: str,
    provider_type: str,
    timeout: float,
    allow_local: bool,
) -> None:
    assert call["method"] == method
    assert call["base_url"] == base_url
    assert call["request_target"] == request_target
    assert call["provider_type"] == provider_type
    assert call["read_timeout"] == timeout
    assert call["max_bytes"] == AI_JSON_MAX_BYTES
    assert call["allow_local"] is allow_local


def test_invoke_uses_safe_sync_client_and_preserves_parsing() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    raw = {"choices": [{"message": {"content": '```json\n{"answer": 7}\n```'}}]}
    runtime_client.sync_results = [raw]
    router = _remote_router(runtime_client)

    result = router.invoke("generate", {"name": "oak"})

    assert result == {
        "provider": PROVIDER_TYPE_OPENAI,
        "model": "default-model",
        "prompt": "Answer for oak",
        "payload": {"name": "oak"},
        "content": {"answer": 7},
        "raw": raw,
    }
    assert len(runtime_client.calls) == 1
    call = runtime_client.calls[0]
    _assert_json_call(
        call,
        method="post_json",
        base_url="https://default.example/api",
        request_target="/v1/chat/completions",
        provider_type=PROVIDER_TYPE_OPENAI,
        timeout=17,
        allow_local=True,
    )
    assert call["headers"] == {"Authorization": "Bearer default-key"}
    assert call["json_body"] == {
        "model": "default-model",
        "messages": [
            {"role": "system", "content": "Answer for oak"},
            {
                "role": "user",
                "content": json.dumps({"name": "oak"}, ensure_ascii=False, indent=2),
            },
        ],
        "temperature": 0,
    }


def test_constructor_base_url_from_settings_uses_the_same_safe_client() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.sync_results = [
        {"choices": [{"message": {"content": "constructor route"}}]}
    ]
    router = _remote_router(
        runtime_client,
        base_url="https://settings.example/v1/",
        api_key="settings-key",
    )

    assert router.invoke("generate", {"name": "fir"})["content"] == "constructor route"

    assert len(runtime_client.calls) == 1
    assert runtime_client.calls[0]["base_url"] == "https://settings.example/v1"
    assert runtime_client.calls[0]["request_target"] == "/chat/completions"
    assert runtime_client.calls[0]["headers"] == {
        "Authorization": "Bearer settings-key"
    }


def test_invoke_blocked_url_returns_only_public_error() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.sync_results = [
        OutboundRequestError("private_network_blocked", 400, PUBLIC_BLOCKED)
    ]
    router = _remote_router(runtime_client)

    result = router.invoke("generate", {"name": "oak"})

    assert result["error"] == PUBLIC_BLOCKED
    assert len(runtime_client.calls) == 1


def test_ainvoke_revalidates_on_every_retry_for_same_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.async_results = [
        OutboundRequestError("outbound_timeout", 504, PUBLIC_TIMEOUT),
        OutboundRequestError("outbound_timeout", 504, PUBLIC_TIMEOUT),
        {
            "candidates": [
                {"content": {"parts": [{"text": '{"answer": "cedar"}'}]}}
            ]
        },
    ]
    router = _remote_router(runtime_client, allow_local=False)
    router.configure_load_balance(True)
    router.set_provider_pool(
        "generate",
        [
            ProviderPoolConfig(
                provider_id="selected-google",
                base_url="https://pool.example/v1beta",
                api_key="selected+key",
                provider_type=PROVIDER_TYPE_GOOGLE,
                model="gemini pool/model",
            ),
            ProviderPoolConfig(
                provider_id="must-not-select",
                base_url="https://other.example/v1",
                api_key="other-key",
                provider_type=PROVIDER_TYPE_OPENAI,
                model="other-model",
            ),
        ],
    )
    runtime_client.on_async_call = lambda count: setattr(
        router, "allow_local_ai_endpoints", count >= 1
    )

    async def no_backoff(_: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", no_backoff)

    result = asyncio.run(router.ainvoke("generate", {"name": "cedar"}))

    assert result["content"] == {"answer": "cedar"}
    assert router._lb_counters["generate"] == 1
    assert len(runtime_client.calls) == 3
    expected_target = (
        "/models/gemini%20pool%2Fmodel:generateContent?"
        + urlencode({"key": "selected+key"})
    )
    for index, call in enumerate(runtime_client.calls):
        _assert_json_call(
            call,
            method="apost_json",
            base_url="https://pool.example/v1beta",
            request_target=expected_target,
            provider_type=PROVIDER_TYPE_GOOGLE,
            timeout=17,
            allow_local=index > 0,
        )
        assert call["headers"] == {"Content-Type": "application/json", "Connection": "close"}
        assert call["json_body"]["contents"][0]["role"] == "user"
        assert "generationConfig" not in call["json_body"]


def test_ainvoke_unsafe_selected_pool_provider_does_not_fail_over() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.async_results = [
        OutboundRequestError("private_network_blocked", 400, PUBLIC_BLOCKED)
    ]
    router = _remote_router(runtime_client, allow_local=False)
    router.configure_load_balance(True)
    router.set_provider_pool(
        "generate",
        [
            ProviderPoolConfig(
                provider_id="unsafe-selected",
                base_url="https://unsafe.example/v1",
                api_key="unsafe-key",
                model="selected-model",
            ),
            ProviderPoolConfig(
                provider_id="safe-but-must-not-fail-over",
                base_url="https://safe.example/v1",
                api_key="safe-key",
                model="other-model",
            ),
        ],
    )

    result = asyncio.run(router.ainvoke("generate", {"name": "oak"}))

    assert result["error"] == PUBLIC_BLOCKED
    assert len(runtime_client.calls) == 1
    assert runtime_client.calls[0]["base_url"] == "https://unsafe.example/v1"
    assert router._lb_counters["generate"] == 1


async def _collect_stream(router: ModelRouter) -> list[Any]:
    return [event async for event in router.astream("generate", {"name": "elm"})]


def test_astream_uses_safe_lines_and_preserves_status_heartbeat_and_chunks() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.stream_lines = [
        ": heartbeat",
        "",
        'data: {"choices":[{"delta":{"content":"hel"}}]}',
        ": heartbeat",
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        "data: [DONE]",
    ]
    router = _remote_router(runtime_client, allow_local=False)
    router.configure_overrides(
        {
            "generate": {
                "base_url": "https://override.example/v1/",
                "api_key": "override-key",
                "provider_type": PROVIDER_TYPE_OPENAI,
                "model": "override-model",
                "timeout": 23,
            }
        }
    )

    events = asyncio.run(_collect_stream(router))

    assert [(event.get("type"), event.get("state")) for event in events if isinstance(event, dict)] == [
        ("status", "connecting"),
        ("status", "connected"),
        ("status", "receiving"),
        ("status", "completed"),
    ]
    assert [event for event in events if isinstance(event, str)] == ["hel", "lo"]
    assert len(runtime_client.calls) == 1
    call = runtime_client.calls[0]
    assert call == {
        "method": "astream_lines",
        "base_url": "https://override.example/v1",
        "request_target": "/chat/completions",
        "provider_type": PROVIDER_TYPE_OPENAI,
        "headers": {
            "Authorization": "Bearer override-key",
            "Connection": "close",
        },
        "json_body": {
            "model": "override-model",
            "messages": [
                {"role": "system", "content": "Answer for elm"},
                {
                    "role": "user",
                    "content": json.dumps({"name": "elm"}, ensure_ascii=False, indent=2),
                },
            ],
            "temperature": 0,
            "stream": True,
        },
        "allow_local": False,
        "max_bytes": STREAM_MAX_BYTES,
        "max_event_bytes": STREAM_EVENT_MAX_BYTES,
        "idle_timeout": 120.0,
        "total_timeout": None,
    }


@pytest.mark.parametrize(
    ("provider_type", "lines", "expected_target", "expected_chunks"),
    [
        (
            PROVIDER_TYPE_ANTHROPIC,
            [
                'data: {"type":"content_block_delta","delta":{"text":"ant"}}',
                'data: {"type":"content_block_delta","delta":{"text":"hropic"}}',
                "data: [DONE]",
            ],
            "/messages",
            ["ant", "hropic"],
        ),
        (
            PROVIDER_TYPE_GOOGLE,
            [
                '[{"candidates":[{"content":{"parts":[{"text":"goo"}]}}]},',
                '{"candidates":[{"content":{"parts":[{"text":"gle"}]}}]}]',
            ],
            "/models/default-model:streamGenerateContent?key=default-key",
            ["goo", "gle"],
        ),
    ],
)
def test_astream_preserves_anthropic_and_google_parsing(
    provider_type: str,
    lines: list[str],
    expected_target: str,
    expected_chunks: list[str],
) -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.stream_lines = lines
    router = _remote_router(runtime_client, provider_type=provider_type)

    events = asyncio.run(_collect_stream(router))

    assert [event for event in events if isinstance(event, str)] == expected_chunks
    assert [event["state"] for event in events if isinstance(event, dict)] == [
        "connecting",
        "connected",
        "receiving",
        "completed",
    ]
    call = runtime_client.calls[0]
    assert call["provider_type"] == provider_type
    assert call["request_target"] == expected_target


def test_astream_security_error_emits_one_redacted_error_then_closes() -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.stream_error = OutboundRequestError(
        "private_network_blocked", 400, PUBLIC_BLOCKED
    )
    router = _remote_router(runtime_client, allow_local=False)

    events = asyncio.run(_collect_stream(router))

    error_events = [event for event in events if event["type"] == "error"]
    assert [event["state"] for event in events if event["type"] == "status"] == [
        "connecting"
    ]
    assert len(error_events) == 1
    assert error_events[0]["message"] == PUBLIC_BLOCKED
    assert len(runtime_client.calls) == 1
    assert runtime_client.stream_iterator is not None
    assert runtime_client.stream_iterator.close_count == 1


@pytest.mark.parametrize(
    "lines",
    [
        ['data: {"error":{"message":"UPSTREAM-SENTINEL"}}'],
        ['data: {"malformed":"MALFORMED-SENTINEL"'],
    ],
)
def test_astream_upstream_error_and_malformed_fragment_are_redacted(
    lines: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    runtime_client = RecordingSafeRuntimeClient()
    runtime_client.stream_lines = lines
    router = _remote_router(runtime_client)

    with caplog.at_level(logging.WARNING):
        events = asyncio.run(_collect_stream(router))

    error_events = [event for event in events if isinstance(event, dict) and event["type"] == "error"]
    assert len(error_events) == 1
    assert error_events[0]["message"] == UPSTREAM_STREAM_ERROR
    rendered_events = repr(events)
    assert "UPSTREAM-SENTINEL" not in rendered_events
    assert "MALFORMED-SENTINEL" not in rendered_events
    assert "UPSTREAM-SENTINEL" not in caplog.text
    assert "MALFORMED-SENTINEL" not in caplog.text
    assert runtime_client.stream_iterator is not None
    assert runtime_client.stream_iterator.close_count == 1
