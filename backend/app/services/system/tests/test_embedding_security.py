from __future__ import annotations

import ast
import math
import threading
from pathlib import Path
from typing import Any, Callable

import httpx
import pytest

from app.security import EMBEDDING_JSON_MAX_BYTES, OutboundRequestError
from app.services.system import embedding as embedding_module
from app.services.system.embedding import EmbeddingService


FORBIDDEN_NETWORK_MODULES = (
    "httpx",
    "requests",
    "urllib.request",
    "urllib3",
    "aiohttp",
    "httpcore",
    "socket",
)
FORBIDDEN_NETWORK_CALLS = {
    "httpx.post",
    "httpx.Client",
    "httpx.AsyncClient",
    "httpx.request",
    "httpx.stream",
    "httpx.get",
    "httpx.put",
    "httpx.patch",
    "httpx.delete",
    "httpx.head",
    "httpx.options",
}


def _find_forbidden_network_uses(source: str) -> list[str]:
    tree = ast.parse(source)
    findings: list[str] = []
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound_name = alias.asname or alias.name.split(".", 1)[0]
                aliases[bound_name] = alias.name if alias.asname else bound_name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                if alias.name == "*":
                    continue
                aliases[alias.asname or alias.name] = ".".join(
                    part for part in (module, alias.name) if part
                )

    def name_of(node: ast.AST) -> str | None:
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
            resolved = list(reversed(parts))
            resolved[0] = aliases.get(resolved[0], resolved[0])
            return ".".join(resolved)
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any(
                    alias.name == module or alias.name.startswith(module + ".")
                    for module in FORBIDDEN_NETWORK_MODULES
                ):
                    findings.append(alias.name)
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imported_name = ".".join(
                    part for part in (module, alias.name) if part
                )
                if any(
                    imported_name == banned
                    or imported_name.startswith(banned + ".")
                    for banned in FORBIDDEN_NETWORK_MODULES
                ):
                    findings.append(imported_name)
        if isinstance(node, ast.Call):
            call_name = name_of(node.func)
            if call_name in FORBIDDEN_NETWORK_CALLS or (
                call_name is not None
                and any(
                    call_name == module or call_name.startswith(module + ".")
                    for module in FORBIDDEN_NETWORK_MODULES
                )
            ):
                findings.append(call_name)
    return findings


class ForbiddenSyncClient:
    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("DIRECT_HTTPX_FORBIDDEN")


@pytest.fixture(autouse=True)
def forbid_direct_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden_request(*args: object, **kwargs: object) -> None:
        raise AssertionError("DIRECT_HTTPX_FORBIDDEN")

    for verb in ("delete", "get", "head", "options", "patch", "post", "put", "request", "stream"):
        monkeypatch.setattr(httpx, verb, forbidden_request)
    monkeypatch.setattr(httpx, "Client", ForbiddenSyncClient)
    monkeypatch.setattr(httpx, "AsyncClient", ForbiddenSyncClient)


class RecordingSafeRuntimeClient:
    def __init__(self, results: list[dict[str, Any] | BaseException]) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results = results
        self.on_call: Callable[[int], None] | None = None

    def post_json(self, base_url: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"base_url": base_url, **kwargs})
        if self.on_call is not None:
            self.on_call(len(self.calls))
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


def remote_service(client: RecordingSafeRuntimeClient) -> EmbeddingService:
    return EmbeddingService(
        base_url="https://embedding.example/v1/",
        api_key="secret-key",
        model="text-embedding-3-small",
        enabled=True,
        timeout=23,
        runtime_client=client,
        allow_local_ai_endpoints=True,
    )


def test_embedding_request_uses_safe_client_and_preserves_payload() -> None:
    client = RecordingSafeRuntimeClient(
        [{"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}]
    )

    assert remote_service(client)._request_embedding_chunk(["x" * 2001, "oak"], True) == [[1.0], [2.0]]

    assert client.calls == [{
        "base_url": "https://embedding.example/v1/",
        "request_target": "/embeddings",
        "headers": {"Authorization": "Bearer secret-key"},
        "json_body": {"model": "text-embedding-3-small", "input": ["x" * 2000 + "...", "oak"]},
        "allow_local": True,
        "read_timeout": 23,
        "max_bytes": EMBEDDING_JSON_MAX_BYTES,
    }]


@pytest.mark.parametrize("payload", [
    {},
    {"data": "not-a-list"},
    {"data": []},
    {"data": ["not-a-dict"]},
    {"data": [{"embedding": [1.0]}]},
    {"data": [{"index": "0", "embedding": [1.0]}]},
    {"data": [{"index": 0, "embedding": []}]},
    {"data": [{"index": 0, "embedding": [True]}]},
    {"data": [{"index": 0, "embedding": [math.inf]}]},
])
def test_malformed_embedding_schema_has_fixed_public_error(payload: dict[str, Any]) -> None:
    client = RecordingSafeRuntimeClient([payload])

    with pytest.raises(RuntimeError, match="外部服务响应无效"):
        remote_service(client)._request_embedding_chunk(["oak"], False)


@pytest.mark.parametrize("code", ["outbound_connect_failed", "outbound_timeout"])
@pytest.mark.parametrize(
    ("require_real", "allow_fake", "uses_fake"),
    [(False, True, True), (False, False, False), (True, True, False), (True, False, False)],
)
def test_only_exhausted_availability_errors_can_use_fake_vectors(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    require_real: bool,
    allow_fake: bool,
    uses_fake: bool,
) -> None:
    public_message = "PUBLIC-AVAILABILITY-ERROR"
    client = RecordingSafeRuntimeClient(
        [OutboundRequestError(code, 504, public_message) for _ in range(3)]
    )
    service = remote_service(client)
    service.allow_fake_embeddings = allow_fake
    sleeps: list[int] = []
    monkeypatch.setattr("app.services.system.embedding.time.sleep", sleeps.append)

    if uses_fake:
        assert len(service._request_embedding_chunk(["oak"], require_real)) == 1
        assert service._stats["fake_embeds"] == 1
    else:
        with pytest.raises(RuntimeError, match=public_message):
            service._request_embedding_chunk(["oak"], require_real)
        assert service._stats["fake_embeds"] == 0

    assert len(client.calls) == 3
    assert sleeps == [1, 2]
    assert {call["base_url"] for call in client.calls} == {"https://embedding.example/v1/"}


@pytest.mark.parametrize(
    "error",
    [
        OutboundRequestError("private_network_blocked", 400, "PUBLIC-POLICY"),
        OutboundRequestError("outbound_bad_response", 502, "PUBLIC-SCHEMA"),
        OutboundRequestError("outbound_response_too_large", 502, "PUBLIC-SIZE"),
    ],
)
def test_nonretryable_outbound_errors_never_use_fake_vectors(
    error: OutboundRequestError,
) -> None:
    client = RecordingSafeRuntimeClient([error])
    service = remote_service(client)

    with pytest.raises(RuntimeError, match=error.public_message):
        service._request_embedding_chunk(["oak"], False)

    assert len(client.calls) == 1
    assert service._stats["fake_embeds"] == 0


def test_chunk_uses_one_local_policy_snapshot_for_all_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingSafeRuntimeClient(
        [OutboundRequestError("outbound_timeout", 504, "PUBLIC") for _ in range(3)]
    )
    service = remote_service(client)
    service.allow_fake_embeddings = False
    client.on_call = lambda _: setattr(service, "allow_local_ai_endpoints", False)
    monkeypatch.setattr("app.services.system.embedding.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="PUBLIC"):
        service._request_embedding_chunk(["oak"], False)

    assert [call["allow_local"] for call in client.calls] == [True, True, True]


def test_chunk_snapshots_complete_provider_config_across_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingSafeRuntimeClient([
        OutboundRequestError("outbound_timeout", 504, "PUBLIC-A"),
        {"data": [{"index": 0, "embedding": [1.0]}]},
        {"data": [{"index": 0, "embedding": [2.0]}]},
    ])
    service = EmbeddingService(
        provider="provider-a",
        base_url="https://provider-a.example/v1",
        api_key="key-a",
        model="model-a",
        enabled=True,
        timeout=11,
        allow_fake_embeddings=False,
        runtime_client=client,
        allow_local_ai_endpoints=True,
    )

    def switch_to_provider_b(call_number: int) -> None:
        if call_number != 1:
            return
        service.provider = "provider-b"
        service.api_base_url = "https://provider-b.example/v1"
        service.api_key = "key-b"
        service.model = "model-b"
        service.enabled = True
        service.timeout = 29
        service.allow_local_ai_endpoints = False
        service.allow_fake_embeddings = True

    client.on_call = switch_to_provider_b
    monkeypatch.setattr("app.services.system.embedding.time.sleep", lambda _: None)

    assert service._request_embedding_chunk(["first"], True) == [[1.0]]
    assert service._request_embedding_chunk(["second"], True) == [[2.0]]

    assert client.calls == [
        {
            "base_url": "https://provider-a.example/v1",
            "request_target": "/embeddings",
            "headers": {"Authorization": "Bearer key-a"},
            "json_body": {"model": "model-a", "input": ["first"]},
            "allow_local": True,
            "read_timeout": 11,
            "max_bytes": EMBEDDING_JSON_MAX_BYTES,
        },
        {
            "base_url": "https://provider-a.example/v1",
            "request_target": "/embeddings",
            "headers": {"Authorization": "Bearer key-a"},
            "json_body": {"model": "model-a", "input": ["first"]},
            "allow_local": True,
            "read_timeout": 11,
            "max_bytes": EMBEDDING_JSON_MAX_BYTES,
        },
        {
            "base_url": "https://provider-b.example/v1",
            "request_target": "/embeddings",
            "headers": {"Authorization": "Bearer key-b"},
            "json_body": {"model": "model-b", "input": ["second"]},
            "allow_local": False,
            "read_timeout": 29,
            "max_bytes": EMBEDDING_JSON_MAX_BYTES,
        },
    ]


def test_chunk_does_not_enable_fake_fallback_after_first_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = RecordingSafeRuntimeClient([
        OutboundRequestError("outbound_timeout", 504, "PUBLIC")
        for _ in range(6)
    ])
    service = remote_service(client)
    service.allow_fake_embeddings = False

    def enable_fake_after_first_failure(call_number: int) -> None:
        if call_number == 1:
            service.allow_fake_embeddings = True

    client.on_call = enable_fake_after_first_failure
    monkeypatch.setattr("app.services.system.embedding.time.sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="PUBLIC"):
        service._request_embedding_chunk(["strict-chunk"], False)
    assert service._stats["fake_embeds"] == 0

    assert len(service._request_embedding_chunk(["next-chunk"], False)) == 1
    assert service._stats["fake_embeds"] == 1


def test_concurrent_chunks_preserve_input_order_and_stats() -> None:
    client = RecordingSafeRuntimeClient([])
    first_started = threading.Event()
    second_started = threading.Event()

    def delayed_post(base_url: str, **kwargs: Any) -> dict[str, Any]:
        text = kwargs["json_body"]["input"][0]
        client.calls.append({"base_url": base_url, **kwargs})
        if text == "first":
            first_started.set()
            assert second_started.wait(timeout=2)
        elif text == "second":
            assert first_started.wait(timeout=2)
            second_started.set()
        return {"data": [{"index": 0, "embedding": [float(len(text))]}]}

    client.post_json = delayed_post  # type: ignore[method-assign]
    service = remote_service(client)
    service.enable_concurrency = True
    service.max_parallel_requests = 2

    assert service._remote_embed_batch(
        ["first", "second", "third", "fourth"], require_real=True, batch_size=1
    ) == [[5.0], [6.0], [5.0], [6.0]]
    assert service._stats["api_calls"] == 4
    assert len(client.calls) == 4


def test_secrets_are_redacted_from_unexpected_client_failures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = RecordingSafeRuntimeClient([RuntimeError("UNDERLYING-SECRET")])
    service = EmbeddingService(
        base_url="https://BASE-SECRET.example",
        api_key="KEY-SECRET",
        model="model",
        enabled=True,
        runtime_client=client,
    )

    with pytest.raises(RuntimeError) as exc_info:
        with caplog.at_level("WARNING"):
            service._request_embedding_chunk(["TEXT-SECRET"], True)

    combined = caplog.text + str(exc_info.value)
    for secret in ("UNDERLYING-SECRET", "BASE-SECRET", "KEY-SECRET", "TEXT-SECRET"):
        assert secret not in combined


def test_malformed_response_content_is_redacted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = RecordingSafeRuntimeClient([{"data": "RESPONSE-SECRET"}])

    with pytest.raises(RuntimeError) as exc_info:
        with caplog.at_level("WARNING"):
            remote_service(client)._request_embedding_chunk(["TEXT-SECRET"], False)

    assert "RESPONSE-SECRET" not in caplog.text + str(exc_info.value)


@pytest.mark.parametrize(
    ("payload", "expected_count"),
    [
        ({"data": [{"index": 0, "embedding": [1.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": [1.0]}, {"index": 1, "embedding": [2.0]}, {"index": 2, "embedding": [3.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": [1.0]}, {"index": 2, "embedding": [2.0]}]}, 2),
        ({"data": [{"index": 0, "embedding": ["not-a-number"]}]}, 1),
    ],
)
def test_embedding_parser_rejects_complete_schema_failure_matrix(
    payload: dict[str, Any], expected_count: int,
) -> None:
    with pytest.raises(OutboundRequestError) as exc_info:
        EmbeddingService._parse_embedding_response(payload, expected_count)
    assert exc_info.value.code == "outbound_bad_response"
    assert str(exc_info.value) == "外部服务响应无效"


def test_embedding_parser_orders_valid_out_of_order_indices() -> None:
    assert EmbeddingService._parse_embedding_response(
        {"data": [{"index": 2, "embedding": [3]}, {"index": 0, "embedding": [1.5]}, {"index": 1, "embedding": [2]}]},
        3,
    ) == [[1.5], [2.0], [3.0]]


def test_embedding_module_has_no_direct_network_bypass() -> None:
    source = Path(embedding_module.__file__).read_text(encoding="utf-8")
    assert _find_forbidden_network_uses(source) == []


@pytest.mark.parametrize(
    "source",
    [
        "from urllib import request as req\nreq.urlopen('https://example.com')\n",
        "import urllib as u\nu.request.urlopen('https://example.com')\n",
    ],
)
def test_embedding_network_guard_rejects_urllib_alias_mutations(source: str) -> None:
    assert _find_forbidden_network_uses(source)
