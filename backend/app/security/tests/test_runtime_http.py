from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from ipaddress import ip_address
from typing import Any, Literal, cast
from urllib.parse import quote, urlencode

import httpcore
import httpx
import pytest

import app.security as security
import app.security.runtime_http as runtime_http
from app.security.outbound_url import (
    OutboundRequestError,
    OutboundURLPolicy,
    ValidatedOutboundURL,
)
from app.security.runtime_http import (
    AI_JSON_MAX_BYTES,
    EMBEDDING_JSON_MAX_BYTES,
    RETRYABLE_OUTBOUND_CODES,
    STREAM_EVENT_MAX_BYTES,
    STREAM_MAX_BYTES,
    RuntimeTimeouts,
    SafeRuntimeClient,
)


Mode = Literal["sync", "async"]

INVALID_TARGETS = [
    "",
    "relative",
    "//evil.example/path",
    "https://evil.example/path",
    "/ok#fragment",
    "/bad\\path",
    "/bad path",
    "/bad\npath",
    "/forward/https://evil.example",
    "/forward?next=https://evil.example",
]
VALIDATED_BASE = "https://public.example/v1"
VALID_TARGET = "/chat/completions"
EXPECTED_FINAL_URL = "https://public.example/v1/chat/completions"


def _validated() -> ValidatedOutboundURL:
    return ValidatedOutboundURL(
        url=VALIDATED_BASE,
        scheme="https",
        hostname="public.example",
        port=443,
        resolved_ips=(ip_address("93.184.216.34"),),
        is_local=False,
    )


class RecordingPolicy:
    def __init__(
        self,
        *,
        result: ValidatedOutboundURL | None = None,
        error: OutboundRequestError | None = None,
    ) -> None:
        self.result = _validated() if result is None else result
        self.error = error
        self.calls: list[tuple[object, bool]] = []

    def validate(
        self,
        url: str,
        *,
        allow_local: bool,
    ) -> ValidatedOutboundURL:
        self.calls.append((url, allow_local))
        if self.error is not None:
            raise self.error
        return self.result


_AUTO_CONTENT_LENGTH = object()


def _response_head(
    *,
    status: int = 200,
    body: bytes = b'{"ok":true}',
    headers: Mapping[str, str] | None = None,
    content_length: object = _AUTO_CONTENT_LENGTH,
) -> bytes:
    reasons = {
        200: "OK",
        201: "Created",
        204: "No Content",
        302: "Found",
        400: "Bad Request",
        500: "Internal Server Error",
    }
    lines = [f"HTTP/1.1 {status} {reasons.get(status, 'Status')}"]
    for key, value in (headers or {}).items():
        lines.append(f"{key}: {value}")
    if content_length is _AUTO_CONTENT_LENGTH:
        lines.append(f"Content-Length: {len(body)}")
    elif content_length is not None:
        lines.append(f"Content-Length: {content_length}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")


class RecordingSyncStream(httpcore.NetworkStream):
    def __init__(
        self,
        *,
        body: bytes = b'{"ok":true}',
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        content_length: object = _AUTO_CONTENT_LENGTH,
        raw_head: bytes | None = None,
        body_chunk_size: int | None = None,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
        tls_error: Exception | None = None,
    ) -> None:
        self._head = bytearray(
            raw_head
            if raw_head is not None
            else _response_head(
                status=status,
                body=body,
                headers=headers,
                content_length=content_length,
            )
        )
        self._body = bytearray(body)
        self._body_chunk_size = body_chunk_size
        self.read_error = read_error
        self.write_error = write_error
        self.tls_error = tls_error
        self.request_bytes = bytearray()
        self.body_bytes_returned = 0
        self.close_count = 0

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._head:
            chunk = bytes(self._head[:max_bytes])
            del self._head[:max_bytes]
            return chunk
        if self.read_error is not None:
            raise self.read_error
        if not self._body:
            return b""
        size = min(max_bytes, len(self._body))
        if self._body_chunk_size is not None:
            size = min(size, self._body_chunk_size)
        chunk = bytes(self._body[:size])
        del self._body[:size]
        self.body_bytes_returned += len(chunk)
        return chunk

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.request_bytes.extend(buffer)
        if self.write_error is not None:
            raise self.write_error

    def close(self) -> None:
        self.close_count += 1

    def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        if self.tls_error is not None:
            raise self.tls_error
        return self


class RecordingAsyncStream(httpcore.AsyncNetworkStream):
    def __init__(
        self,
        *,
        body: bytes = b'{"ok":true}',
        status: int = 200,
        headers: Mapping[str, str] | None = None,
        content_length: object = _AUTO_CONTENT_LENGTH,
        raw_head: bytes | None = None,
        body_chunk_size: int | None = None,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
        tls_error: Exception | None = None,
        block_body: bool = False,
    ) -> None:
        self._head = bytearray(
            raw_head
            if raw_head is not None
            else _response_head(
                status=status,
                body=body,
                headers=headers,
                content_length=content_length,
            )
        )
        self._body = bytearray(body)
        self._body_chunk_size = body_chunk_size
        self.read_error = read_error
        self.write_error = write_error
        self.tls_error = tls_error
        self.block_body = block_body
        self.request_bytes = bytearray()
        self.body_bytes_returned = 0
        self.close_count = 0
        self.body_read_started = asyncio.Event()
        self._never_ready = asyncio.Event()

    async def read(
        self,
        max_bytes: int,
        timeout: float | None = None,
    ) -> bytes:
        if self._head:
            chunk = bytes(self._head[:max_bytes])
            del self._head[:max_bytes]
            return chunk
        self.body_read_started.set()
        if self.block_body:
            await self._never_ready.wait()
        if self.read_error is not None:
            raise self.read_error
        if not self._body:
            return b""
        size = min(max_bytes, len(self._body))
        if self._body_chunk_size is not None:
            size = min(size, self._body_chunk_size)
        chunk = bytes(self._body[:size])
        del self._body[:size]
        self.body_bytes_returned += len(chunk)
        return chunk

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.request_bytes.extend(buffer)
        if self.write_error is not None:
            raise self.write_error

    async def aclose(self) -> None:
        self.close_count += 1

    async def start_tls(
        self,
        ssl_context: Any,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if self.tls_error is not None:
            raise self.tls_error
        return self


class RecordingSyncBackend(httpcore.NetworkBackend):
    def __init__(
        self,
        streams: RecordingSyncStream | list[RecordingSyncStream],
        *,
        connect_error: Exception | None = None,
    ) -> None:
        self.streams = list(streams) if isinstance(streams, list) else [streams]
        self.connect_error = connect_error
        self.connect_calls: list[tuple[str, int]] = []

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        self.connect_calls.append((host, port))
        if self.connect_error is not None:
            raise self.connect_error
        return self.streams[len(self.connect_calls) - 1]

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        raise AssertionError("Unix sockets must not be used")

    def sleep(self, seconds: float) -> None:
        return None


class RecordingAsyncBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self,
        streams: RecordingAsyncStream | list[RecordingAsyncStream],
        *,
        connect_error: Exception | None = None,
    ) -> None:
        self.streams = list(streams) if isinstance(streams, list) else [streams]
        self.connect_error = connect_error
        self.connect_calls: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connect_calls.append((host, port))
        if self.connect_error is not None:
            raise self.connect_error
        return self.streams[len(self.connect_calls) - 1]

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise AssertionError("Unix sockets must not be used")

    async def sleep(self, seconds: float) -> None:
        return None


def _runtime_client(
    mode: Mode,
    stream: RecordingSyncStream | RecordingAsyncStream,
    *,
    policy: Any | None = None,
    connect_error: Exception | None = None,
    timeouts: RuntimeTimeouts = RuntimeTimeouts(),
) -> tuple[SafeRuntimeClient, RecordingPolicy | Any, Any]:
    active_policy = RecordingPolicy() if policy is None else policy
    if mode == "sync":
        backend = RecordingSyncBackend(
            cast(RecordingSyncStream, stream),
            connect_error=connect_error,
        )
        client = SafeRuntimeClient(
            active_policy,
            sync_network_backend=backend,
            timeouts=timeouts,
        )
    else:
        backend = RecordingAsyncBackend(
            cast(RecordingAsyncStream, stream),
            connect_error=connect_error,
        )
        client = SafeRuntimeClient(
            active_policy,
            async_network_backend=backend,
            timeouts=timeouts,
        )
    return client, active_policy, backend


def _stream_for(mode: Mode, **kwargs: Any) -> Any:
    if mode == "sync":
        return RecordingSyncStream(**kwargs)
    return RecordingAsyncStream(**kwargs)


async def _invoke(
    mode: Mode,
    client: SafeRuntimeClient,
    *,
    base_url: str = VALIDATED_BASE,
    request_target: Any = VALID_TARGET,
    headers: Mapping[str, str] | None = None,
    json_body: Mapping[str, Any] | None = None,
    allow_local: bool = False,
    read_timeout: float = 7.5,
    max_bytes: int = AI_JSON_MAX_BYTES,
) -> dict[str, Any]:
    kwargs = {
        "request_target": request_target,
        "headers": {} if headers is None else headers,
        "json_body": {"request": True} if json_body is None else json_body,
        "allow_local": allow_local,
        "read_timeout": read_timeout,
        "max_bytes": max_bytes,
    }
    if mode == "sync":
        return client.post_json(base_url, **kwargs)
    return await client.apost_json(base_url, **kwargs)


def _request_bytes(stream: Any) -> bytes:
    return bytes(stream.request_bytes)


def _record_parser_buffer_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> list[int]:
    lengths: list[int] = []

    class RecordingParserBuffer(bytearray):
        def extend(self, data: Any) -> None:
            super().extend(data)
            lengths.append(len(self))

    monkeypatch.setattr(
        runtime_http,
        "bytearray",
        RecordingParserBuffer,
        raising=False,
    )
    return lengths


def test_public_runtime_contract_is_exported() -> None:
    assert AI_JSON_MAX_BYTES == 8 * 1024 * 1024
    assert EMBEDDING_JSON_MAX_BYTES == 16 * 1024 * 1024
    assert STREAM_MAX_BYTES == 16 * 1024 * 1024
    assert STREAM_EVENT_MAX_BYTES == 1 * 1024 * 1024
    assert RETRYABLE_OUTBOUND_CODES == frozenset(
        {"outbound_connect_failed", "outbound_timeout"}
    )
    assert RuntimeTimeouts() == RuntimeTimeouts(
        connect=10.0,
        read=60.0,
        write=30.0,
        pool=10.0,
        stream_idle=120.0,
        stream_total=600.0,
    )
    with pytest.raises(FrozenInstanceError):
        RuntimeTimeouts().connect = 99.0  # type: ignore[misc]

    for name in (
        "AI_JSON_MAX_BYTES",
        "EMBEDDING_JSON_MAX_BYTES",
        "STREAM_MAX_BYTES",
        "STREAM_EVENT_MAX_BYTES",
        "RETRYABLE_OUTBOUND_CODES",
        "RuntimeTimeouts",
        "SafeRuntimeClient",
    ):
        assert getattr(security, name) is getattr(runtime_http, name)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("request_target", INVALID_TARGETS)
@pytest.mark.asyncio
async def test_request_target_rejects_origin_escape_and_ambiguous_forms(
    mode: Mode,
    request_target: str,
) -> None:
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, request_target=request_target)

    assert exc_info.value.code == "outbound_url_invalid"
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("request_target", [None, b"/bytes", 7])
@pytest.mark.asyncio
async def test_request_target_rejects_non_strings(
    mode: Mode,
    request_target: Any,
) -> None:
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, request_target=request_target)

    assert exc_info.value.code == "outbound_url_invalid"
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_request_target_rejects_unpaired_unicode_surrogate(
    mode: Mode,
) -> None:
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, request_target="/\ud800")

    assert exc_info.value.code == "outbound_url_invalid"
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("suffix", ["?key=query-secret", "#fragment-secret"])
@pytest.mark.asyncio
async def test_base_url_query_and_fragment_still_use_real_policy(
    mode: Mode,
    suffix: str,
) -> None:
    resolver_calls: list[tuple[str, int]] = []

    def resolver(hostname: str, port: int) -> tuple[Any, ...]:
        resolver_calls.append((hostname, port))
        return (ip_address("93.184.216.34"),)

    policy = OutboundURLPolicy(resolver)
    stream = _stream_for(mode)
    client, _, backend = _runtime_client(mode, stream, policy=policy)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, base_url=VALIDATED_BASE + suffix)

    assert exc_info.value.code == "outbound_url_invalid"
    assert resolver_calls == []
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_request_target_preserves_validated_base_path_without_urljoin(
    mode: Mode,
) -> None:
    stream = _stream_for(mode)
    client, policy, _ = _runtime_client(mode, stream)

    assert await _invoke(mode, client, allow_local=True) == {"ok": True}

    target = EXPECTED_FINAL_URL.removeprefix("https://public.example")
    request_line = f"POST {target} HTTP/1.1"
    assert request_line.encode() in _request_bytes(stream)
    assert policy.calls == [(VALIDATED_BASE, True)]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_encoded_google_target_keeps_approved_origin_and_hides_query_log(
    mode: Mode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    query_sentinel = "google-query-secret-sentinel"
    model_segment = quote("gemini/2?preview", safe="")
    target = (
        f"/models/{model_segment}:generateContent?"
        + urlencode({"key": query_sentinel})
    )
    stream = _stream_for(mode)
    client, _, backend = _runtime_client(mode, stream)
    for logger_name in ("httpx", "httpcore.connection", "httpcore.http11"):
        caplog.set_level(logging.DEBUG, logger=logger_name)

    assert await _invoke(mode, client, request_target=target) == {"ok": True}

    request = _request_bytes(stream)
    assert (
        b"POST /v1/models/gemini%2F2%3Fpreview:generateContent?key="
        + query_sentinel.encode()
        + b" HTTP/1.1"
    ) in request
    assert b"host: public.example\r\n" in request.lower()
    assert backend.connect_calls == [("93.184.216.34", 443)]
    assert query_sentinel not in caplog.text


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"answer":42}', {"answer": 42}),
        (b"[]", None),
        (b"null", None),
        (b'"value"', None),
        (b"{", None),
        (b"\xff", None),
    ],
    ids=["object", "array", "null", "string", "malformed", "non-utf8"],
)
@pytest.mark.asyncio
async def test_json_response_must_be_a_valid_object(
    mode: Mode,
    body: bytes,
    expected: dict[str, Any] | None,
) -> None:
    stream = _stream_for(mode, body=body)
    client, _, _ = _runtime_client(mode, stream)

    if expected is not None:
        assert await _invoke(mode, client) == expected
    else:
        with pytest.raises(OutboundRequestError) as exc_info:
            await _invoke(mode, client)
        assert exc_info.value.code == "outbound_bad_response"
    assert stream.close_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "body",
    [
        b'{"number":' + (b"9" * 5_000) + b"}",
        b'{"nested":' + (b"[" * 20_000) + b"0" + (b"]" * 20_000) + b"}",
    ],
    ids=["integer-digit-limit", "recursion-limit"],
)
@pytest.mark.asyncio
async def test_json_parser_limits_map_to_fixed_bad_response(
    mode: Mode,
    body: bytes,
) -> None:
    stream = _stream_for(mode, body=body)
    client, _, _ = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client)

    assert exc_info.value.code == "outbound_bad_response"
    assert str(exc_info.value) == "外部服务响应无效"
    assert stream.close_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "case",
    ["header-unicode", "body-unicode", "unserializable", "nan", "recursive"],
)
@pytest.mark.asyncio
async def test_request_construction_errors_are_fixed_and_redacted(
    mode: Mode,
    case: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    header_sentinel = "header-construction-secret-sentinel"
    body_sentinel = "body-construction-secret-sentinel"
    repr_sentinel = "repr-construction-secret-sentinel"
    value_sentinel = f"value-{case}-secret-sentinel"
    headers: Mapping[str, str] = {}
    json_body: Mapping[str, Any] = {"ok": True}

    if case == "header-unicode":
        headers = {"X-Secret": header_sentinel + "\ud800"}
        active_sentinel = header_sentinel
    elif case == "body-unicode":
        json_body = {"prompt": body_sentinel + "\ud800"}
        active_sentinel = body_sentinel
    elif case == "unserializable":
        secret_type = type(
            repr_sentinel,
            (),
            {"__repr__": lambda self: repr_sentinel},
        )
        json_body = {"value": secret_type()}
        active_sentinel = repr_sentinel
    elif case == "nan":
        json_body = {value_sentinel: float("nan")}
        active_sentinel = value_sentinel
    else:
        nested: Any = 0
        for _ in range(20_000):
            nested = [nested]
        json_body = {value_sentinel: nested}
        active_sentinel = value_sentinel

    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)
    for logger_name in (
        "httpx",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        caplog.set_level(logging.DEBUG, logger=logger_name)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(
            mode,
            client,
            headers=headers,
            json_body=json_body,
        )

    assert exc_info.value.code == "outbound_bad_response"
    assert str(exc_info.value) == "外部服务响应无效"
    assert active_sentinel not in repr(exc_info.value)
    assert active_sentinel not in caplog.text
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [302, 400, 500])
@pytest.mark.asyncio
async def test_non_2xx_reads_zero_body_bytes_and_never_follows_redirects(
    mode: Mode,
    status: int,
) -> None:
    body = b"upstream-body-secret"
    stream = _stream_for(
        mode,
        status=status,
        body=body,
        headers={
            "Location": "https://evil.example/redirect-secret",
            "Content-Encoding": "gzip",
        },
        content_length=len(body) + 1_000,
    )
    client, _, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=8)

    assert exc_info.value.code == "outbound_bad_response"
    assert stream.body_bytes_returned == 0
    assert backend.connect_calls == [("93.184.216.34", 443)]
    assert b"evil.example" not in _request_bytes(stream)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("encoding", "expected_code"),
    [
        ("gzip", "outbound_bad_response"),
        ("br", "outbound_bad_response"),
        ("", "outbound_bad_response"),
        ("identity", None),
        (None, None),
    ],
    ids=["gzip", "br", "empty", "identity", "missing"],
)
@pytest.mark.asyncio
async def test_content_encoding_is_checked_before_body_read(
    mode: Mode,
    encoding: str | None,
    expected_code: str | None,
) -> None:
    body = b'{"ok":true}'
    headers = {} if encoding is None else {"Content-Encoding": encoding}
    stream = _stream_for(mode, body=body, headers=headers)
    client, _, _ = _runtime_client(mode, stream)

    if expected_code is None:
        assert await _invoke(mode, client) == {"ok": True}
        assert stream.body_bytes_returned == len(body)
    else:
        with pytest.raises(OutboundRequestError) as exc_info:
            await _invoke(mode, client)
        assert exc_info.value.code == expected_code
        assert stream.body_bytes_returned == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_content_length_equal_to_limit_is_allowed(mode: Mode) -> None:
    body = b'{"value":"ok"}'
    stream = _stream_for(mode, body=body, content_length=len(body))
    client, _, _ = _runtime_client(mode, stream)

    assert await _invoke(mode, client, max_bytes=len(body)) == {"value": "ok"}
    assert stream.body_bytes_returned == len(body)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("declared_length", "expected_code"),
    [
        (9, "outbound_response_too_large"),
        (-1, "outbound_bad_response"),
        ("not-an-integer", "outbound_bad_response"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_or_oversized_content_length_reads_zero_body_bytes(
    mode: Mode,
    declared_length: object,
    expected_code: str,
) -> None:
    stream = _stream_for(
        mode,
        body=b'{"ok":true}',
        content_length=declared_length,
    )
    client, _, _ = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=8)

    assert exc_info.value.code == expected_code
    assert stream.body_bytes_returned == 0


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_missing_content_length_incremental_body_exceeds_parser_limit(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 17
    parser_buffer_lengths = _record_parser_buffer_lengths(monkeypatch)
    stream = _stream_for(
        mode,
        body=b"x" * (limit + 100),
        content_length=None,
        body_chunk_size=1,
    )
    client, _, _ = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=limit)

    assert exc_info.value.code == "outbound_response_too_large"
    assert parser_buffer_lengths
    assert max(parser_buffer_lengths) == limit + 1
    assert stream.body_bytes_returned >= limit + 1
    assert stream.close_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_coalesced_socket_read_still_enforces_parser_limit(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit = 17
    body = b"x" * (limit + 100)
    parser_buffer_lengths = _record_parser_buffer_lengths(monkeypatch)
    stream = _stream_for(
        mode,
        body=body,
        content_length=None,
    )
    client, _, _ = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=limit)

    assert exc_info.value.code == "outbound_response_too_large"
    assert parser_buffer_lengths
    assert max(parser_buffer_lengths) == limit + 1
    # This counter observes the fake TCP stream. HTTPcore may read ahead into
    # its protocol buffer; only bytes handed to the runtime parser are bounded.
    assert stream.body_bytes_returned > limit + 1
    assert stream.close_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_negative_max_bytes_is_rejected_before_network(mode: Mode) -> None:
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=-1)

    assert exc_info.value.code == "outbound_url_invalid"
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "max_bytes",
    [True, 1.5, float("inf"), float("nan"), "8"],
    ids=["bool", "float", "infinity", "nan", "string"],
)
@pytest.mark.asyncio
async def test_invalid_max_bytes_type_is_rejected_before_network(
    mode: Mode,
    max_bytes: Any,
) -> None:
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client, max_bytes=max_bytes)

    assert exc_info.value.code == "outbound_url_invalid"
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


def _error_stream(mode: Mode, stage: str, error: Exception) -> Any:
    kwargs: dict[str, Any] = {}
    if stage == "read":
        kwargs["read_error"] = error
    elif stage == "write":
        kwargs["write_error"] = error
    elif stage == "tls":
        kwargs["tls_error"] = error
    elif stage == "protocol":
        kwargs["raw_head"] = b"not-an-http-response\r\n\r\n"
        kwargs["body"] = b""
        kwargs["content_length"] = None
    return _stream_for(mode, **kwargs)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    ("stage", "error", "expected_code"),
    [
        ("connect", OSError("connect-backend-secret"), "outbound_connect_failed"),
        ("read", OSError("read-backend-secret"), "outbound_connect_failed"),
        ("tls", OSError("tls-backend-secret"), "outbound_connect_failed"),
        ("read", httpcore.ReadTimeout("read-timeout-secret"), "outbound_timeout"),
        ("write", httpcore.WriteTimeout("write-timeout-secret"), "outbound_timeout"),
        ("tls", httpcore.ConnectTimeout("tls-timeout-secret"), "outbound_timeout"),
        ("protocol", RuntimeError("unused"), "outbound_bad_response"),
    ],
    ids=[
        "connect",
        "read",
        "tls",
        "read-timeout",
        "write-timeout",
        "tls-timeout",
        "protocol",
    ],
)
@pytest.mark.asyncio
async def test_transport_errors_map_to_fixed_public_errors(
    mode: Mode,
    stage: str,
    error: Exception,
    expected_code: str,
) -> None:
    stream = _error_stream(mode, stage, error)
    connect_error = error if stage == "connect" else None
    client, policy, backend = _runtime_client(
        mode,
        stream,
        connect_error=connect_error,
    )

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client)

    assert exc_info.value.code == expected_code
    assert str(exc_info.value) in {
        "外部服务连接失败",
        "外部服务请求超时",
        "外部服务响应无效",
    }
    assert "secret" not in repr(exc_info.value)
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert len(backend.connect_calls) == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_client_cleans_headers_and_disables_env_and_redirects(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = _stream_for(mode)
    timeouts = RuntimeTimeouts(
        connect=1.25,
        read=2.5,
        write=3.75,
        pool=4.5,
        stream_idle=5.0,
        stream_total=6.0,
    )
    client, _, _ = _runtime_client(mode, stream, timeouts=timeouts)
    constructor_kwargs: list[dict[str, Any]] = []

    if mode == "sync":
        real_client = httpx.Client

        def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
            constructor_kwargs.append(kwargs.copy())
            return real_client(*args, **kwargs)

        monkeypatch.setattr(runtime_http.httpx, "Client", client_factory)
    else:
        real_async_client = httpx.AsyncClient

        def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            constructor_kwargs.append(kwargs.copy())
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(
            runtime_http.httpx,
            "AsyncClient",
            async_client_factory,
        )

    result = await _invoke(
        mode,
        client,
        headers={
            "hOsT": "evil-host-secret",
            "aCcEpT-eNcOdInG": "gzip",
            "Authorization": "Bearer caller-header-secret",
        },
        read_timeout=9.25,
    )

    assert result == {"ok": True}
    request = _request_bytes(stream).lower()
    assert b"host: public.example\r\n" in request
    assert b"evil-host-secret" not in request
    assert request.count(b"accept-encoding: identity\r\n") == 1
    assert b"accept-encoding: gzip" not in request
    assert len(constructor_kwargs) == 1
    assert constructor_kwargs[0]["trust_env"] is False
    assert constructor_kwargs[0]["follow_redirects"] is False
    timeout = constructor_kwargs[0]["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (
        1.25,
        9.25,
        3.75,
        4.5,
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_client_constructor_failure_closes_new_pinned_transport(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructor_sentinel = "client-constructor-secret"
    stream = _stream_for(mode)
    client, policy, backend = _runtime_client(mode, stream)
    created_transports: list[Any] = []

    if mode == "sync":
        real_transport = runtime_http.PinnedSyncTransport

        class TrackingSyncTransport(real_transport):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.close_count = 0
                created_transports.append(self)

            def close(self) -> None:
                self.close_count += 1
                super().close()

        def failing_client(*args: Any, **kwargs: Any) -> None:
            raise OSError(constructor_sentinel)

        monkeypatch.setattr(
            runtime_http,
            "PinnedSyncTransport",
            TrackingSyncTransport,
        )
        monkeypatch.setattr(runtime_http.httpx, "Client", failing_client)
    else:
        real_async_transport = runtime_http.PinnedAsyncTransport

        class TrackingAsyncTransport(real_async_transport):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.close_count = 0
                created_transports.append(self)

            async def aclose(self) -> None:
                self.close_count += 1
                await super().aclose()

        def failing_async_client(*args: Any, **kwargs: Any) -> None:
            raise OSError(constructor_sentinel)

        monkeypatch.setattr(
            runtime_http,
            "PinnedAsyncTransport",
            TrackingAsyncTransport,
        )
        monkeypatch.setattr(
            runtime_http.httpx,
            "AsyncClient",
            failing_async_client,
        )

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client)

    assert exc_info.value.code == "outbound_connect_failed"
    assert constructor_sentinel not in repr(exc_info.value)
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []
    assert len(created_transports) == 1
    assert created_transports[0].close_count == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_errors_and_logs_redact_all_independent_sentinels(
    mode: Mode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sentinels = {
        "header": "header-secret-sentinel",
        "query": "query-secret-sentinel",
        "body": "body-secret-sentinel",
        "backend": "backend-secret-sentinel",
    }
    stream = _stream_for(
        mode,
        read_error=OSError(sentinels["backend"]),
        body=b'{"unread":true}',
    )
    client, _, _ = _runtime_client(mode, stream)
    for logger_name in (
        "httpx",
        "httpcore.connection",
        "httpcore.http11",
        "httpcore.http2",
        "httpcore.proxy",
        "httpcore.socks",
    ):
        caplog.set_level(logging.DEBUG, logger=logger_name)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(
            mode,
            client,
            request_target=VALID_TARGET + "?" + urlencode({"key": sentinels["query"]}),
            headers={"Authorization": "Bearer " + sentinels["header"]},
            json_body={"prompt": sentinels["body"]},
        )

    wire_request = _request_bytes(stream).decode("utf-8")
    assert sentinels["header"] in wire_request
    assert sentinels["query"] in wire_request
    assert sentinels["body"] in wire_request
    for sentinel in sentinels.values():
        assert sentinel not in repr(exc_info.value)
        assert sentinel not in caplog.text

    outside_sentinel = f"outside-{mode}-log-is-visible"
    logging.getLogger("httpx").info(outside_sentinel)
    assert outside_sentinel in caplog.text


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_existing_outbound_error_passes_through_unchanged(mode: Mode) -> None:
    original = OutboundRequestError(
        "private_network_blocked",
        400,
        "该网络地址不允许访问",
    )
    policy = RecordingPolicy(error=original)
    stream = _stream_for(mode)
    client, _, backend = _runtime_client(mode, stream, policy=policy)

    with pytest.raises(OutboundRequestError) as exc_info:
        await _invoke(mode, client)

    assert exc_info.value is original
    assert policy.calls == [(VALIDATED_BASE, False)]
    assert backend.connect_calls == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_real_policy_resolves_exactly_once_per_invocation(mode: Mode) -> None:
    resolver_calls: list[tuple[str, int]] = []

    def resolver(hostname: str, port: int) -> tuple[Any, ...]:
        resolver_calls.append((hostname, port))
        return (ip_address("93.184.216.34"),)

    stream = _stream_for(mode)
    policy = OutboundURLPolicy(resolver)
    client, _, backend = _runtime_client(mode, stream, policy=policy)

    assert await _invoke(mode, client) == {"ok": True}
    assert resolver_calls == [("public.example", 443)]
    assert backend.connect_calls == [("93.184.216.34", 443)]


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.asyncio
async def test_each_invocation_validates_once_and_builds_a_new_client(
    mode: Mode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RecordingPolicy()
    constructor_count = 0
    if mode == "sync":
        streams: list[Any] = [RecordingSyncStream(), RecordingSyncStream()]
        backend: Any = RecordingSyncBackend(streams)
        client = SafeRuntimeClient(policy, sync_network_backend=backend)
        real_client = httpx.Client

        def client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
            nonlocal constructor_count
            constructor_count += 1
            return real_client(*args, **kwargs)

        monkeypatch.setattr(runtime_http.httpx, "Client", client_factory)
    else:
        streams = [RecordingAsyncStream(), RecordingAsyncStream()]
        backend = RecordingAsyncBackend(streams)
        client = SafeRuntimeClient(policy, async_network_backend=backend)
        real_async_client = httpx.AsyncClient

        def async_client_factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
            nonlocal constructor_count
            constructor_count += 1
            return real_async_client(*args, **kwargs)

        monkeypatch.setattr(
            runtime_http.httpx,
            "AsyncClient",
            async_client_factory,
        )

    assert await _invoke(mode, client) == {"ok": True}
    assert await _invoke(mode, client) == {"ok": True}

    assert policy.calls == [(VALIDATED_BASE, False), (VALIDATED_BASE, False)]
    assert constructor_count == 2
    assert len(backend.connect_calls) == 2
    assert all(stream.close_count == 1 for stream in streams)


@pytest.mark.asyncio
async def test_async_cancellation_propagates_closes_and_keeps_other_task_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    query_sentinel = "cancelled-runtime-query-secret"
    outside_sentinel = "concurrent-unrelated-httpx-log"
    stream = RecordingAsyncStream(
        body=b'{"ok":true}',
        block_body=True,
    )
    client, _, _ = _runtime_client("async", stream)
    caplog.set_level(logging.DEBUG, logger="httpx")
    caplog.set_level(logging.DEBUG, logger="httpcore.http11")
    task = asyncio.create_task(
        _invoke(
            "async",
            client,
            request_target=VALID_TARGET + "?" + urlencode({"key": query_sentinel}),
        )
    )

    await stream.body_read_started.wait()
    logging.getLogger("httpx").info(outside_sentinel)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.close_count == 1
    assert outside_sentinel in caplog.text
    assert query_sentinel not in caplog.text
