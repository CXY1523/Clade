from __future__ import annotations

import asyncio
import ssl
from collections.abc import AsyncIterator
from ipaddress import ip_address
from types import SimpleNamespace
from typing import Any

import httpcore
import httpx
import pytest

import app.security.pinned_transport as pinned_transport
from app.security.outbound_url import ValidatedOutboundURL
from app.security.pinned_transport import PinnedAsyncTransport, PinnedSyncTransport


class RecordingStream(httpcore.NetworkStream):
    def __init__(
        self,
        *,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
        tls_error: Exception | None = None,
    ) -> None:
        self._response = bytearray(
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        self.read_error = read_error
        self.write_error = write_error
        self.tls_error = tls_error
        self.request_bytes = bytearray()
        self.tls_server_names: list[str | None] = []
        self.close_count = 0

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        chunk = bytes(self._response[:max_bytes])
        del self._response[:max_bytes]
        return chunk

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.request_bytes.extend(buffer)
        if self.write_error is not None:
            raise self.write_error

    def close(self) -> None:
        self.close_count += 1

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        self.tls_server_names.append(server_hostname)
        if self.tls_error is not None:
            raise self.tls_error
        return self


class RecordingBackend(httpcore.NetworkBackend):
    def __init__(
        self,
        stream: RecordingStream,
        *,
        connect_errors: list[Exception] | None = None,
    ) -> None:
        self.stream = stream
        self.connect_errors = list(connect_errors or [])
        self.connect_calls: list[
            tuple[str, int, float | None, str | None, Any]
        ] = []
        self.unix_calls = 0

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        self.connect_calls.append(
            (host, port, timeout, local_address, socket_options)
        )
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        return self.stream

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        self.unix_calls += 1
        return self.stream

    def sleep(self, seconds: float) -> None:
        return None


class RecordingAsyncStream(httpcore.AsyncNetworkStream):
    def __init__(
        self,
        *,
        read_error: Exception | None = None,
        write_error: Exception | None = None,
        tls_error: Exception | None = None,
    ) -> None:
        self._response = bytearray(
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
        )
        self.read_error = read_error
        self.write_error = write_error
        self.tls_error = tls_error
        self.request_bytes = bytearray()
        self.tls_server_names: list[str | None] = []
        self.close_count = 0

    async def read(
        self, max_bytes: int, timeout: float | None = None
    ) -> bytes:
        if self.read_error is not None:
            raise self.read_error
        chunk = bytes(self._response[:max_bytes])
        del self._response[:max_bytes]
        return chunk

    async def write(
        self, buffer: bytes, timeout: float | None = None
    ) -> None:
        self.request_bytes.extend(buffer)
        if self.write_error is not None:
            raise self.write_error

    async def aclose(self) -> None:
        self.close_count += 1

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.tls_server_names.append(server_hostname)
        if self.tls_error is not None:
            raise self.tls_error
        return self


class RecordingAsyncBackend(httpcore.AsyncNetworkBackend):
    def __init__(
        self,
        stream: RecordingAsyncStream,
        *,
        connect_errors: list[Exception] | None = None,
    ) -> None:
        self.stream = stream
        self.connect_errors = list(connect_errors or [])
        self.connect_calls: list[
            tuple[str, int, float | None, str | None, Any]
        ] = []
        self.unix_calls = 0

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connect_calls.append(
            (host, port, timeout, local_address, socket_options)
        )
        if self.connect_errors:
            raise self.connect_errors.pop(0)
        return self.stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        self.unix_calls += 1
        return self.stream

    async def sleep(self, seconds: float) -> None:
        return None


class RecordingResponseStream:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.close_count = 0

    def __iter__(self):
        if self.error is not None:
            raise self.error
        yield b"response"

    def close(self) -> None:
        self.close_count += 1


class RecordingAsyncResponseStream:
    def __init__(
        self,
        error: Exception | None = None,
        *,
        wait_for_cancellation: bool = False,
    ) -> None:
        self.error = error
        self.wait_for_cancellation = wait_for_cancellation
        self.read_started = asyncio.Event()
        self._never_ready = asyncio.Event()
        self.close_count = 0

    async def read(self) -> bytes:
        self.read_started.set()
        if self.wait_for_cancellation:
            await self._never_ready.wait()
        if self.error is not None:
            raise self.error
        return b"response"

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield await self.read()

    async def aclose(self) -> None:
        self.close_count += 1


def _validated(
    *resolved_ips: str,
    hostname: str = "public.example",
    port: int = 443,
) -> ValidatedOutboundURL:
    return ValidatedOutboundURL(
        url=f"https://{hostname}/v1",
        scheme="https",
        hostname=hostname,
        port=port,
        resolved_ips=tuple(ip_address(value) for value in resolved_ips),
        is_local=False,
    )


def _request(url: str = "https://public.example/v1/models") -> httpx.Request:
    return httpx.Request("GET", url)


def _close_response_and_transport(
    response: httpx.Response,
    transport: PinnedSyncTransport,
) -> None:
    response.close()
    transport.close()


async def _aclose_response_and_transport(
    response: httpx.Response,
    transport: PinnedAsyncTransport,
) -> None:
    await response.aclose()
    await transport.aclose()


def test_sync_transport_tries_only_approved_ips_in_order() -> None:
    stream = RecordingStream()
    backend = RecordingBackend(
        stream,
        connect_errors=[httpcore.ConnectError("first-IP-secret-sentinel")],
    )
    transport = PinnedSyncTransport(
        _validated("93.184.216.34", "8.8.8.8"),
        backend,
    )

    response = transport.handle_request(_request())
    _close_response_and_transport(response, transport)

    assert [call[0] for call in backend.connect_calls] == [
        "93.184.216.34",
        "8.8.8.8",
    ]


@pytest.mark.parametrize(
    "request_url",
    [
        "https://attacker.example/v1/models",
        "https://public.example:444/v1/models",
    ],
    ids=["host", "port"],
)
def test_sync_transport_rejects_request_origin_mismatch(
    request_url: str,
) -> None:
    backend = RecordingBackend(RecordingStream())
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)

    try:
        with pytest.raises(httpcore.ConnectError) as exc_info:
            transport.handle_request(_request(request_url))
    finally:
        transport.close()

    assert str(exc_info.value) == "outbound connect failed"
    assert backend.connect_calls == []


def test_sync_transport_rejects_request_scheme_mismatch() -> None:
    stream = RecordingStream()
    backend = RecordingBackend(stream)
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)

    try:
        with pytest.raises(httpcore.ConnectError) as exc_info:
            transport.handle_request(
                _request("http://public.example:443/v1/models")
            )
    finally:
        transport.close()

    assert str(exc_info.value) == "outbound origin does not match approval"
    assert backend.connect_calls == []
    assert stream.request_bytes == b""


def test_sync_transport_rejects_unix_socket() -> None:
    backend = RecordingBackend(RecordingStream())
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)
    network_backend = transport._pool._network_backend

    try:
        with pytest.raises(httpcore.ConnectError) as exc_info:
            network_backend.connect_unix_socket("/tmp/secret.sock")
    finally:
        transport.close()

    assert str(exc_info.value) == "unix sockets are not allowed"
    assert backend.unix_calls == 0


def test_sync_transport_keeps_original_host_and_tls_server_name() -> None:
    stream = RecordingStream()
    backend = RecordingBackend(stream)
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)

    response = transport.handle_request(_request())
    _close_response_and_transport(response, transport)

    assert b"Host: public.example\r\n" in stream.request_bytes
    assert stream.tls_server_names == ["public.example"]
    assert [call[0] for call in backend.connect_calls] == ["93.184.216.34"]


def test_sync_transport_ignores_sni_override_and_preserves_timeout() -> None:
    sentinel = "sni-override-secret-sentinel"
    stream = RecordingStream()
    backend = RecordingBackend(stream)
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)
    request = httpx.Request(
        "GET",
        "https://public.example/v1/models",
        extensions={
            "sni_hostname": sentinel,
            "timeout": {"connect": 3.25},
        },
    )

    response = transport.handle_request(request)
    _close_response_and_transport(response, transport)

    assert stream.tls_server_names == ["public.example"]
    assert sentinel not in repr(stream.tls_server_names)
    assert backend.connect_calls == [("93.184.216.34", 443, 3.25, None, None)]
    assert request.extensions["sni_hostname"] == sentinel
    assert request.extensions["timeout"] == {"connect": 3.25}


@pytest.mark.parametrize(
    ("stage", "expected_type", "expected_message"),
    [
        ("connect", httpcore.ConnectError, "outbound connect failed"),
        ("read", httpcore.ReadError, "outbound read failed"),
        ("write", httpcore.WriteError, "outbound write failed"),
        ("tls", httpcore.ConnectError, "outbound TLS failed"),
    ],
)
def test_sync_transport_sanitizes_connect_read_write_and_tls_errors(
    stage: str,
    expected_type: type[Exception],
    expected_message: str,
) -> None:
    sentinel = f"{stage}-secret-sentinel"
    stream = RecordingStream(
        read_error=OSError(sentinel) if stage == "read" else None,
        write_error=OSError(sentinel) if stage == "write" else None,
        tls_error=OSError(sentinel) if stage == "tls" else None,
    )
    backend = RecordingBackend(
        stream,
        connect_errors=[OSError(sentinel)] if stage == "connect" else None,
    )
    transport = PinnedSyncTransport(_validated("93.184.216.34"), backend)
    network_backend = transport._pool._network_backend

    try:
        with pytest.raises(expected_type) as exc_info:
            if stage == "connect":
                network_backend.connect_tcp("public.example", 443)
            else:
                network_stream = network_backend.connect_tcp(
                    "public.example", 443
                )
                if stage == "read":
                    network_stream.read(1)
                elif stage == "write":
                    network_stream.write(b"request")
                else:
                    network_stream.start_tls(
                        ssl.create_default_context(),
                        server_hostname="public.example",
                    )
    finally:
        transport.close()

    assert str(exc_info.value) == expected_message
    assert sentinel not in repr(exc_info.value)


def test_sync_transport_closes_response_and_pool_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools: list[Any] = []
    response_streams: list[RecordingResponseStream] = []

    class RecordingPool:
        def __init__(self, **kwargs: Any) -> None:
            self.close_count = 0
            pools.append(self)

        def handle_request(self, request: httpcore.Request) -> Any:
            stream = response_streams[-1]
            return SimpleNamespace(
                status=200,
                headers=[],
                stream=stream,
                extensions={},
            )

        def close(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(pinned_transport.httpcore, "ConnectionPool", RecordingPool)

    for error in (None, httpcore.ReadError("fixed response read failure")):
        response_stream = RecordingResponseStream(error)
        response_streams.append(response_stream)
        transport = PinnedSyncTransport(
            _validated("93.184.216.34"),
            RecordingBackend(RecordingStream()),
        )
        response = transport.handle_request(_request())

        try:
            if error is None:
                assert response.read() == b"response"
            else:
                with pytest.raises(httpcore.ReadError) as exc_info:
                    response.read()
                assert str(exc_info.value) == "fixed response read failure"
        finally:
            response.close()
            transport.close()

        assert response_stream.close_count == 1
        assert pools[-1].close_count == 1


@pytest.mark.asyncio
async def test_async_transport_tries_only_approved_ips_in_order() -> None:
    stream = RecordingAsyncStream()
    backend = RecordingAsyncBackend(
        stream,
        connect_errors=[httpcore.ConnectError("first-IP-secret-sentinel")],
    )
    transport = PinnedAsyncTransport(
        _validated("93.184.216.34", "8.8.8.8"),
        backend,
    )

    response = await transport.handle_async_request(_request())
    await _aclose_response_and_transport(response, transport)

    assert [call[0] for call in backend.connect_calls] == [
        "93.184.216.34",
        "8.8.8.8",
    ]


@pytest.mark.asyncio
async def test_async_transport_keeps_original_host_and_tls_server_name() -> None:
    stream = RecordingAsyncStream()
    backend = RecordingAsyncBackend(stream)
    transport = PinnedAsyncTransport(_validated("93.184.216.34"), backend)

    response = await transport.handle_async_request(_request())
    await _aclose_response_and_transport(response, transport)

    assert b"Host: public.example\r\n" in stream.request_bytes
    assert stream.tls_server_names == ["public.example"]
    assert [call[0] for call in backend.connect_calls] == ["93.184.216.34"]


@pytest.mark.parametrize(
    "request_url",
    [
        "https://attacker.example/v1/models",
        "https://public.example:444/v1/models",
    ],
    ids=["host", "port"],
)
@pytest.mark.asyncio
async def test_async_transport_rejects_origin_mismatch_and_unix_socket(
    request_url: str,
) -> None:
    backend = RecordingAsyncBackend(RecordingAsyncStream())
    transport = PinnedAsyncTransport(_validated("93.184.216.34"), backend)
    network_backend = transport._pool._network_backend

    try:
        with pytest.raises(httpcore.ConnectError) as origin_exc_info:
            await transport.handle_async_request(_request(request_url))
        with pytest.raises(httpcore.ConnectError) as unix_exc_info:
            await network_backend.connect_unix_socket("/tmp/secret.sock")
    finally:
        await transport.aclose()

    assert str(origin_exc_info.value) == "outbound connect failed"
    assert str(unix_exc_info.value) == "unix sockets are not allowed"
    assert backend.connect_calls == []
    assert backend.unix_calls == 0


@pytest.mark.asyncio
async def test_async_transport_rejects_request_scheme_mismatch() -> None:
    stream = RecordingAsyncStream()
    backend = RecordingAsyncBackend(stream)
    transport = PinnedAsyncTransport(_validated("93.184.216.34"), backend)

    try:
        with pytest.raises(httpcore.ConnectError) as exc_info:
            await transport.handle_async_request(
                _request("http://public.example:443/v1/models")
            )
    finally:
        await transport.aclose()

    assert str(exc_info.value) == "outbound origin does not match approval"
    assert backend.connect_calls == []
    assert stream.request_bytes == b""


@pytest.mark.asyncio
async def test_async_transport_ignores_sni_override_and_preserves_timeout() -> None:
    sentinel = "sni-override-secret-sentinel"
    stream = RecordingAsyncStream()
    backend = RecordingAsyncBackend(stream)
    transport = PinnedAsyncTransport(_validated("93.184.216.34"), backend)
    request = httpx.Request(
        "GET",
        "https://public.example/v1/models",
        extensions={
            "sni_hostname": sentinel,
            "timeout": {"connect": 3.25},
        },
    )

    response = await transport.handle_async_request(request)
    await _aclose_response_and_transport(response, transport)

    assert stream.tls_server_names == ["public.example"]
    assert sentinel not in repr(stream.tls_server_names)
    assert backend.connect_calls == [("93.184.216.34", 443, 3.25, None, None)]
    assert request.extensions["sni_hostname"] == sentinel
    assert request.extensions["timeout"] == {"connect": 3.25}


@pytest.mark.parametrize(
    ("stage", "expected_type", "expected_message"),
    [
        ("connect", httpcore.ConnectError, "outbound connect failed"),
        ("read", httpcore.ReadError, "outbound read failed"),
        ("write", httpcore.WriteError, "outbound write failed"),
        ("tls", httpcore.ConnectError, "outbound TLS failed"),
    ],
)
@pytest.mark.asyncio
async def test_async_transport_sanitizes_backend_error_text(
    stage: str,
    expected_type: type[Exception],
    expected_message: str,
) -> None:
    sentinel = f"{stage}-secret-sentinel"
    stream = RecordingAsyncStream(
        read_error=OSError(sentinel) if stage == "read" else None,
        write_error=OSError(sentinel) if stage == "write" else None,
        tls_error=OSError(sentinel) if stage == "tls" else None,
    )
    backend = RecordingAsyncBackend(
        stream,
        connect_errors=[OSError(sentinel)] if stage == "connect" else None,
    )
    transport = PinnedAsyncTransport(_validated("93.184.216.34"), backend)
    network_backend = transport._pool._network_backend

    try:
        with pytest.raises(expected_type) as exc_info:
            if stage == "connect":
                await network_backend.connect_tcp("public.example", 443)
            else:
                network_stream = await network_backend.connect_tcp(
                    "public.example", 443
                )
                if stage == "read":
                    await network_stream.read(1)
                elif stage == "write":
                    await network_stream.write(b"request")
                else:
                    await network_stream.start_tls(
                        ssl.create_default_context(),
                        server_hostname="public.example",
                    )
    finally:
        await transport.aclose()

    assert str(exc_info.value) == expected_message
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_async_transport_closes_on_success_error_and_cancellation_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pools: list[Any] = []
    response_streams: list[RecordingAsyncResponseStream] = []

    class RecordingAsyncPool:
        def __init__(self, **kwargs: Any) -> None:
            self.close_count = 0
            pools.append(self)

        async def handle_async_request(self, request: httpcore.Request) -> Any:
            stream = response_streams[-1]
            return SimpleNamespace(
                status=200,
                headers=[],
                stream=stream,
                extensions={},
            )

        async def aclose(self) -> None:
            self.close_count += 1

    monkeypatch.setattr(
        pinned_transport.httpcore,
        "AsyncConnectionPool",
        RecordingAsyncPool,
    )

    outcomes = (
        ("success", None, False),
        ("error", httpcore.ReadError("fixed response read failure"), False),
        ("cancellation", None, True),
    )
    for outcome, error, wait_for_cancellation in outcomes:
        response_stream = RecordingAsyncResponseStream(
            error,
            wait_for_cancellation=wait_for_cancellation,
        )
        response_streams.append(response_stream)
        transport = PinnedAsyncTransport(
            _validated("93.184.216.34"),
            RecordingAsyncBackend(RecordingAsyncStream()),
        )
        response = await transport.handle_async_request(_request())

        try:
            if outcome == "success":
                assert await response.aread() == b"response"
            elif outcome == "error":
                with pytest.raises(httpcore.ReadError) as exc_info:
                    await response.aread()
                assert str(exc_info.value) == "fixed response read failure"
            else:
                read_task = asyncio.create_task(response.aread())
                await response_stream.read_started.wait()
                read_task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await read_task
        finally:
            await response.aclose()
            await transport.aclose()

        assert response_stream.close_count == 1
        assert pools[-1].close_count == 1
