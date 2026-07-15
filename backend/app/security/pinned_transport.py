from __future__ import annotations

import ssl
from collections.abc import Iterator
from typing import Any

import httpcore
import httpx

from .outbound_url import ValidatedOutboundURL


class _PinnedNetworkBackend(httpcore.NetworkBackend):
    def __init__(
        self,
        validated: ValidatedOutboundURL,
        delegate: httpcore.NetworkBackend,
    ) -> None:
        self._validated = validated
        self._delegate = delegate

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        if host != self._validated.hostname or port != self._validated.port:
            raise httpcore.ConnectError("outbound origin does not match approval")

        last_error: BaseException | None = None
        for approved_ip in self._validated.resolved_ips:
            try:
                return self._delegate.connect_tcp(
                    str(approved_ip),
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_error = exc
        raise httpcore.ConnectError("all approved addresses failed") from last_error

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        raise httpcore.ConnectError("unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._delegate.sleep(seconds)


class _SanitizedNetworkStream(httpcore.NetworkStream):
    """Prevent delegate exception details from reaching httpcore trace logs."""

    def __init__(self, delegate: httpcore.NetworkStream) -> None:
        self._delegate = delegate

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        try:
            return self._delegate.read(max_bytes, timeout=timeout)
        except httpcore.ReadTimeout:
            raise httpcore.ReadTimeout("outbound read timed out") from None
        except Exception:
            raise httpcore.ReadError("outbound read failed") from None

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        try:
            self._delegate.write(buffer, timeout=timeout)
        except httpcore.WriteTimeout:
            raise httpcore.WriteTimeout("outbound write timed out") from None
        except Exception:
            raise httpcore.WriteError("outbound write failed") from None

    def close(self) -> None:
        try:
            self._delegate.close()
        except Exception:
            raise httpcore.NetworkError("outbound close failed") from None

    def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.NetworkStream:
        try:
            stream = self._delegate.start_tls(
                ssl_context,
                server_hostname=server_hostname,
                timeout=timeout,
            )
        except httpcore.ConnectTimeout:
            raise httpcore.ConnectTimeout("outbound TLS timed out") from None
        except Exception:
            raise httpcore.ConnectError("outbound TLS failed") from None
        return _SanitizedNetworkStream(stream)


class _SanitizedNetworkBackend(httpcore.NetworkBackend):
    def __init__(self, delegate: httpcore.NetworkBackend) -> None:
        self._delegate = delegate

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        try:
            stream = self._delegate.connect_tcp(
                host,
                port,
                timeout=timeout,
                local_address=local_address,
                socket_options=socket_options,
            )
        except httpcore.ConnectTimeout:
            raise httpcore.ConnectTimeout("outbound connect timed out") from None
        except Exception:
            raise httpcore.ConnectError("outbound connect failed") from None
        return _SanitizedNetworkStream(stream)

    def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.NetworkStream:
        raise httpcore.ConnectError("unix sockets are not allowed")

    def sleep(self, seconds: float) -> None:
        self._delegate.sleep(seconds)


class _CoreResponseStream(httpx.SyncByteStream):
    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def __iter__(self) -> Iterator[bytes]:
        yield from self._stream

    def close(self) -> None:
        self._stream.close()


class PinnedSyncTransport(httpx.BaseTransport):
    def __init__(
        self,
        validated: ValidatedOutboundURL,
        network_backend: httpcore.NetworkBackend | None = None,
    ) -> None:
        delegate = (
            httpcore.SyncBackend()
            if network_backend is None
            else network_backend
        )
        self._validated = validated
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(),
            retries=0,
            network_backend=_SanitizedNetworkBackend(
                _PinnedNetworkBackend(validated, delegate)
            ),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.scheme != self._validated.scheme:
            raise httpcore.ConnectError("outbound origin does not match approval")

        extensions = request.extensions.copy()
        extensions.pop("sni_hostname", None)
        core_request = httpcore.Request(
            method=request.method,
            url=httpcore.URL(
                scheme=request.url.raw_scheme,
                host=request.url.raw_host,
                port=request.url.port,
                target=request.url.raw_path,
            ),
            headers=request.headers.raw,
            content=request.stream,
            extensions=extensions,
        )
        core_response = self._pool.handle_request(core_request)
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_CoreResponseStream(core_response.stream),
            extensions=core_response.extensions,
        )

    def close(self) -> None:
        self._pool.close()
