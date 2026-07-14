from __future__ import annotations

import json
import logging
import queue
import ssl
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar, cast

import httpcore
import httpx

from .outbound_url import (
    OutboundRequestError,
    OutboundURLPolicy,
    ValidatedOutboundURL,
)


Endpoint = Literal["models", "messages"]
T = TypeVar("T")


_LOG_CONTEXT = threading.local()


class _SafeProbeTraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not bool(getattr(_LOG_CONTEXT, "suppress_httpcore_trace", False))


_TRACE_FILTER = _SafeProbeTraceFilter()
for _logger_name in (
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
):
    logging.getLogger(_logger_name).addFilter(_TRACE_FILTER)


@dataclass(frozen=True)
class ProbeTimeouts:
    connect: float
    read: float
    write: float
    pool: float
    total: float


CONNECTION_TEST_TIMEOUTS = ProbeTimeouts(5.0, 5.0, 5.0, 5.0, 10.0)
MODEL_LIST_TIMEOUTS = ProbeTimeouts(5.0, 10.0, 5.0, 5.0, 15.0)
MODEL_LIST_MAX_BYTES = 1_048_576


def _invalid_url() -> OutboundRequestError:
    return OutboundRequestError("outbound_url_invalid", 400, "外部服务地址无效")


def _connect_failed() -> OutboundRequestError:
    return OutboundRequestError("outbound_connect_failed", 502, "外部服务连接失败")


def _response_too_large() -> OutboundRequestError:
    return OutboundRequestError(
        "outbound_response_too_large", 502, "外部服务响应过大"
    )


def _bad_response() -> OutboundRequestError:
    return OutboundRequestError("outbound_bad_response", 502, "外部服务响应无效")


def _timeout() -> OutboundRequestError:
    return OutboundRequestError("outbound_timeout", 504, "外部服务请求超时")


class _Runner(Protocol):
    def run(self, operation: Callable[[], T], *, timeout: float) -> T: ...


class _BoundedDaemonRunner:
    """Run blocking operations under a bounded daemon-thread total deadline."""

    def __init__(self, max_workers: int = 4) -> None:
        if max_workers <= 0 or max_workers > 4:
            raise ValueError("max_workers must be between 1 and 4")
        self._slots = threading.BoundedSemaphore(max_workers)
        self._state_lock = threading.Lock()
        self._active = 0
        self._max_active = 0

    @property
    def max_active(self) -> int:
        with self._state_lock:
            return self._max_active

    def run(self, operation: Callable[[], T], *, timeout: float) -> T:
        deadline = time.monotonic() + max(timeout, 0.0)
        if not self._slots.acquire(timeout=max(deadline - time.monotonic(), 0.0)):
            raise TimeoutError

        result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

        def worker() -> None:
            with self._state_lock:
                self._active += 1
                self._max_active = max(self._max_active, self._active)
            try:
                try:
                    result.put((True, operation()))
                except BaseException as exc:
                    result.put((False, exc))
            finally:
                with self._state_lock:
                    self._active -= 1
                self._slots.release()

        thread = threading.Thread(
            target=worker,
            name="clade-safe-http",
            daemon=True,
        )
        try:
            thread.start()
        except BaseException:
            self._slots.release()
            raise

        try:
            succeeded, value = result.get(
                timeout=max(deadline - time.monotonic(), 0.0)
            )
        except queue.Empty:
            raise TimeoutError from None
        if succeeded:
            return cast(T, value)
        raise cast(BaseException, value)


_DEFAULT_RUNNER = _BoundedDaemonRunner(max_workers=4)


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


class _PinnedTransport(httpx.BaseTransport):
    def __init__(
        self,
        validated: ValidatedOutboundURL,
        network_backend: httpcore.NetworkBackend,
    ) -> None:
        self._pool = httpcore.ConnectionPool(
            ssl_context=ssl.create_default_context(),
            retries=0,
            network_backend=_SanitizedNetworkBackend(
                _PinnedNetworkBackend(validated, network_backend)
            ),
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
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
            extensions=request.extensions,
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


class SafeProbeClient:
    def __init__(
        self,
        policy: OutboundURLPolicy | None = None,
        *,
        network_backend: httpcore.NetworkBackend | None = None,
        runner: _Runner | None = None,
        connection_timeouts: ProbeTimeouts = CONNECTION_TEST_TIMEOUTS,
        model_list_timeouts: ProbeTimeouts = MODEL_LIST_TIMEOUTS,
    ) -> None:
        self._policy = OutboundURLPolicy() if policy is None else policy
        self._network_backend = (
            httpcore.SyncBackend() if network_backend is None else network_backend
        )
        self._runner = _DEFAULT_RUNNER if runner is None else runner
        self._connection_timeouts = connection_timeouts
        self._model_list_timeouts = model_list_timeouts

    def probe_status(
        self,
        base_url: str,
        *,
        endpoint: Literal["models", "messages"],
        headers: Mapping[str, str],
        allow_local: bool,
    ) -> int:
        return self._run_with_deadline(
            lambda: self._probe_status(
                base_url,
                endpoint=endpoint,
                headers=headers,
                allow_local=allow_local,
                timeouts=self._connection_timeouts,
            ),
            self._connection_timeouts.total,
        )

    def fetch_json(
        self,
        base_url: str,
        *,
        endpoint: Literal["models", "messages"],
        headers: Mapping[str, str],
        allow_local: bool,
        max_bytes: int = MODEL_LIST_MAX_BYTES,
    ) -> dict[str, Any]:
        return self._run_with_deadline(
            lambda: self._fetch_json(
                base_url,
                endpoint=endpoint,
                headers=headers,
                allow_local=allow_local,
                max_bytes=max_bytes,
                timeouts=self._model_list_timeouts,
            ),
            self._model_list_timeouts.total,
        )

    def _run_with_deadline(self, operation: Callable[[], T], total: float) -> T:
        def guarded_operation() -> T:
            previous = bool(
                getattr(_LOG_CONTEXT, "suppress_httpcore_trace", False)
            )
            _LOG_CONTEXT.suppress_httpcore_trace = True
            try:
                return operation()
            finally:
                _LOG_CONTEXT.suppress_httpcore_trace = previous

        try:
            return self._runner.run(guarded_operation, timeout=total)
        except OutboundRequestError:
            raise
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
            raise _timeout() from None
        except (httpx.DecodingError, httpx.ProtocolError, httpcore.ProtocolError):
            raise _bad_response() from None
        except (
            httpx.RequestError,
            httpcore.NetworkError,
            OSError,
        ):
            raise _connect_failed() from None

    def _validated_endpoint(
        self,
        base_url: str,
        *,
        endpoint: Endpoint,
        allow_local: bool,
    ) -> tuple[ValidatedOutboundURL, str]:
        if endpoint not in ("models", "messages"):
            raise _invalid_url()
        validated = self._policy.validate(base_url, allow_local=allow_local)
        request_url = f"{validated.url.rstrip('/')}/{endpoint}"
        return validated, request_url

    def _client(
        self,
        validated: ValidatedOutboundURL,
        timeouts: ProbeTimeouts,
    ) -> httpx.Client:
        return httpx.Client(
            transport=_PinnedTransport(validated, self._network_backend),
            timeout=httpx.Timeout(
                connect=timeouts.connect,
                read=timeouts.read,
                write=timeouts.write,
                pool=timeouts.pool,
            ),
            trust_env=False,
            follow_redirects=False,
        )

    @staticmethod
    def _request_headers(headers: Mapping[str, str]) -> dict[str, str]:
        request_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in ("host", "accept-encoding")
        }
        request_headers["Accept-Encoding"] = "identity"
        return request_headers

    def _probe_status(
        self,
        base_url: str,
        *,
        endpoint: Endpoint,
        headers: Mapping[str, str],
        allow_local: bool,
        timeouts: ProbeTimeouts,
    ) -> int:
        validated, request_url = self._validated_endpoint(
            base_url, endpoint=endpoint, allow_local=allow_local
        )
        client = self._client(validated, timeouts)
        try:
            with client.stream(
                "GET", request_url, headers=self._request_headers(headers)
            ) as response:
                return response.status_code
        finally:
            client.close()

    def _fetch_json(
        self,
        base_url: str,
        *,
        endpoint: Endpoint,
        headers: Mapping[str, str],
        allow_local: bool,
        max_bytes: int,
        timeouts: ProbeTimeouts,
    ) -> dict[str, Any]:
        if max_bytes < 0:
            raise _invalid_url()
        max_bytes = min(max_bytes, MODEL_LIST_MAX_BYTES)
        validated, request_url = self._validated_endpoint(
            base_url, endpoint=endpoint, allow_local=allow_local
        )
        client = self._client(validated, timeouts)
        try:
            with client.stream(
                "GET", request_url, headers=self._request_headers(headers)
            ) as response:
                content_encoding = response.headers.get("Content-Encoding")
                if content_encoding is not None:
                    encodings = [
                        value.strip().lower()
                        for value in content_encoding.split(",")
                        if value.strip()
                    ]
                    if not encodings or any(
                        value != "identity" for value in encodings
                    ):
                        raise _bad_response()

                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        raise _bad_response() from None
                    if declared_length < 0:
                        raise _bad_response()
                    if declared_length > max_bytes:
                        raise _response_too_large()

                content = bytearray()
                for chunk in response.iter_bytes(chunk_size=max_bytes + 1):
                    remaining = max_bytes + 1 - len(content)
                    content.extend(chunk[:remaining])
                    if len(content) > max_bytes or len(chunk) > remaining:
                        raise _response_too_large()
                try:
                    parsed = json.loads(content)
                except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                    raise _bad_response() from None
                if not isinstance(parsed, dict):
                    raise _bad_response()
                return cast(dict[str, Any], parsed)
        finally:
            client.close()
