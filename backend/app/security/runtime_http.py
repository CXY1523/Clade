from __future__ import annotations

import json
import logging
import unicodedata
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, cast
from urllib.parse import urlsplit

import httpcore
import httpx

from .outbound_url import (
    OutboundRequestError,
    OutboundURLPolicy,
    ValidatedOutboundURL,
)
from .pinned_transport import PinnedAsyncTransport, PinnedSyncTransport


AI_JSON_MAX_BYTES = 8 * 1024 * 1024
EMBEDDING_JSON_MAX_BYTES = 16 * 1024 * 1024
STREAM_MAX_BYTES = 16 * 1024 * 1024
STREAM_EVENT_MAX_BYTES = 1 * 1024 * 1024

RETRYABLE_OUTBOUND_CODES = frozenset(
    {"outbound_connect_failed", "outbound_timeout"}
)


_SUPPRESS_RUNTIME_HTTP_LOGS: ContextVar[bool] = ContextVar(
    "suppress_runtime_http_logs",
    default=False,
)


class _RuntimeHTTPLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SUPPRESS_RUNTIME_HTTP_LOGS.get()


_RUNTIME_HTTP_LOG_FILTER = _RuntimeHTTPLogFilter()
for _logger_name in (
    "httpx",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
):
    logging.getLogger(_logger_name).addFilter(_RUNTIME_HTTP_LOG_FILTER)


@dataclass(frozen=True)
class RuntimeTimeouts:
    connect: float = 10.0
    read: float = 60.0
    write: float = 30.0
    pool: float = 10.0
    stream_idle: float = 120.0
    stream_total: float = 600.0


def _invalid_url() -> OutboundRequestError:
    return OutboundRequestError("outbound_url_invalid", 400, "外部服务地址无效")


def _connect_failed() -> OutboundRequestError:
    return OutboundRequestError("outbound_connect_failed", 502, "外部服务连接失败")


def _response_too_large() -> OutboundRequestError:
    return OutboundRequestError(
        "outbound_response_too_large",
        502,
        "外部服务响应过大",
    )


def _bad_response() -> OutboundRequestError:
    return OutboundRequestError("outbound_bad_response", 502, "外部服务响应无效")


def _timeout() -> OutboundRequestError:
    return OutboundRequestError("outbound_timeout", 504, "外部服务请求超时")


def _validated_request_url(
    validated: ValidatedOutboundURL,
    request_target: str,
) -> str:
    if not isinstance(request_target, str) or not request_target:
        raise _invalid_url()
    if any(
        char.isspace() or unicodedata.category(char) == "Cc"
        for char in request_target
    ):
        raise _invalid_url()
    if "\\" in request_target:
        raise _invalid_url()
    if "#" in request_target:
        raise _invalid_url()
    if not request_target.startswith("/") or request_target.startswith("//"):
        raise _invalid_url()
    try:
        parsed = urlsplit(request_target)
    except (TypeError, ValueError):
        raise _invalid_url() from None
    if parsed.scheme or parsed.netloc or parsed.fragment:
        raise _invalid_url()
    if "://" in request_target:
        raise _invalid_url()
    try:
        request_target.encode("utf-8")
    except UnicodeEncodeError:
        raise _invalid_url() from None
    return validated.url.rstrip("/") + request_target


def _request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    request_headers = {
        key: value
        for key, value in headers.items()
        if key.lower() not in ("host", "accept-encoding")
    }
    request_headers["Accept-Encoding"] = "identity"
    return request_headers


def _reject_non_identity_encoding(headers: httpx.Headers) -> None:
    content_encoding = headers.get("Content-Encoding")
    if content_encoding is None:
        return
    encodings = [
        value.strip().lower()
        for value in content_encoding.split(",")
        if value.strip()
    ]
    if not encodings or any(value != "identity" for value in encodings):
        raise _bad_response()


def _reject_oversized_content_length(
    headers: httpx.Headers,
    max_bytes: int,
) -> None:
    content_length = headers.get("Content-Length")
    if content_length is None:
        return
    try:
        declared_length = int(content_length)
    except ValueError:
        raise _bad_response() from None
    if declared_length < 0:
        raise _bad_response()
    if declared_length > max_bytes:
        raise _response_too_large()


def _check_response_headers(response: httpx.Response, max_bytes: int) -> None:
    if not 200 <= response.status_code < 300:
        raise _bad_response()
    _reject_non_identity_encoding(response.headers)
    _reject_oversized_content_length(response.headers, max_bytes)


def _parse_json_object(content: bytearray) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except (UnicodeDecodeError, ValueError, TypeError, RecursionError):
        raise _bad_response() from None
    if not isinstance(parsed, dict):
        raise _bad_response()
    return cast(dict[str, Any], parsed)


def _read_sync_json(response: httpx.Response, max_bytes: int) -> dict[str, Any]:
    _check_response_headers(response, max_bytes)
    content = bytearray()
    for chunk in response.iter_raw(chunk_size=max_bytes + 1):
        remaining = max_bytes + 1 - len(content)
        content.extend(chunk[:remaining])
        if len(content) > max_bytes or len(chunk) > remaining:
            raise _response_too_large()
    return _parse_json_object(content)


async def _read_async_json(
    response: httpx.Response,
    max_bytes: int,
) -> dict[str, Any]:
    _check_response_headers(response, max_bytes)
    content = bytearray()
    async for chunk in response.aiter_raw(chunk_size=max_bytes + 1):
        remaining = max_bytes + 1 - len(content)
        content.extend(chunk[:remaining])
        if len(content) > max_bytes or len(chunk) > remaining:
            raise _response_too_large()
    return _parse_json_object(content)


class SafeRuntimeClient:
    def __init__(
        self,
        policy: OutboundURLPolicy | None = None,
        *,
        sync_network_backend: httpcore.NetworkBackend | None = None,
        async_network_backend: httpcore.AsyncNetworkBackend | None = None,
        timeouts: RuntimeTimeouts = RuntimeTimeouts(),
    ) -> None:
        self._policy = OutboundURLPolicy() if policy is None else policy
        self._sync_network_backend = sync_network_backend
        self._async_network_backend = async_network_backend
        self._timeouts = timeouts

    def post_json(
        self,
        base_url: str,
        *,
        request_target: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any],
        allow_local: bool,
        read_timeout: float,
        max_bytes: int = AI_JSON_MAX_BYTES,
    ) -> dict[str, Any]:
        token = _SUPPRESS_RUNTIME_HTTP_LOGS.set(True)
        try:
            validated = self._policy.validate(base_url, allow_local=allow_local)
            request_url = _validated_request_url(validated, request_target)
            if (
                not isinstance(max_bytes, int)
                or isinstance(max_bytes, bool)
                or max_bytes < 0
            ):
                raise _invalid_url()
            transport = PinnedSyncTransport(
                validated,
                self._sync_network_backend,
            )
            client: httpx.Client | None = None
            try:
                client = httpx.Client(
                    transport=transport,
                    timeout=httpx.Timeout(
                        connect=self._timeouts.connect,
                        read=read_timeout,
                        write=self._timeouts.write,
                        pool=self._timeouts.pool,
                    ),
                    trust_env=False,
                    follow_redirects=False,
                )
                with client.stream(
                    "POST",
                    request_url,
                    headers=_request_headers(headers),
                    json=json_body,
                ) as response:
                    return _read_sync_json(response, max_bytes)
            finally:
                if client is None:
                    transport.close()
                else:
                    client.close()
        except OutboundRequestError:
            raise
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
            raise _timeout() from None
        except (httpx.DecodingError, httpx.ProtocolError, httpcore.ProtocolError):
            raise _bad_response() from None
        except (httpx.RequestError, httpcore.NetworkError, OSError):
            raise _connect_failed() from None
        except Exception:
            raise _bad_response() from None
        finally:
            _SUPPRESS_RUNTIME_HTTP_LOGS.reset(token)

    async def apost_json(
        self,
        base_url: str,
        *,
        request_target: str,
        headers: Mapping[str, str],
        json_body: Mapping[str, Any],
        allow_local: bool,
        read_timeout: float,
        max_bytes: int = AI_JSON_MAX_BYTES,
    ) -> dict[str, Any]:
        token = _SUPPRESS_RUNTIME_HTTP_LOGS.set(True)
        try:
            validated = self._policy.validate(base_url, allow_local=allow_local)
            request_url = _validated_request_url(validated, request_target)
            if (
                not isinstance(max_bytes, int)
                or isinstance(max_bytes, bool)
                or max_bytes < 0
            ):
                raise _invalid_url()
            transport = PinnedAsyncTransport(
                validated,
                self._async_network_backend,
            )
            client: httpx.AsyncClient | None = None
            try:
                client = httpx.AsyncClient(
                    transport=transport,
                    timeout=httpx.Timeout(
                        connect=self._timeouts.connect,
                        read=read_timeout,
                        write=self._timeouts.write,
                        pool=self._timeouts.pool,
                    ),
                    trust_env=False,
                    follow_redirects=False,
                )
                async with client.stream(
                    "POST",
                    request_url,
                    headers=_request_headers(headers),
                    json=json_body,
                ) as response:
                    return await _read_async_json(response, max_bytes)
            finally:
                if client is None:
                    await transport.aclose()
                else:
                    await client.aclose()
        except OutboundRequestError:
            raise
        except (TimeoutError, httpx.TimeoutException, httpcore.TimeoutException):
            raise _timeout() from None
        except (httpx.DecodingError, httpx.ProtocolError, httpcore.ProtocolError):
            raise _bad_response() from None
        except (httpx.RequestError, httpcore.NetworkError, OSError):
            raise _connect_failed() from None
        except Exception:
            raise _bad_response() from None
        finally:
            _SUPPRESS_RUNTIME_HTTP_LOGS.reset(token)
