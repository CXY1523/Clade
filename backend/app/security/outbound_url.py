from __future__ import annotations

import socket
import unicodedata
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Literal, TypeAlias
from urllib.parse import urlsplit, urlunsplit


OutboundErrorCode = Literal[
    "outbound_url_invalid",
    "outbound_https_required",
    "local_ai_disabled",
    "private_network_blocked",
    "outbound_dns_failed",
    "outbound_connect_failed",
    "outbound_response_too_large",
    "outbound_bad_response",
    "outbound_timeout",
]

IPAddress: TypeAlias = IPv4Address | IPv6Address
Resolver: TypeAlias = Callable[[str, int], Iterable[IPAddress]]


@dataclass(frozen=True)
class ValidatedOutboundURL:
    url: str
    scheme: Literal["http", "https"]
    hostname: str
    port: int
    resolved_ips: tuple[IPAddress, ...]
    is_local: bool


class OutboundRequestError(Exception):
    def __init__(
        self,
        code: OutboundErrorCode,
        status_code: int,
        public_message: str,
    ) -> None:
        super().__init__(public_message)
        self.code = code
        self.status_code = status_code
        self.public_message = public_message


def _invalid_url() -> OutboundRequestError:
    return OutboundRequestError("outbound_url_invalid", 400, "外部服务地址无效")


def _dns_failed() -> OutboundRequestError:
    return OutboundRequestError("outbound_dns_failed", 502, "域名解析失败")


def _system_resolver(hostname: str, port: int) -> tuple[IPAddress, ...]:
    answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    return tuple(ip_address(answer[4][0]) for answer in answers)


class OutboundURLPolicy:
    def __init__(self, resolver: Resolver | None = None) -> None:
        self._resolver = _system_resolver if resolver is None else resolver

    def validate(
        self,
        url: str,
        *,
        allow_local: bool,
    ) -> ValidatedOutboundURL:
        if (
            not isinstance(url, str)
            or not url
            or "\\" in url
            or any(
                char.isspace() or unicodedata.category(char) == "Cc"
                for char in url
            )
        ):
            raise _invalid_url()

        if "?" in url or "#" in url:
            raise _invalid_url()

        try:
            parsed = urlsplit(url)
            scheme = parsed.scheme.lower()
            hostname = parsed.hostname
            if (
                scheme not in ("http", "https")
                or not parsed.netloc
                or not hostname
                or "@" in parsed.netloc
                or parsed.netloc.endswith(":")
                or parsed.query
                or parsed.fragment
            ):
                raise _invalid_url()
            parsed_port = parsed.port
        except (TypeError, ValueError):
            raise _invalid_url() from None

        if parsed_port is not None and not 1 <= parsed_port <= 65535:
            raise _invalid_url()
        port = (
            parsed_port
            if parsed_port is not None
            else (443 if scheme == "https" else 80)
        )

        normalized_hostname = hostname.lower()
        if normalized_hostname.endswith("."):
            normalized_hostname = normalized_hostname[:-1]
        if (
            not normalized_hostname
            or ".." in normalized_hostname
            or normalized_hostname.endswith(".")
        ):
            raise _invalid_url()
        try:
            normalized_hostname = normalized_hostname.encode("idna").decode("ascii").lower()
        except UnicodeError:
            raise _invalid_url() from None
        if normalized_hostname.endswith("."):
            normalized_hostname = normalized_hostname[:-1]
        if not normalized_hostname or any(
            not label for label in normalized_hostname.split(".")
        ):
            raise _invalid_url()

        literal_ip: IPAddress | None
        try:
            literal_ip = ip_address(normalized_hostname)
        except ValueError:
            literal_ip = None

        if literal_ip is not None:
            resolved_ips = (literal_ip,)
        else:
            try:
                answers = self._resolver(normalized_hostname, port)
                resolved_ips = tuple(
                    dict.fromkeys(
                        address
                        if isinstance(address, (IPv4Address, IPv6Address))
                        else ip_address(address)
                        for address in answers
                    )
                )
            except (OSError, TypeError, ValueError):
                raise _dns_failed() from None
            if not resolved_ips:
                raise _dns_failed()

        exact_loopback = literal_ip in (ip_address("127.0.0.1"), ip_address("::1"))
        localhost = normalized_hostname == "localhost"
        if localhost and not all(address.is_loopback for address in resolved_ips):
            raise OutboundRequestError(
                "private_network_blocked", 400, "该网络地址不允许访问"
            )
        if localhost or exact_loopback:
            if not allow_local:
                raise OutboundRequestError(
                    "local_ai_disabled", 400, "本地 AI 访问未开启"
                )
            return ValidatedOutboundURL(
                url=self._normalized_url(
                    parsed.path, scheme, normalized_hostname, parsed_port
                ),
                scheme=scheme,
                hostname=normalized_hostname,
                port=port,
                resolved_ips=resolved_ips,
                is_local=True,
            )

        if any(not address.is_global or address.is_multicast for address in resolved_ips):
            raise OutboundRequestError(
                "private_network_blocked", 400, "该网络地址不允许访问"
            )

        if scheme != "https":
            raise OutboundRequestError(
                "outbound_https_required", 400, "公网服务必须使用 HTTPS"
            )

        return ValidatedOutboundURL(
            url=self._normalized_url(
                parsed.path, scheme, normalized_hostname, parsed_port
            ),
            scheme=scheme,
            hostname=normalized_hostname,
            port=port,
            resolved_ips=resolved_ips,
            is_local=False,
        )

    @staticmethod
    def _normalized_url(
        path: str,
        scheme: Literal["http", "https"],
        hostname: str,
        explicit_port: int | None,
    ) -> str:
        url_hostname = f"[{hostname}]" if ":" in hostname else hostname
        netloc = (
            f"{url_hostname}:{explicit_port}"
            if explicit_port is not None
            else url_hostname
        )
        return urlunsplit((scheme, netloc, path, "", ""))
