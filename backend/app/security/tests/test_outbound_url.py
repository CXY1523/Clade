from ipaddress import ip_address

import pytest

from app.security.outbound_url import OutboundRequestError, OutboundURLPolicy


class FakeResolver:
    def __init__(self, answers: dict[str, list[str]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, int]] = []

    def __call__(self, hostname: str, port: int):
        self.calls.append((hostname, port))
        answer = self.answers.get(hostname)
        if answer is None:
            raise OSError("resolver sentinel must not be exposed")
        return tuple(ip_address(value) for value in answer)


@pytest.mark.parametrize(
    ("url", "allow_local", "expected_code"),
    [
        ("http://public.example/v1", False, "outbound_https_required"),
        ("http://localhost:11434/v1", False, "local_ai_disabled"),
        ("http://127.0.0.1:11434/v1", False, "local_ai_disabled"),
        ("http://[::1]:11434/v1", False, "local_ai_disabled"),
        ("http://127.0.0.2:11434/v1", True, "private_network_blocked"),
        ("https://router.lan/v1", True, "private_network_blocked"),
        ("https://public.example/v1?key=secret", False, "outbound_url_invalid"),
        ("https://user:pass@public.example/v1", False, "outbound_url_invalid"),
        ("https://public.example/v1#fragment", False, "outbound_url_invalid"),
        ("file:///etc/passwd", False, "outbound_url_invalid"),
    ],
)
def test_policy_rejects_unsafe_urls(
    url: str,
    allow_local: bool,
    expected_code: str,
) -> None:
    resolver = FakeResolver(
        {
            "public.example": ["93.184.216.34"],
            "localhost": ["127.0.0.1", "::1"],
            "router.lan": ["192.168.1.1"],
        }
    )
    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(url, allow_local=allow_local)
    assert exc_info.value.code == expected_code
    assert "secret" not in str(exc_info.value)


def test_policy_accepts_only_public_https_or_explicit_loopback_exception() -> None:
    resolver = FakeResolver(
        {
            "public.example": [
                "93.184.216.34",
                "2606:2800:220:1:248:1893:25c8:1946",
            ],
            "localhost": ["127.0.0.1", "::1"],
        }
    )
    public = OutboundURLPolicy(resolver=resolver).validate(
        "HTTPS://PUBLIC.EXAMPLE./v1", allow_local=False
    )
    local = OutboundURLPolicy(resolver=resolver).validate(
        "http://localhost.:11434/v1", allow_local=True
    )
    assert public.hostname == "public.example"
    assert public.port == 443
    assert public.is_local is False
    assert public.url == "https://public.example/v1"
    assert local.hostname == "localhost"
    assert local.port == 11434
    assert local.is_local is True
    assert local.url == "http://localhost:11434/v1"


def test_mixed_public_and_private_resolution_rejects_entire_host() -> None:
    resolver = FakeResolver({"mixed.example": ["93.184.216.34", "169.254.169.254"]})
    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(
            "https://mixed.example/v1", allow_local=False
        )
    assert exc_info.value.code == "private_network_blocked"


@pytest.mark.parametrize("answers", [[], ["127.0.0.1", "93.184.216.34"]])
def test_localhost_requires_nonempty_all_loopback_resolution(answers: list[str]) -> None:
    resolver = FakeResolver({"localhost": answers})
    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(
            "http://localhost:11434/v1", allow_local=True
        )
    expected = "outbound_dns_failed" if not answers else "private_network_blocked"
    assert exc_info.value.code == expected


def test_resolution_deduplicates_addresses_while_preserving_order() -> None:
    resolver = FakeResolver(
        {
            "public.example": [
                "2606:2800:220:1:248:1893:25c8:1946",
                "93.184.216.34",
                "2606:2800:220:1:248:1893:25c8:1946",
            ]
        }
    )

    validated = OutboundURLPolicy(resolver=resolver).validate(
        "https://public.example:8443/v1", allow_local=False
    )

    assert validated.resolved_ips == (
        ip_address("2606:2800:220:1:248:1893:25c8:1946"),
        ip_address("93.184.216.34"),
    )
    assert resolver.calls == [("public.example", 8443)]


def test_dns_failure_maps_to_fixed_safe_error() -> None:
    resolver = FakeResolver({})

    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(
            "https://unresolvable.example/v1", allow_local=False
        )

    assert exc_info.value.code == "outbound_dns_failed"
    assert exc_info.value.status_code == 502
    assert exc_info.value.public_message == "域名解析失败"
    assert "resolver sentinel" not in str(exc_info.value)


def test_empty_dns_result_maps_to_dns_failure() -> None:
    resolver = FakeResolver({"empty.example": []})

    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(
            "https://empty.example/v1", allow_local=False
        )

    assert exc_info.value.code == "outbound_dns_failed"


@pytest.mark.parametrize(
    ("host", "answer"),
    [
        ("api.localhost", "127.0.0.1"),
        ("private-v4.example", "10.1.2.3"),
        ("private-v4-b.example", "172.16.2.3"),
        ("private-v4-c.example", "192.168.2.3"),
        ("ula.example", "fd00::1"),
        ("link-v4.example", "169.254.10.20"),
        ("link-v6.example", "fe80::1"),
        ("cgnat.example", "100.64.0.1"),
        ("documentation-v4.example", "192.0.2.1"),
        ("documentation-v6.example", "2001:db8::1"),
        ("reserved.example", "240.0.0.1"),
        ("benchmark.example", "198.18.0.1"),
        ("multicast-v4.example", "224.0.0.1"),
        ("multicast-v6.example", "ff02::1"),
        ("unspecified-v4.example", "0.0.0.0"),
        ("unspecified-v6.example", "::"),
    ],
)
def test_non_global_addresses_are_always_blocked(host: str, answer: str) -> None:
    resolver = FakeResolver({host: [answer]})

    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(
            f"https://{host}/v1", allow_local=True
        )

    assert exc_info.value.code == "private_network_blocked"


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "https://public.example/v 1",
        "https://public.example/\nsecret",
        "https://public.example/\x00secret",
        "https:\\public.example\\v1",
        "https:///v1",
        "https://public.example:0/v1",
        "https://public.example:/v1",
        "https://public.example:65536/v1",
        "https://public.example:not-a-port/v1",
        "https://public.example/v1?",
        "https://public.example/v1#",
    ],
)
def test_malformed_urls_are_rejected_before_dns(url: str) -> None:
    resolver = FakeResolver({"public.example": ["93.184.216.34"]})

    with pytest.raises(OutboundRequestError) as exc_info:
        OutboundURLPolicy(resolver=resolver).validate(url, allow_local=False)

    assert exc_info.value.code == "outbound_url_invalid"
    assert resolver.calls == []


def test_idna_hostname_is_normalized_before_resolution() -> None:
    resolver = FakeResolver({"xn--fsqu00a.xn--0zwm56d": ["93.184.216.34"]})

    validated = OutboundURLPolicy(resolver=resolver).validate(
        "HTTPS://例子.测试./v1", allow_local=False
    )

    assert validated.hostname == "xn--fsqu00a.xn--0zwm56d"
    assert validated.url == "https://xn--fsqu00a.xn--0zwm56d/v1"
    assert resolver.calls == [("xn--fsqu00a.xn--0zwm56d", 443)]


def test_idna_equivalent_public_trailing_dot_is_removed_before_resolution() -> None:
    resolver = FakeResolver({"public.example": ["93.184.216.34"]})

    validated = OutboundURLPolicy(resolver=resolver).validate(
        "HTTPS://PUBLIC.EXAMPLE\u3002/v1", allow_local=False
    )

    assert validated.hostname == "public.example"
    assert validated.url == "https://public.example/v1"
    assert resolver.calls == [("public.example", 443)]


def test_idna_equivalent_localhost_trailing_dot_uses_loopback_exception() -> None:
    resolver = FakeResolver({"localhost": ["127.0.0.1", "::1"]})

    validated = OutboundURLPolicy(resolver=resolver).validate(
        "http://localhost\u3002:11434/v1", allow_local=True
    )

    assert validated.hostname == "localhost"
    assert validated.url == "http://localhost:11434/v1"
    assert validated.is_local is True
    assert resolver.calls == [("localhost", 11434)]


def test_literal_ips_skip_dns_and_ipv6_keeps_url_brackets() -> None:
    resolver = FakeResolver({})

    validated = OutboundURLPolicy(resolver=resolver).validate(
        "https://[2606:2800:220:1:248:1893:25c8:1946]:8443/v1",
        allow_local=False,
    )

    assert validated.hostname == "2606:2800:220:1:248:1893:25c8:1946"
    assert validated.url == (
        "https://[2606:2800:220:1:248:1893:25c8:1946]:8443/v1"
    )
    assert validated.resolved_ips == (
        ip_address("2606:2800:220:1:248:1893:25c8:1946"),
    )
    assert resolver.calls == []


@pytest.mark.parametrize(
    ("url", "expected_port"),
    [
        ("http://localhost/v1", 80),
        ("https://localhost/v1", 443),
        ("http://127.0.0.1:11434/v1", 11434),
        ("http://[::1]:11434/v1", 11434),
    ],
)
def test_exact_loopback_accepts_http_or_https_when_enabled(
    url: str,
    expected_port: int,
) -> None:
    resolver = FakeResolver({"localhost": ["127.0.0.1", "::1"]})

    validated = OutboundURLPolicy(resolver=resolver).validate(url, allow_local=True)

    assert validated.port == expected_port
    assert validated.is_local is True


def test_error_type_exposes_public_contract_fields() -> None:
    error = OutboundRequestError(
        "private_network_blocked", 400, "该网络地址不允许访问"
    )

    assert error.code == "private_network_blocked"
    assert error.status_code == 400
    assert error.public_message == "该网络地址不允许访问"
    assert str(error) == "该网络地址不允许访问"
