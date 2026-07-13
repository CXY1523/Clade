from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from app.security.admin_token import require_admin_token


def make_client(server_token: str | None) -> TestClient:
    app = FastAPI()
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        clade_admin_token=server_token
    )

    @app.get("/probe", dependencies=[Depends(require_admin_token)])
    def get_probe() -> dict[str, bool]:
        return {"ok": True}

    @app.head("/probe", dependencies=[Depends(require_admin_token)])
    def head_probe() -> None:
        return None

    @app.post("/probe", dependencies=[Depends(require_admin_token)])
    def post_probe() -> dict[str, bool]:
        return {"ok": True}

    return TestClient(app, raise_server_exceptions=False)


def test_get_and_head_allow_read_access_without_configured_token() -> None:
    client = make_client(None)

    assert client.get("/probe").status_code == 200
    assert client.head("/probe").status_code == 200


@pytest.mark.parametrize("server_token", [None, ""], ids=["none", "empty"])
def test_post_returns_503_when_server_token_is_not_configured(
    server_token: str | None,
) -> None:
    response = make_client(server_token).post("/probe")

    assert response.status_code == 503
    assert response.json() == {"detail": "管理员功能未启用"}


def test_missing_and_wrong_tokens_return_the_same_403_response() -> None:
    client = make_client("expected-token")

    missing_response = client.post("/probe")
    wrong_response = client.post(
        "/probe", headers={"X-Clade-Admin-Token": "wrong-token"}
    )

    expected_response = {"detail": "管理员令牌无效"}
    assert missing_response.status_code == 403
    assert missing_response.json() == expected_response
    assert wrong_response.status_code == 403
    assert wrong_response.json() == expected_response


def test_exact_configured_token_allows_post() -> None:
    response = make_client("expected-token").post(
        "/probe", headers={"X-Clade-Admin-Token": "expected-token"}
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_non_ascii_override_returns_403_for_missing_and_wrong_headers() -> None:
    client = make_client("管理员令牌")

    missing_response = client.post("/probe")
    wrong_response = client.post(
        "/probe", headers={"X-Clade-Admin-Token": "wrong-token"}
    )

    expected_response = {"detail": "管理员令牌无效"}
    assert missing_response.status_code == 403
    assert missing_response.json() == expected_response
    assert wrong_response.status_code == 403
    assert wrong_response.json() == expected_response


def test_settings_accepts_visible_ascii_admin_token() -> None:
    settings = Settings(
        CLADE_ADMIN_TOKEN="Letters123!#$%&*+-./:=?@^_~",
        _env_file=None,
    )

    assert settings.clade_admin_token == "Letters123!#$%&*+-./:=?@^_~"


@pytest.mark.parametrize(
    "invalid_token",
    [
        "管理员令牌",
        "line\nbreak",
        " leading-token",
        "trailing-token ",
    ],
    ids=["non-ascii", "control-character", "leading-space", "trailing-space"],
)
def test_settings_rejects_unsafe_admin_token_without_echoing_input(
    invalid_token: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(CLADE_ADMIN_TOKEN=invalid_token, _env_file=None)

    assert invalid_token not in str(exc_info.value)


def test_settings_repr_and_dump_exclude_admin_token() -> None:
    token = "private-token-123!"
    settings = Settings(CLADE_ADMIN_TOKEN=token, _env_file=None)

    assert token not in repr(settings)
    assert "clade_admin_token" not in repr(settings)
    assert token not in repr(settings.model_dump())
    assert "clade_admin_token" not in settings.model_dump()
