from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.core.config import get_settings
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

    return TestClient(app)


def test_get_and_head_allow_read_access_without_configured_token() -> None:
    client = make_client(None)

    assert client.get("/probe").status_code == 200
    assert client.head("/probe").status_code == 200


def test_post_returns_503_when_server_token_is_not_configured() -> None:
    response = make_client(None).post("/probe")

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
