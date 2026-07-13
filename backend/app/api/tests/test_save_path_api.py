from __future__ import annotations

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ...main import app
from ...security.save_paths import SavePathError, resolve_save_directory


PRIVATE_PATH_SENTINEL = r"C:\private\clade-tree\saves"


@pytest.fixture
def client(mock_container, mock_session):
    app.state.container = mock_container
    app.state.session = mock_session
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        del app.state.container
        del app.state.session


@pytest.fixture
def stub_complex_route_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep create/load error tests focused on save-manager boundaries."""
    from ...core import database, seed
    from ...services.analytics import achievements, game_hints
    from ...services.species import dispersal_engine, habitat_manager
    from ...services.system import divine_energy, divine_progression

    fake_db_session = MagicMock()
    fake_db_session.exec.return_value.all.return_value = []

    @contextmanager
    def fake_session_scope():
        yield fake_db_session

    monkeypatch.setattr(database, "session_scope", fake_session_scope)
    monkeypatch.setattr(seed, "seed_defaults", lambda: None)
    monkeypatch.setattr(divine_energy.energy_service, "reset", lambda: None)
    monkeypatch.setattr(divine_progression.divine_progression_service, "reset", lambda: None)
    monkeypatch.setattr(achievements.achievement_service, "reset", lambda: None)
    monkeypatch.setattr(game_hints.game_hints_service, "clear_cooldown", lambda: None)
    monkeypatch.setattr(habitat_manager.habitat_manager, "clear_all_caches", lambda: None)
    monkeypatch.setattr(dispersal_engine.dispersal_engine, "clear_caches", lambda: None)


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("post", "/api/saves/create", {"save_name": "../outside", "scenario": "原初大陆"}),
        ("post", "/api/saves/save", {"save_name": r"..\outside"}),
        ("post", "/api/saves/load", {"save_name": "NUL"}),
        ("delete", "/api/saves/CON", None),
    ],
)
def test_save_routes_reject_unsafe_names_before_manager_access(
    client: TestClient,
    mock_container,
    method: str,
    path: str,
    body: dict | None,
) -> None:
    response = client.request(method.upper(), path, json=body)

    assert response.status_code in {400, 422}
    assert response.json().get("detail")
    mock_container.save_manager.create_save.assert_not_called()
    mock_container.save_manager.save_game.assert_not_called()
    mock_container.save_manager.load_game.assert_not_called()
    mock_container.save_manager.delete_save.assert_not_called()


def test_unicode_save_name_reaches_the_manager_unchanged(
    client: TestClient,
    mock_container,
) -> None:
    mock_container.simulation_engine.turn_counter = 4

    response = client.post("/api/saves/save", json={"save_name": "历史 存档-1"})

    assert response.status_code == 200
    mock_container.save_manager.save_game.assert_called_once_with(
        "历史 存档-1",
        turn_index=4,
    )


@pytest.mark.parametrize(
    ("method", "path", "body", "manager_method"),
    [
        ("post", "/api/saves/create", {"save_name": "safe-create"}, "create_save"),
        ("post", "/api/saves/save", {"save_name": "safe-save"}, "save_game"),
        ("post", "/api/saves/load", {"save_name": "safe-load"}, "load_game"),
        ("delete", "/api/saves/safe-delete", None, "delete_save"),
    ],
)
def test_save_routes_map_manager_path_errors_to_safe_422(
    client: TestClient,
    mock_container,
    stub_complex_route_side_effects: None,
    method: str,
    path: str,
    body: dict | None,
    manager_method: str,
) -> None:
    manager_call = getattr(mock_container.save_manager, manager_method)
    with pytest.raises(SavePathError) as exc_info:
        resolve_save_directory(PRIVATE_PATH_SENTINEL, PRIVATE_PATH_SENTINEL)
    manager_call.side_effect = exc_info.value

    response = client.request(method.upper(), path, json=body)

    assert response.status_code == 422
    assert response.json().get("detail")
    assert PRIVATE_PATH_SENTINEL not in response.text


@pytest.mark.parametrize(
    ("method", "path", "body", "manager_method", "expected_detail"),
    [
        (
            "post",
            "/api/saves/create",
            {"save_name": "safe-create"},
            "create_save",
            "创建存档失败",
        ),
        (
            "post",
            "/api/saves/save",
            {"save_name": "safe-save"},
            "save_game",
            "保存游戏失败",
        ),
        (
            "post",
            "/api/saves/load",
            {"save_name": "safe-load"},
            "load_game",
            "加载存档失败",
        ),
        (
            "delete",
            "/api/saves/safe-delete",
            None,
            "delete_save",
            "删除存档失败",
        ),
    ],
)
def test_save_routes_hide_generic_error_paths_from_response_and_logs(
    client: TestClient,
    mock_container,
    stub_complex_route_side_effects: None,
    caplog: pytest.LogCaptureFixture,
    method: str,
    path: str,
    body: dict | None,
    manager_method: str,
    expected_detail: str,
) -> None:
    manager_call = getattr(mock_container.save_manager, manager_method)
    manager_call.side_effect = RuntimeError(f"failed under {PRIVATE_PATH_SENTINEL}")

    with caplog.at_level(logging.ERROR, logger="app.api.simulation"):
        response = client.request(method.upper(), path, json=body)

    assert response.status_code == 500
    assert response.json()["detail"] == expected_detail
    assert PRIVATE_PATH_SENTINEL not in response.text
    assert PRIVATE_PATH_SENTINEL not in caplog.text


def test_delete_save_returns_manager_result_as_found(
    client: TestClient,
    mock_container,
) -> None:
    mock_container.save_manager.delete_save.return_value = False

    response = client.delete("/api/saves/missing-save")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "deleted": "missing-save",
        "found": False,
    }
    mock_container.save_manager.delete_save.assert_called_once_with("missing-save")
