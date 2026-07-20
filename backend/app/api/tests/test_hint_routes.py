"""Game-hint route contracts for the split API router."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
from fastapi.testclient import TestClient

from ...main import app
from ...services.analytics import game_hints as hints_module
from ...services.analytics.game_hints import (
    GameHint,
    GameHintsService,
    HintPriority,
    HintType,
)


@pytest.fixture
def hints_service(monkeypatch: pytest.MonkeyPatch):
    """Replace only the stateful hint-generation boundary."""
    service = create_autospec(GameHintsService, instance=True)
    monkeypatch.setattr(hints_module, "game_hints_service", service)
    return service


@pytest.fixture
def client(mock_container, mock_session):
    app.state.container = mock_container
    app.state.session = mock_session
    test_client = TestClient(app, raise_server_exceptions=False)
    try:
        yield test_client
    finally:
        if hasattr(app.state, "container"):
            del app.state.container
        if hasattr(app.state, "session"):
            del app.state.session


def _turn_report_payload(turn_index: int) -> dict:
    return {
        "turn_index": turn_index,
        "pressures_summary": "stable",
        "narrative": f"turn {turn_index}",
        "species": [],
        "branching_events": [],
    }


def _hint() -> GameHint:
    return GameHint(
        hint_type=HintType.OPPORTUNITY,
        priority=HintPriority.HIGH,
        title="Open niche",
        message="A trophic niche is available.",
        icon="target",
        related_species=["SP-001"],
        suggested_actions=["Introduce a compatible species"],
    )


def test_get_hints_passes_recent_reports_and_preserves_the_response(
    client: TestClient,
    mock_container,
    hints_service,
) -> None:
    species = [SimpleNamespace(lineage_code="SP-001", status="alive")]
    mock_container.species_repository.list_species.return_value = species
    mock_container.simulation_engine.turn_counter = 12
    mock_container.history_repository.list_turns.return_value = [
        SimpleNamespace(record_data=_turn_report_payload(12)),
        SimpleNamespace(record_data=json.dumps(_turn_report_payload(11))),
    ]
    hint = _hint()
    hints_service.generate_hints.return_value = [hint]

    response = client.get("/api/hints")

    assert response.status_code == 200
    assert response.json() == {"hints": [hint.to_dict()], "turn": 12}
    hints_service.generate_hints.assert_called_once()
    call = hints_service.generate_hints.call_args
    assert call.kwargs["all_species"] == species
    assert call.kwargs["current_turn"] == 12
    assert call.kwargs["recent_report"].turn_index == 12
    assert call.kwargs["previous_report"].turn_index == 11


def test_get_hints_ignores_invalid_history_records(
    client: TestClient,
    mock_container,
    hints_service,
) -> None:
    mock_container.simulation_engine.turn_counter = 7
    mock_container.history_repository.list_turns.return_value = [
        SimpleNamespace(record_data={"invalid": "report"}),
        SimpleNamespace(record_data=None),
    ]
    hints_service.generate_hints.return_value = []

    response = client.get("/api/hints")

    assert response.status_code == 200
    assert response.json() == {"hints": [], "turn": 7}
    call = hints_service.generate_hints.call_args
    assert call.kwargs["recent_report"] is None
    assert call.kwargs["previous_report"] is None


def test_get_hints_degrades_when_species_state_is_unavailable(
    client: TestClient,
    mock_container,
    hints_service,
) -> None:
    mock_container.species_repository.list_species.side_effect = RuntimeError(
        "species unavailable"
    )

    response = client.get("/api/hints")

    assert response.status_code == 200
    assert response.json() == {"hints": [], "turn": 0}
    hints_service.generate_hints.assert_not_called()


def test_get_hints_degrades_when_generation_fails(
    client: TestClient,
    mock_container,
    hints_service,
) -> None:
    mock_container.simulation_engine.turn_counter = 8
    mock_container.history_repository.list_turns.return_value = []
    hints_service.generate_hints.side_effect = RuntimeError("generation failed")

    response = client.get("/api/hints")

    assert response.status_code == 200
    assert response.json() == {
        "hints": [],
        "turn": 8,
        "error": "failed_to_generate_hints",
    }


def test_clear_hints_preserves_the_legacy_response(
    client: TestClient,
    hints_service,
) -> None:
    response = client.post("/api/hints/clear")

    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "提示冷却已清除"}
    hints_service.clear_cooldown.assert_called_once_with()


def test_legacy_router_no_longer_registers_hint_endpoints() -> None:
    routes_path = Path(__file__).parents[1] / "routes.py"
    module = ast.parse(routes_path.read_text(encoding="utf-8"))
    registered_paths: list[str] = []

    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or not decorator.args:
                continue
            function = decorator.func
            if (
                isinstance(function, ast.Attribute)
                and isinstance(function.value, ast.Name)
                and function.value.id == "router"
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
            ):
                registered_paths.append(decorator.args[0].value)

    duplicate_hint_paths = sorted(
        path
        for path in registered_paths
        if path == "/hints" or path.startswith("/hints/")
    )

    assert duplicate_hint_paths == []
