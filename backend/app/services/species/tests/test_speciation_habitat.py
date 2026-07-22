import logging
from types import SimpleNamespace

import pytest

from ....repositories.environment_repository import environment_repository
from ..speciation import SpeciationService
from ..speciation_habitat import (
    calculate_initial_habitat_for_child,
    calculate_suitability_for_species,
)


def _species(**overrides) -> SimpleNamespace:
    values = {
        "id": 7,
        "common_name": "测试物种",
        "habitat_type": "terrestrial",
        "abstract_traits": {"耐热性": 5, "耐寒性": 5, "耐旱性": 5},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _tile(
    tile_id: int,
    resources: float = 900.0,
    biome: str = "平原",
    temperature: float = 15.0,
    humidity: float = 0.6,
    is_lake: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=tile_id,
        biome=biome,
        temperature=temperature,
        humidity=humidity,
        resources=resources,
        is_lake=is_lake,
    )


def test_suitability_preserves_temperature_humidity_and_resource_formula() -> None:
    species = _species()

    assert calculate_suitability_for_species(species, _tile(1)) == pytest.approx(
        1.0
    )
    assert calculate_suitability_for_species(
        species, _tile(2, resources=450.0)
    ) == pytest.approx(0.8)
    assert (
        calculate_suitability_for_species(
            species, _tile(3, temperature=30.0)
        )
        == 0.0
    )


def test_initial_habitat_filters_assigned_tiles_sorts_and_normalizes(
    monkeypatch,
) -> None:
    tiles = [_tile(1), _tile(2, resources=450.0), _tile(3)]
    written = []
    monkeypatch.setattr(environment_repository, "list_tiles", lambda: tiles)
    monkeypatch.setattr(
        environment_repository,
        "write_habitats",
        lambda rows: written.extend(rows),
    )

    calculate_initial_habitat_for_child(
        _species(), SimpleNamespace(), 12, assigned_tiles={1, 2}
    )

    assert [row.tile_id for row in written] == [1, 2]
    assert [row.species_id for row in written] == [7, 7]
    assert [row.population for row in written] == [0, 0]
    assert [row.turn_index for row in written] == [12, 12]
    assert [row.suitability for row in written] == pytest.approx(
        [1 / 1.8, 0.8 / 1.8]
    )


def test_initial_habitat_preserves_logger_category_when_no_tiles(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(environment_repository, "list_tiles", lambda: [])

    with caplog.at_level(
        logging.ERROR,
        logger="app.services.species.speciation",
    ):
        calculate_initial_habitat_for_child(_species(), SimpleNamespace(), 12)

    assert any(
        record.name == "app.services.species.speciation"
        and "没有可用地块" in record.getMessage()
        for record in caplog.records
    )


def test_service_delegate_preserves_overridden_suitability_method(
    monkeypatch,
) -> None:
    tiles = [_tile(1), _tile(2)]
    written = []
    monkeypatch.setattr(environment_repository, "list_tiles", lambda: tiles)
    monkeypatch.setattr(
        environment_repository,
        "write_habitats",
        lambda rows: written.extend(rows),
    )

    class OverriddenSpeciationService(SpeciationService):
        def __init__(self) -> None:
            pass

        def _calculate_suitability_for_species(self, species, tile) -> float:
            return {1: 0.2, 2: 0.8}[tile.id]

    OverriddenSpeciationService()._calculate_initial_habitat_for_child(
        _species(), SimpleNamespace(), 12
    )

    assert [row.tile_id for row in written] == [2, 1]
    assert [row.suitability for row in written] == pytest.approx([0.8, 0.2])
