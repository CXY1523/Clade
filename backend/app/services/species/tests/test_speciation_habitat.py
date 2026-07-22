import logging
from types import SimpleNamespace

import pytest

from ....repositories.environment_repository import environment_repository
from ..speciation import SpeciationService
from ..speciation_habitat import (
    calculate_initial_habitat_for_child,
    calculate_suitability_for_species,
    inherit_habitat_distribution,
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


def _habitat(
    tile_id: int,
    species_id: int,
    population: int,
    suitability: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        tile_id=tile_id,
        species_id=species_id,
        population=population,
        suitability=suitability,
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


def test_inheritance_transfers_assigned_population_and_writes_parent_second(
    monkeypatch,
) -> None:
    parent = _species(id=1, common_name="父代")
    child = _species(id=2, common_name="子代")
    habitats = [
        _habitat(10, 1, 100, 0.8),
        _habitat(20, 1, 40, 0.5),
    ]
    batches = []
    monkeypatch.setattr(
        environment_repository,
        "latest_habitats",
        lambda: habitats,
    )
    monkeypatch.setattr(
        environment_repository,
        "write_habitats",
        lambda rows: batches.append(list(rows)),
    )

    inherit_habitat_distribution(
        parent,
        child,
        15,
        assigned_tiles={10},
        reproduction_bonus=0.4,
    )

    assert len(batches) == 2
    assert [
        (
            row.tile_id,
            row.species_id,
            row.population,
            row.suitability,
            row.turn_index,
        )
        for row in batches[0]
    ] == [(10, 2, 132, 0.8, 15)]
    assert [
        (
            row.tile_id,
            row.species_id,
            row.population,
            row.suitability,
            row.turn_index,
        )
        for row in batches[1]
    ] == [(10, 1, 0, 0.8, 15)]


def test_inheritance_without_assignment_splits_population_without_parent_write(
    monkeypatch,
) -> None:
    parent = _species(id=1, common_name="父代")
    child = _species(id=2, common_name="子代")
    batches = []
    monkeypatch.setattr(
        environment_repository,
        "latest_habitats",
        lambda: [_habitat(10, 1, 101, 0.5)],
    )
    monkeypatch.setattr(
        environment_repository,
        "write_habitats",
        lambda rows: batches.append(list(rows)),
    )

    inherit_habitat_distribution(
        parent,
        child,
        15,
        reproduction_bonus=0.2,
    )

    assert len(batches) == 1
    assert batches[0][0].population == 55
    assert batches[0][0].species_id == 2


def test_inheritance_uses_assigned_tiles_when_parent_tiles_do_not_overlap(
    monkeypatch,
) -> None:
    parent = _species(id=1, common_name="父代")
    child = _species(
        id=2,
        common_name="子代",
        morphology_stats={"population": 9},
    )
    batches = []
    monkeypatch.setattr(
        environment_repository,
        "latest_habitats",
        lambda: [_habitat(10, 1, 100, 0.8)],
    )
    monkeypatch.setattr(
        environment_repository,
        "write_habitats",
        lambda rows: batches.append(list(rows)),
    )

    inherit_habitat_distribution(
        parent,
        child,
        15,
        assigned_tiles={30, 40},
    )

    assert len(batches) == 1
    assert {row.tile_id for row in batches[0]} == {30, 40}
    assert {row.population for row in batches[0]} == {4}
    assert {row.suitability for row in batches[0]} == {0.5}


def test_service_inheritance_delegate_preserves_initial_habitat_override(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(environment_repository, "latest_habitats", lambda: [])

    class OverriddenSpeciationService(SpeciationService):
        def __init__(self) -> None:
            pass

        def _calculate_initial_habitat_for_child(
            self,
            child,
            parent,
            turn_index,
            assigned_tiles=None,
        ) -> None:
            calls.append((child.id, parent.id, turn_index, assigned_tiles))

    OverriddenSpeciationService()._inherit_habitat_distribution(
        _species(id=1),
        _species(id=2),
        15,
        assigned_tiles={10},
    )

    assert calls == [(2, 1, 15, {10})]
