"""Species-repository isolation coverage for the vegetation cover stage."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

from ..stages import VegetationCoverStage


class _RecordingSpeciesRepository:
    def __init__(self, species: list[object]) -> None:
        self.species = species
        self.list_calls = 0

    def list_species(self) -> list[object]:
        self.list_calls += 1
        return self.species


class _RecordingEnvironmentRepository:
    def __init__(self, tiles: list[object], habitats: list[object]) -> None:
        self.tiles = tiles
        self.habitats = habitats
        self.list_tiles_calls = 0
        self.latest_habitats_calls = 0
        self.upsert_batches: list[list[object]] = []

    def list_tiles(self) -> list[object]:
        self.list_tiles_calls += 1
        return self.tiles

    def latest_habitats(self) -> list[object]:
        self.latest_habitats_calls += 1
        return self.habitats

    def upsert_tiles(self, tiles: list[object]) -> None:
        self.upsert_batches.append(tiles)


class _RecordingVegetationCoverService:
    def __init__(self, updated_tiles: list[object]) -> None:
        self.updated_tiles = updated_tiles
        self.calls: list[tuple[list[object], list[object], dict[int, object]]] = []

    def update_vegetation_cover(
        self,
        tiles: list[object],
        habitats: list[object],
        species_map: dict[int, object],
    ) -> list[object]:
        self.calls.append((tiles, habitats, species_map))
        return self.updated_tiles


def test_vegetation_cover_stage_uses_injected_species_repository(
    monkeypatch,
) -> None:
    environment_repository_module = importlib.import_module(
        "app.repositories.environment_repository"
    )
    tiles = [object()]
    habitats = [object()]
    updated_tiles = [object()]
    environment_repository = _RecordingEnvironmentRepository(tiles, habitats)
    monkeypatch.setattr(
        environment_repository_module,
        "environment_repository",
        environment_repository,
    )

    species_repository_module = importlib.import_module(
        "app.repositories.species_repository"
    )
    global_repository = _RecordingSpeciesRepository([SimpleNamespace(id=99)])
    monkeypatch.setattr(
        species_repository_module,
        "species_repository",
        global_repository,
    )

    vegetation_cover_module = importlib.import_module(
        "app.services.geo.vegetation_cover"
    )
    vegetation_cover_service = _RecordingVegetationCoverService(updated_tiles)
    monkeypatch.setattr(
        vegetation_cover_module,
        "vegetation_cover_service",
        vegetation_cover_service,
    )

    mapped_species = SimpleNamespace(id=1)
    unmapped_species = SimpleNamespace(id=None)
    injected_repository = _RecordingSpeciesRepository(
        [mapped_species, unmapped_species]
    )
    context = SimpleNamespace(emit_event=lambda *_args: None)
    engine = SimpleNamespace(species_repository=injected_repository)

    asyncio.run(VegetationCoverStage().execute(context, engine))

    assert injected_repository.list_calls == 1
    assert global_repository.list_calls == 0
    assert environment_repository.list_tiles_calls == 1
    assert environment_repository.latest_habitats_calls == 1
    assert vegetation_cover_service.calls == [
        (tiles, habitats, {1: mapped_species})
    ]
    assert environment_repository.upsert_batches == [updated_tiles]
