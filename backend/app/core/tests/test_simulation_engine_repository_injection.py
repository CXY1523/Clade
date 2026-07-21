"""Container wiring coverage for simulation-engine repositories."""

from __future__ import annotations

from app.core.container import ServiceContainer
from app.repositories.species_repository import SpeciesRepository


def test_simulation_engine_uses_container_species_repository() -> None:
    container = ServiceContainer()
    species_repository = SpeciesRepository()
    container.override("species_repository", species_repository)

    engine = container.simulation_engine

    assert engine.species_repository is species_repository
