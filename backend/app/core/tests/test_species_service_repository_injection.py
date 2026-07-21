"""Container wiring coverage for species services."""

from __future__ import annotations

from types import SimpleNamespace

from app.core.container import ServiceContainer
from app.models.config import SpeciationConfig
from app.repositories.genus_repository import GenusRepository


def test_speciation_service_uses_container_genus_repository() -> None:
    container = ServiceContainer()
    genus_repository = GenusRepository()
    container.override("model_router", object())
    container.override(
        "config_service",
        SimpleNamespace(get_speciation=lambda: SpeciationConfig()),
    )
    container.override("genus_repository", genus_repository)

    service = container.speciation_service

    assert service._genus_repository is genus_repository
